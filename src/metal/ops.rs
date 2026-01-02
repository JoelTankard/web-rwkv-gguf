//! Metal-accelerated operations for RWKV inference.
//!
//! These functions provide layer-level Metal execution to avoid per-op sync overhead.
//! Each function batches multiple matmul operations into a single Metal command buffer.

use half::f16;
use std::sync::OnceLock;

use crate::context::Context;
use crate::num::Float;
use crate::tensor::{kind::ReadWrite, matrix::Matrix, ops::Activation, TensorGpu};

use super::kernels::{kernel_name_q4k_for_activation, MATMUL_Q4K_BIAS_SOURCE};
use super::{BufferBridge, MetalContext};

/// Cached Metal context for Q4K operations.
/// Avoids recompiling shaders for every layer.
static METAL_CTX: OnceLock<std::sync::Mutex<Option<MetalContext>>> = OnceLock::new();

fn get_or_init_metal_ctx() -> Option<std::sync::MutexGuard<'static, Option<MetalContext>>> {
    let mutex = METAL_CTX.get_or_init(|| {
        let ctx = MetalContext::new().ok().and_then(|mut ctx| {
            ctx.compile_library(MATMUL_Q4K_BIAS_SOURCE).ok()?;
            Some(ctx)
        });
        std::sync::Mutex::new(ctx)
    });

    let guard = mutex.lock().ok()?;
    if guard.is_some() {
        Some(guard)
    } else {
        None
    }
}

/// Sync pending Metal commands (no-op currently, kept for API compatibility).
pub fn sync_pending_metal_commands() {
    // Currently synchronous - Metal syncs after each FFN layer
}

/// Execute a Q4K FFN layer using Metal.
///
/// This batches the two matmul operations (w_k and w_v) into a single Metal command buffer,
/// avoiding the per-op sync overhead that kills performance.
///
/// FFN structure:
/// 1. ffn_k = matmul(ffn_kx, w_k) with SquaredRelu activation
/// 2. ffn_v = matmul(ffn_k, w_v) with no activation
///
/// Returns true if Metal execution succeeded, false to fall back to WebGPU.
#[allow(clippy::too_many_arguments)]
pub fn execute_metal_ffn_q4k<F: Float>(
    _context: &Context,
    w_k: &Matrix,
    w_v: &Matrix,
    input: &TensorGpu<F, ReadWrite>,
    intermediate: &TensorGpu<F, ReadWrite>,
    output: &TensorGpu<F, ReadWrite>,
) -> bool {
    // Only handle Q4K matrices
    let (w_k_data, w_v_data) = match (w_k, w_v) {
        (Matrix::Q4K { w: wk, .. }, Matrix::Q4K { w: wv, .. }) => (wk, wv),
        _ => return false,
    };

    // Try to extract Metal buffers
    let metal_w_k = match unsafe { BufferBridge::extract_from_arc(&w_k_data.buffer) } {
        Some(b) => b,
        None => return false,
    };
    let metal_w_v = match unsafe { BufferBridge::extract_from_arc(&w_v_data.buffer) } {
        Some(b) => b,
        None => return false,
    };
    let metal_input = match unsafe { BufferBridge::extract_from_arc(&input.buffer) } {
        Some(b) => b,
        None => return false,
    };
    let metal_intermediate = match unsafe { BufferBridge::extract_from_arc(&intermediate.buffer) } {
        Some(b) => b,
        None => return false,
    };
    let metal_output = match unsafe { BufferBridge::extract_from_arc(&output.buffer) } {
        Some(b) => b,
        None => return false,
    };

    // Get cached Metal context (avoids recompiling shaders every layer)
    let mut guard = match get_or_init_metal_ctx() {
        Some(g) => g,
        None => return false,
    };
    let metal_ctx = match guard.as_mut() {
        Some(ctx) => ctx,
        None => return false,
    };

    // Get pipelines for both activations (cached in context)
    let pipeline_squared_relu =
        match metal_ctx.get_pipeline(kernel_name_q4k_for_activation(Activation::SquaredRelu)) {
            Ok(p) => p.clone(),
            Err(_) => return false,
        };
    let pipeline_none =
        match metal_ctx.get_pipeline(kernel_name_q4k_for_activation(Activation::None)) {
            Ok(p) => p.clone(),
            Err(_) => return false,
        };

    // Get dimensions using TensorShape trait
    use crate::tensor::TensorShape;
    let input_shape = input.shape();
    let intermediate_shape = intermediate.shape();
    let output_shape = output.shape();
    let [k1, t, b, _] = input_shape.into();
    let [m1, _, _, _] = intermediate_shape.into();
    let [m2, _, _, _] = output_shape.into();

    // Create command buffer with both operations
    let cmd = metal_ctx.queue().new_command_buffer();

    // First matmul: input -> intermediate (with SquaredRelu)
    {
        let params: [u32; 4] = [k1 as u32, m1 as u32, t as u32, b as u32];
        let metal_params =
            metal_ctx.new_buffer_with_data(&params, metal::MTLResourceOptions::StorageModeShared);

        // Create a zero bias buffer (no bias for this op)
        let zero_bias: Vec<f16> = vec![f16::from_f32(0.0); m1];
        let metal_bias = metal_ctx
            .new_buffer_with_data(&zero_bias, metal::MTLResourceOptions::StorageModeShared);

        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_squared_relu);
        encoder.set_buffer(0, Some(&metal_w_k), 0);
        encoder.set_buffer(1, Some(&metal_input), 0);
        encoder.set_buffer(2, Some(&metal_bias), 0);
        encoder.set_buffer(3, Some(&metal_intermediate), 0);
        encoder.set_buffer(4, Some(&metal_params), 0);
        encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

        let num_groups_m = (m1 + 3) / 4;
        let thread_group_size = metal::MTLSize::new(64, 1, 1);
        let grid_size = metal::MTLSize::new(num_groups_m as u64, t as u64, b as u64);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();
    }

    // Second matmul: intermediate -> output (no activation)
    {
        let k2 = m1; // Output of first matmul is input to second
        let params: [u32; 4] = [k2 as u32, m2 as u32, t as u32, b as u32];
        let metal_params =
            metal_ctx.new_buffer_with_data(&params, metal::MTLResourceOptions::StorageModeShared);

        let zero_bias: Vec<f16> = vec![f16::from_f32(0.0); m2];
        let metal_bias = metal_ctx
            .new_buffer_with_data(&zero_bias, metal::MTLResourceOptions::StorageModeShared);

        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline_none);
        encoder.set_buffer(0, Some(&metal_w_v), 0);
        encoder.set_buffer(1, Some(&metal_intermediate), 0);
        encoder.set_buffer(2, Some(&metal_bias), 0);
        encoder.set_buffer(3, Some(&metal_output), 0);
        encoder.set_buffer(4, Some(&metal_params), 0);
        encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

        let num_groups_m = (m2 + 3) / 4;
        let thread_group_size = metal::MTLSize::new(64, 1, 1);
        let grid_size = metal::MTLSize::new(num_groups_m as u64, t as u64, b as u64);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();
    }

    // Commit and wait - we need the output before WebGPU post-ops run
    cmd.commit();
    cmd.wait_until_completed();

    log::trace!(
        "Metal Q4K FFN executed: {}x{} -> {}x{} @ {} tokens",
        k1,
        m1,
        m1,
        m2,
        t
    );

    true
}

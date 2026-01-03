//! Pure Metal V7 layer execution.
//!
//! This module executes an entire RWKV V7 layer using only Metal,
//! with a single sync point at the end of the layer.

use std::sync::OnceLock;

use half::f16 as _;
use metal::{Buffer as MetalBuffer, ComputePipelineState};

use crate::context::Context;
use crate::num::Float;
use crate::runtime::v7::Layer;
use crate::tensor::matrix::Matrix;
use crate::tensor::{kind::ReadWrite, TensorGpu, TensorGpuView, TensorShape};

use super::v7_kernels::V7_LAYER_SOURCE;
use super::{BufferBridge, MetalContext};

/// Cached Metal context with V7 layer pipelines.
static V7_METAL_CTX: OnceLock<std::sync::Mutex<Option<V7MetalContext>>> = OnceLock::new();

struct V7MetalContext {
    ctx: MetalContext,
    // Pipelines for each kernel
    layer_norm: ComputePipelineState,
    token_shift: ComputePipelineState,
    add_inplace: ComputePipelineState,
    mul_inplace: ComputePipelineState,
    blit: ComputePipelineState,
    matmul_q4k: ComputePipelineState,
    matmul_q4k_squared_relu: ComputePipelineState,
    matmul_q4k_tanh: ComputePipelineState,
    matmul_q4k_sigmoid: ComputePipelineState,
    matmul_int8: ComputePipelineState,
    matmul_int8_squared_relu: ComputePipelineState,
    matmul_fp16: ComputePipelineState,
    add_activate_sigmoid: ComputePipelineState,
    group_norm: ComputePipelineState,
    channel_mix_v7: ComputePipelineState,
    l2_norm: ComputePipelineState,
    pack_kvakk: ComputePipelineState,
    time_mix_v7: ComputePipelineState,
    time_first_v7: ComputePipelineState,
    lerp_op: ComputePipelineState,
}

impl V7MetalContext {
    fn new() -> Option<Self> {
        let mut ctx = MetalContext::new().ok()?;
        ctx.compile_library(V7_LAYER_SOURCE).ok()?;

        let layer_norm = ctx.get_pipeline("layer_norm").ok()?.clone();
        let token_shift = ctx.get_pipeline("token_shift").ok()?.clone();
        let add_inplace = ctx.get_pipeline("add_inplace").ok()?.clone();
        let mul_inplace = ctx.get_pipeline("mul_inplace").ok()?.clone();
        let blit = ctx.get_pipeline("blit").ok()?.clone();
        let matmul_q4k = ctx.get_pipeline("matmul_q4k").ok()?.clone();
        let matmul_q4k_squared_relu = ctx.get_pipeline("matmul_q4k_squared_relu").ok()?.clone();
        let matmul_q4k_tanh = ctx.get_pipeline("matmul_q4k_tanh").ok()?.clone();
        let matmul_q4k_sigmoid = ctx.get_pipeline("matmul_q4k_sigmoid").ok()?.clone();
        let matmul_int8 = ctx.get_pipeline("matmul_int8").ok()?.clone();
        let matmul_int8_squared_relu = ctx.get_pipeline("matmul_int8_squared_relu").ok()?.clone();
        let matmul_fp16 = ctx.get_pipeline("matmul_fp16").ok()?.clone();
        let add_activate_sigmoid = ctx.get_pipeline("add_activate_sigmoid").ok()?.clone();
        let group_norm = ctx.get_pipeline("group_norm").ok()?.clone();
        let channel_mix_v7 = ctx.get_pipeline("channel_mix_v7").ok()?.clone();
        let l2_norm = ctx.get_pipeline("l2_norm").ok()?.clone();
        let pack_kvakk = ctx.get_pipeline("pack_kvakk").ok()?.clone();
        let time_mix_v7 = ctx.get_pipeline("time_mix_v7").ok()?.clone();
        let time_first_v7 = ctx.get_pipeline("time_first_v7").ok()?.clone();
        let lerp_op = ctx.get_pipeline("lerp_op").ok()?.clone();

        Some(Self {
            ctx,
            layer_norm,
            token_shift,
            add_inplace,
            mul_inplace,
            blit,
            matmul_q4k,
            matmul_q4k_squared_relu,
            matmul_q4k_tanh,
            matmul_q4k_sigmoid,
            matmul_int8,
            matmul_int8_squared_relu,
            matmul_fp16,
            add_activate_sigmoid,
            group_norm,
            channel_mix_v7,
            l2_norm,
            pack_kvakk,
            time_mix_v7,
            time_first_v7,
            lerp_op,
        })
    }
}

/// Parameters for time mix operations
#[repr(C)]
struct TimeMixParams {
    s: u32, // Head size (elements per head / 4)
    h: u32, // Number of heads
    t: u32, // Number of tokens
    c: u32, // Total channels
}

/// State view parameters
#[repr(C)]
struct StateView {
    shape_x: u32,
    shape_y: u32,
    shape_z: u32,
    stride_x: u32,
    stride_y: u32,
    offset_x: u32,
    offset_y: u32,
    offset_z: u32,
}

fn get_v7_metal_ctx() -> Option<std::sync::MutexGuard<'static, Option<V7MetalContext>>> {
    let mutex = V7_METAL_CTX.get_or_init(|| std::sync::Mutex::new(V7MetalContext::new()));
    let guard = mutex.lock().ok()?;
    if guard.is_some() {
        Some(guard)
    } else {
        None
    }
}

/// Parameters for layer operations
#[repr(C)]
struct LayerParams {
    c: u32,   // Embedding dimension
    t: u32,   // Number of tokens
    b: u32,   // Batch size
    h: u32,   // Number of heads
    s: u32,   // Head size
    eps: f32, // Epsilon for normalization
}

/// Parameters for matmul operations
#[repr(C)]
struct MatmulParams {
    k: u32, // Input dimension
    m: u32, // Output dimension
    t: u32, // Number of tokens
    b: u32, // Batch size
}

/// Execute a complete V7 layer using pure Metal.
///
/// This function runs the entire layer (attention + FFN) in Metal with a single
/// sync point at the end, avoiding the per-operation WebGPU/Metal sync overhead.
///
/// Returns true if Metal execution succeeded, false to fall back to WebGPU.
#[allow(clippy::too_many_arguments)]
pub fn execute_metal_v7_layer<'a, F: Float>(
    _context: &Context,
    layer: &Layer,
    layer_index: usize,
    num_token: usize,
    head_size: usize,
    // Buffers
    x: &TensorGpu<F, ReadWrite>,
    att_x: &TensorGpu<F, ReadWrite>,
    att_state: &TensorGpuView<'a, f32>,
    ffn_x: &TensorGpu<F, ReadWrite>,
    ffn_state: &TensorGpuView<'a, f32>,
    cursors: &TensorGpu<u32, ReadWrite>,
    // Intermediate buffers for attention
    att_rx: &TensorGpu<F, ReadWrite>,
    att_wx: &TensorGpu<F, ReadWrite>,
    att_kx: &TensorGpu<F, ReadWrite>,
    att_vx: &TensorGpu<F, ReadWrite>,
    att_ax: &TensorGpu<F, ReadWrite>,
    att_gx: &TensorGpu<F, ReadWrite>,
    att_r: &TensorGpu<F, ReadWrite>,
    att_k: &TensorGpu<F, ReadWrite>,
    att_v: &TensorGpu<F, ReadWrite>,
    att_w: &TensorGpu<F, ReadWrite>,
    att_a: &TensorGpu<F, ReadWrite>,
    att_g: &TensorGpu<F, ReadWrite>,
    att_o: &TensorGpu<F, ReadWrite>,
    aux_w: &TensorGpu<F, ReadWrite>,
    aux_a: &TensorGpu<F, ReadWrite>,
    aux_g: &TensorGpu<F, ReadWrite>,
    att_kk: &TensorGpu<F, ReadWrite>,
    att_v0: &TensorGpu<F, ReadWrite>,
    att_vv: &TensorGpu<F, ReadWrite>,
    aux_v: &TensorGpu<F, ReadWrite>,
    att_n: &TensorGpu<F, ReadWrite>,
    // FFN buffers
    ffn_kx: &TensorGpu<F, ReadWrite>,
    ffn_k: &TensorGpu<F, ReadWrite>,
    ffn_v: &TensorGpu<F, ReadWrite>,
) -> bool {
    execute_metal_v7_layer_inner(
        _context,
        layer,
        layer_index,
        num_token,
        head_size,
        x,
        att_x,
        att_state,
        ffn_x,
        ffn_state,
        cursors,
        att_rx,
        att_wx,
        att_kx,
        att_vx,
        att_ax,
        att_gx,
        att_r,
        att_k,
        att_v,
        att_w,
        att_a,
        att_g,
        att_o,
        aux_w,
        aux_a,
        aux_g,
        att_kk,
        att_v0,
        att_vv,
        aux_v,
        att_n,
        ffn_kx,
        ffn_k,
        ffn_v,
    )
    .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
fn execute_metal_v7_layer_inner<'a, F: Float>(
    _context: &Context,
    layer: &Layer,
    layer_index: usize,
    num_token: usize,
    head_size: usize,
    x: &TensorGpu<F, ReadWrite>,
    att_x: &TensorGpu<F, ReadWrite>,
    att_state: &TensorGpuView<'a, f32>,
    ffn_x: &TensorGpu<F, ReadWrite>,
    ffn_state: &TensorGpuView<'a, f32>,
    cursors: &TensorGpu<u32, ReadWrite>,
    att_rx: &TensorGpu<F, ReadWrite>,
    att_wx: &TensorGpu<F, ReadWrite>,
    att_kx: &TensorGpu<F, ReadWrite>,
    att_vx: &TensorGpu<F, ReadWrite>,
    att_ax: &TensorGpu<F, ReadWrite>,
    att_gx: &TensorGpu<F, ReadWrite>,
    att_r: &TensorGpu<F, ReadWrite>,
    att_k: &TensorGpu<F, ReadWrite>,
    att_v: &TensorGpu<F, ReadWrite>,
    att_w: &TensorGpu<F, ReadWrite>,
    att_a: &TensorGpu<F, ReadWrite>,
    att_g: &TensorGpu<F, ReadWrite>,
    att_o: &TensorGpu<F, ReadWrite>,
    aux_w: &TensorGpu<F, ReadWrite>,
    aux_a: &TensorGpu<F, ReadWrite>,
    aux_g: &TensorGpu<F, ReadWrite>,
    att_kk: &TensorGpu<F, ReadWrite>,
    att_v0: &TensorGpu<F, ReadWrite>,
    att_vv: &TensorGpu<F, ReadWrite>,
    aux_v: &TensorGpu<F, ReadWrite>,
    att_n: &TensorGpu<F, ReadWrite>,
    ffn_kx: &TensorGpu<F, ReadWrite>,
    ffn_k: &TensorGpu<F, ReadWrite>,
    ffn_v: &TensorGpu<F, ReadWrite>,
) -> Option<bool> {
    // Get cached Metal context
    let mut guard = match get_v7_metal_ctx() {
        Some(g) => g,
        None => {
            log::warn!("Metal V7 context not available, falling back to WebGPU");
            return Some(false);
        }
    };
    let metal_ctx = match guard.as_mut() {
        Some(ctx) => ctx,
        None => {
            log::warn!("Metal V7 context is None, falling back to WebGPU");
            return Some(false);
        }
    };

    // Check if all matrices are a supported quantization type (Int8, Q4K, or Fp16)
    let is_supported = |m: &Matrix| {
        matches!(
            m,
            Matrix::Int8 { .. } | Matrix::Q4K { .. } | Matrix::Fp16(_)
        )
    };

    let all_supported = is_supported(&layer.att.w_r)
        && is_supported(&layer.att.w_k)
        && is_supported(&layer.att.w_v)
        && is_supported(&layer.att.w_o)
        && is_supported(&layer.ffn.w_k)
        && is_supported(&layer.ffn.w_v);

    if !all_supported {
        return Some(false);
    }

    // Extract Metal buffers
    let metal_x = unsafe { BufferBridge::extract_from_arc(&x.buffer)? };
    let metal_att_x = unsafe { BufferBridge::extract_from_arc(&att_x.buffer)? };
    let metal_cursors = unsafe { BufferBridge::extract_from_arc(&cursors.buffer)? };
    let metal_att_state = unsafe { BufferBridge::extract_from_arc(&att_state.tensor().buffer)? };
    let metal_ffn_x = unsafe { BufferBridge::extract_from_arc(&ffn_x.buffer)? };
    let metal_ffn_state = unsafe { BufferBridge::extract_from_arc(&ffn_state.tensor().buffer)? };

    // Extract intermediate buffers
    let metal_att_rx = unsafe { BufferBridge::extract_from_arc(&att_rx.buffer)? };
    let metal_att_wx = unsafe { BufferBridge::extract_from_arc(&att_wx.buffer)? };
    let metal_att_kx = unsafe { BufferBridge::extract_from_arc(&att_kx.buffer)? };
    let metal_att_vx = unsafe { BufferBridge::extract_from_arc(&att_vx.buffer)? };
    let metal_att_ax = unsafe { BufferBridge::extract_from_arc(&att_ax.buffer)? };
    let metal_att_gx = unsafe { BufferBridge::extract_from_arc(&att_gx.buffer)? };
    let metal_att_r = unsafe { BufferBridge::extract_from_arc(&att_r.buffer)? };
    let metal_att_k = unsafe { BufferBridge::extract_from_arc(&att_k.buffer)? };
    let metal_att_v = unsafe { BufferBridge::extract_from_arc(&att_v.buffer)? };
    let metal_att_w = unsafe { BufferBridge::extract_from_arc(&att_w.buffer)? };
    let metal_att_a = unsafe { BufferBridge::extract_from_arc(&att_a.buffer)? };
    let metal_att_g = unsafe { BufferBridge::extract_from_arc(&att_g.buffer)? };
    let metal_att_o = unsafe { BufferBridge::extract_from_arc(&att_o.buffer)? };
    let metal_aux_w = unsafe { BufferBridge::extract_from_arc(&aux_w.buffer)? };
    let metal_aux_a = unsafe { BufferBridge::extract_from_arc(&aux_a.buffer)? };
    let metal_aux_g = unsafe { BufferBridge::extract_from_arc(&aux_g.buffer)? };
    let metal_att_kk = unsafe { BufferBridge::extract_from_arc(&att_kk.buffer)? };
    let metal_att_v0 = unsafe { BufferBridge::extract_from_arc(&att_v0.buffer)? };
    let metal_att_vv = unsafe { BufferBridge::extract_from_arc(&att_vv.buffer)? };
    let metal_aux_v = unsafe { BufferBridge::extract_from_arc(&aux_v.buffer)? };
    let metal_att_n = unsafe { BufferBridge::extract_from_arc(&att_n.buffer)? };
    let metal_ffn_kx = unsafe { BufferBridge::extract_from_arc(&ffn_kx.buffer)? };
    let metal_ffn_k = unsafe { BufferBridge::extract_from_arc(&ffn_k.buffer)? };
    let metal_ffn_v = unsafe { BufferBridge::extract_from_arc(&ffn_v.buffer)? };

    // Extract layer weight buffers
    let metal_att_ln_w = unsafe { BufferBridge::extract_from_arc(&layer.att_ln.w.buffer)? };
    let metal_att_ln_b = unsafe { BufferBridge::extract_from_arc(&layer.att_ln.b.buffer)? };
    let metal_ffn_ln_w = unsafe { BufferBridge::extract_from_arc(&layer.ffn_ln.w.buffer)? };
    let metal_ffn_ln_b = unsafe { BufferBridge::extract_from_arc(&layer.ffn_ln.b.buffer)? };

    // Extract token shift mix weights
    let metal_x_r = unsafe { BufferBridge::extract_from_arc(&layer.att.x_r.buffer)? };
    let metal_x_w = unsafe { BufferBridge::extract_from_arc(&layer.att.x_w.buffer)? };
    let metal_x_k = unsafe { BufferBridge::extract_from_arc(&layer.att.x_k.buffer)? };
    let metal_x_v = unsafe { BufferBridge::extract_from_arc(&layer.att.x_v.buffer)? };
    let metal_x_a = unsafe { BufferBridge::extract_from_arc(&layer.att.x_a.buffer)? };
    let metal_x_g = unsafe { BufferBridge::extract_from_arc(&layer.att.x_g.buffer)? };
    let metal_ffn_x_k = unsafe { BufferBridge::extract_from_arc(&layer.ffn.x_k.buffer)? };

    // Extract Q4K weight matrices
    let (metal_w_r, w_r_k, w_r_m) = extract_q4k_matrix(&layer.att.w_r)?;
    let (metal_w_k, w_k_k, w_k_m) = extract_q4k_matrix(&layer.att.w_k)?;
    let (metal_w_v, w_v_k, w_v_m) = extract_q4k_matrix(&layer.att.w_v)?;
    let (metal_w_o, w_o_k, w_o_m) = extract_q4k_matrix(&layer.att.w_o)?;
    let (metal_w1, w1_k, w1_m) = extract_q4k_matrix(&layer.att.w1)?;
    let (metal_w2, w2_k, w2_m) = extract_q4k_matrix(&layer.att.w2)?;
    let (metal_a1, a1_k, a1_m) = extract_q4k_matrix(&layer.att.a1)?;
    let (metal_a2, a2_k, a2_m) = extract_q4k_matrix(&layer.att.a2)?;
    let (metal_g1, g1_k, g1_m) = extract_q4k_matrix(&layer.att.g1)?;
    let (metal_g2, g2_k, g2_m) = extract_q4k_matrix(&layer.att.g2)?;
    let (metal_ffn_w_k, ffn_w_k_k, ffn_w_k_m) = extract_q4k_matrix(&layer.ffn.w_k)?;
    let (metal_ffn_w_v, ffn_w_v_k, ffn_w_v_m) = extract_q4k_matrix(&layer.ffn.w_v)?;

    // Get dimensions
    let [c, _, _, _] = x.shape().into();
    let num_head = c / head_size;

    // Create single command buffer for entire layer
    let cmd = metal_ctx.ctx.queue().new_command_buffer();

    // Layer parameters
    let layer_params = LayerParams {
        c: c as u32,
        t: num_token as u32,
        b: 1,
        h: num_head as u32,
        s: head_size as u32,
        eps: 1e-5,
    };
    let layer_params_slice = unsafe {
        std::slice::from_raw_parts(
            &layer_params as *const LayerParams as *const u8,
            std::mem::size_of::<LayerParams>(),
        )
    };
    let metal_layer_params = metal_ctx.ctx.new_buffer_with_data(
        layer_params_slice,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // ========================================================================
    // ATTENTION BLOCK
    // ========================================================================

    // 1. Blit x -> att_x
    encode_blit(
        cmd,
        &metal_ctx.blit,
        &metal_x,
        &metal_att_x,
        (c * num_token) as u32,
    );

    // 2. Layer norm on att_x
    encode_layer_norm(
        cmd,
        &metal_ctx.layer_norm,
        &metal_att_ln_w,
        &metal_att_ln_b,
        &metal_att_x,
        &metal_layer_params,
        c,
        num_token,
    );

    // 3. Token shifts (6 of them)
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_r,
        &metal_att_state,
        &metal_att_x,
        &metal_att_rx,
        &metal_layer_params,
        c,
        num_token,
    );
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_w,
        &metal_att_state,
        &metal_att_x,
        &metal_att_wx,
        &metal_layer_params,
        c,
        num_token,
    );
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_k,
        &metal_att_state,
        &metal_att_x,
        &metal_att_kx,
        &metal_layer_params,
        c,
        num_token,
    );
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_v,
        &metal_att_state,
        &metal_att_x,
        &metal_att_vx,
        &metal_layer_params,
        c,
        num_token,
    );
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_a,
        &metal_att_state,
        &metal_att_x,
        &metal_att_ax,
        &metal_layer_params,
        c,
        num_token,
    );
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_x_g,
        &metal_att_state,
        &metal_att_x,
        &metal_att_gx,
        &metal_layer_params,
        c,
        num_token,
    );

    // 4. Attention matmuls: w_r, w_k, w_v (no activation)
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_w_r,
        &metal_att_rx,
        &metal_att_r,
        w_r_k,
        w_r_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_w_k,
        &metal_att_kx,
        &metal_att_k,
        w_k_k,
        w_k_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_w_v,
        &metal_att_vx,
        &metal_att_v,
        w_v_k,
        w_v_m,
        num_token,
        &metal_ctx.ctx,
    );

    // 5. Adapt matmuls: w1->w2 (tanh), a1->a2, g1->g2 (sigmoid)
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k_tanh,
        &metal_w1,
        &metal_att_wx,
        &metal_aux_w,
        w1_k,
        w1_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_w2,
        &metal_aux_w,
        &metal_att_w,
        w2_k,
        w2_m,
        num_token,
        &metal_ctx.ctx,
    );
    // Add w0 to att_w
    let metal_w0 = unsafe { BufferBridge::extract_from_arc(&layer.att.w0.buffer)? };
    encode_add_inplace(
        cmd,
        &metal_ctx.add_inplace,
        &metal_w0,
        &metal_att_w,
        (c * num_token) as u32,
    );

    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_a1,
        &metal_att_ax,
        &metal_aux_a,
        a1_k,
        a1_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_a2,
        &metal_aux_a,
        &metal_att_a,
        a2_k,
        a2_m,
        num_token,
        &metal_ctx.ctx,
    );
    // Add a0 and apply sigmoid
    let metal_a0 = unsafe { BufferBridge::extract_from_arc(&layer.att.a0.buffer)? };
    encode_add_activate_sigmoid(
        cmd,
        &metal_ctx.add_activate_sigmoid,
        &metal_a0,
        &metal_att_a,
        (c * num_token) as u32,
    );

    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k_sigmoid,
        &metal_g1,
        &metal_att_gx,
        &metal_aux_g,
        g1_k,
        g1_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_g2,
        &metal_aux_g,
        &metal_att_g,
        g2_k,
        g2_m,
        num_token,
        &metal_ctx.ctx,
    );

    // 6. Control operations (blit k->kk, mul k_k, l2_norm, control_k)
    encode_blit(
        cmd,
        &metal_ctx.blit,
        &metal_att_k,
        &metal_att_kk,
        (c * num_token) as u32,
    );
    let metal_k_k = unsafe { BufferBridge::extract_from_arc(&layer.att.k_k.buffer)? };
    encode_mul_inplace(
        cmd,
        &metal_ctx.mul_inplace,
        &metal_k_k,
        &metal_att_kk,
        (c * num_token) as u32,
    );
    // L2 norm on att_kk (per head)
    encode_l2_norm(
        cmd,
        &metal_ctx.l2_norm,
        &metal_att_kk,
        &metal_layer_params,
        num_head,
        head_size,
        num_token,
    );

    // 7. Value residual (layer 0 just blits, others need v1/v2 matmuls + lerp)
    if layer_index == 0 {
        encode_blit(
            cmd,
            &metal_ctx.blit,
            &metal_att_v,
            &metal_att_v0,
            (c * num_token) as u32,
        );
    } else {
        // v1 matmul: att_vx -> aux_v
        let (metal_v1, v1_k, v1_m) = extract_q4k_matrix(&layer.att.v1)?;
        encode_matmul_q4k(
            cmd,
            &metal_ctx.matmul_q4k,
            &metal_v1,
            &metal_att_vx,
            &metal_aux_v,
            v1_k,
            v1_m,
            num_token,
            &metal_ctx.ctx,
        );

        // v2 matmul: aux_v -> att_vv
        let (metal_v2, v2_k, v2_m) = extract_q4k_matrix(&layer.att.v2)?;
        encode_matmul_q4k(
            cmd,
            &metal_ctx.matmul_q4k,
            &metal_v2,
            &metal_aux_v,
            &metal_att_vv,
            v2_k,
            v2_m,
            num_token,
            &metal_ctx.ctx,
        );

        // Add v0 bias and apply sigmoid to att_vv
        let metal_v0_bias = unsafe { BufferBridge::extract_from_arc(&layer.att.v0.buffer)? };
        encode_add_activate_sigmoid(
            cmd,
            &metal_ctx.add_activate_sigmoid,
            &metal_v0_bias,
            &metal_att_vv,
            (c * num_token) as u32,
        );

        // Lerp: att_v0 = mix(att_v0, att_v, att_vv) with reversed=true
        // This computes: att_v0 = att_v0 * (1 - att_vv) + att_v * att_vv
        // First blit att_v to att_v0, then lerp
        encode_blit(
            cmd,
            &metal_ctx.blit,
            &metal_att_v,
            &metal_att_v0,
            (c * num_token) as u32,
        );
        encode_lerp(
            cmd,
            &metal_ctx.lerp_op,
            &metal_att_vv,
            &metal_att_v,
            &metal_att_v0,
            (c * num_token) as u32,
            true, // reversed
        );
    }

    // 8. Pack k, v, a, kk into n tensor
    encode_pack_kvakk(
        cmd,
        &metal_ctx.pack_kvakk,
        &metal_att_k,
        &metal_att_v0,
        &metal_att_a,
        &metal_att_kk,
        &metal_att_n,
        (c * num_token) as u32,
    );

    // 9. Time mix V7
    let metal_r_k = unsafe { BufferBridge::extract_from_arc(&layer.att.r_k.buffer)? };
    encode_time_mix_v7(
        cmd,
        &metal_ctx.time_mix_v7,
        &metal_cursors,
        &metal_att_state,
        &metal_att_r,
        &metal_att_w,
        &metal_att_n,
        &metal_att_x,
        num_head,
        head_size,
        num_token,
        &metal_ctx.ctx,
    );

    // 10. Group norm
    let metal_gn_w = unsafe { BufferBridge::extract_from_arc(&layer.att.gn.w.buffer)? };
    let metal_gn_b = unsafe { BufferBridge::extract_from_arc(&layer.att.gn.b.buffer)? };
    encode_group_norm(
        cmd,
        &metal_ctx.group_norm,
        &metal_gn_w,
        &metal_gn_b,
        &metal_att_x,
        &metal_layer_params,
        num_head,
        head_size,
        num_token,
    );

    // 11. Time first V7 (adds the "first" contribution)
    encode_time_first_v7(
        cmd,
        &metal_ctx.time_first_v7,
        &metal_r_k,
        &metal_att_r,
        &metal_att_n,
        &metal_att_x,
        num_head,
        head_size,
        num_token,
        &metal_ctx.ctx,
    );

    // 12. Gate: mul att_g * att_x
    encode_mul_inplace(
        cmd,
        &metal_ctx.mul_inplace,
        &metal_att_g,
        &metal_att_x,
        (c * num_token) as u32,
    );

    // 11. Output projection: w_o matmul
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_w_o,
        &metal_att_x,
        &metal_att_o,
        w_o_k,
        w_o_m,
        num_token,
        &metal_ctx.ctx,
    );

    // 12. Residual: add att_o to x
    encode_add_inplace(
        cmd,
        &metal_ctx.add_inplace,
        &metal_att_o,
        &metal_x,
        (c * num_token) as u32,
    );

    // ========================================================================
    // FFN BLOCK
    // ========================================================================

    // 13. Blit x -> ffn_x
    encode_blit(
        cmd,
        &metal_ctx.blit,
        &metal_x,
        &metal_ffn_x,
        (c * num_token) as u32,
    );

    // 14. Layer norm on ffn_x
    encode_layer_norm(
        cmd,
        &metal_ctx.layer_norm,
        &metal_ffn_ln_w,
        &metal_ffn_ln_b,
        &metal_ffn_x,
        &metal_layer_params,
        c,
        num_token,
    );

    // 15. Token shift for FFN
    encode_token_shift(
        cmd,
        &metal_ctx.token_shift,
        &metal_cursors,
        &metal_ffn_x_k,
        &metal_ffn_state,
        &metal_ffn_x,
        &metal_ffn_kx,
        &metal_layer_params,
        c,
        num_token,
    );

    // 16. FFN matmuls: w_k (squared relu), w_v (none)
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k_squared_relu,
        &metal_ffn_w_k,
        &metal_ffn_kx,
        &metal_ffn_k,
        ffn_w_k_k,
        ffn_w_k_m,
        num_token,
        &metal_ctx.ctx,
    );
    encode_matmul_q4k(
        cmd,
        &metal_ctx.matmul_q4k,
        &metal_ffn_w_v,
        &metal_ffn_k,
        &metal_ffn_v,
        ffn_w_v_k,
        ffn_w_v_m,
        num_token,
        &metal_ctx.ctx,
    );

    // 17. Channel mix V7
    encode_channel_mix_v7(
        cmd,
        &metal_ctx.channel_mix_v7,
        &metal_cursors,
        &metal_ffn_state,
        &metal_ffn_v,
        &metal_ffn_x,
        &metal_layer_params,
        c,
        num_token,
    );

    // 18. Residual: add ffn_x to x
    encode_add_inplace(
        cmd,
        &metal_ctx.add_inplace,
        &metal_ffn_x,
        &metal_x,
        (c * num_token) as u32,
    );

    // ========================================================================
    // SINGLE SYNC POINT
    // ========================================================================
    cmd.commit();
    cmd.wait_until_completed();

    log::debug!(
        "Metal V7 layer {} executed: {} tokens",
        layer_index,
        num_token
    );
    Some(true)
}

/// Extracted matrix data for Metal execution
enum ExtractedMatrix {
    Q4K {
        w: MetalBuffer,
        k: usize,
        m: usize,
    },
    Int8 {
        w: MetalBuffer,
        minmax: MetalBuffer,
        k: usize,
        m: usize,
    },
    Fp16 {
        w: MetalBuffer,
        k: usize,
        m: usize,
    },
}

fn extract_matrix(matrix: &Matrix) -> Option<ExtractedMatrix> {
    match matrix {
        Matrix::Q4K { w, s } => {
            let metal_buf = unsafe { BufferBridge::extract_from_arc(&w.buffer)? };
            let [k, m, _, _] = s.shape().into();
            Some(ExtractedMatrix::Q4K { w: metal_buf, k, m })
        }
        Matrix::Int8 { w, m: minmax } => {
            let metal_w = unsafe { BufferBridge::extract_from_arc(&w.buffer)? };
            let metal_mm = unsafe { BufferBridge::extract_from_arc(&minmax.buffer)? };
            let [k, m_dim, _, _] = w.shape().into();
            Some(ExtractedMatrix::Int8 {
                w: metal_w,
                minmax: metal_mm,
                k,
                m: m_dim,
            })
        }
        Matrix::Fp16(w) => {
            let metal_buf = unsafe { BufferBridge::extract_from_arc(&w.buffer)? };
            let [k, m, _, _] = w.shape().into();
            Some(ExtractedMatrix::Fp16 { w: metal_buf, k, m })
        }
        _ => None,
    }
}

// Legacy helper for backward compatibility
fn extract_q4k_matrix(matrix: &Matrix) -> Option<(MetalBuffer, usize, usize)> {
    match extract_matrix(matrix)? {
        ExtractedMatrix::Q4K { w, k, m } => Some((w, k, m)),
        ExtractedMatrix::Int8 { w, k, m, .. } => Some((w, k, m)),
        ExtractedMatrix::Fp16 { w, k, m } => Some((w, k, m)),
    }
}

/// Dispatch matmul to the appropriate kernel based on matrix type
fn encode_matmul_dispatch(
    cmd: &metal::CommandBufferRef,
    matrix: &Matrix,
    input: &MetalBuffer,
    output: &MetalBuffer,
    num_token: usize,
    ctx: &V7MetalContext,
) -> Option<()> {
    match extract_matrix(matrix)? {
        ExtractedMatrix::Q4K { w, k, m } => {
            encode_matmul_q4k(
                cmd,
                &ctx.matmul_q4k,
                &w,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
        }
        ExtractedMatrix::Int8 { w, minmax, k, m } => {
            encode_matmul_int8(
                cmd,
                &ctx.matmul_int8,
                &w,
                &minmax,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
        }
        ExtractedMatrix::Fp16 { w, k, m } => {
            encode_matmul_fp16(
                cmd,
                &ctx.matmul_fp16,
                &w,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
        }
    }
    Some(())
}

/// Dispatch matmul with squared relu activation
fn encode_matmul_squared_relu_dispatch(
    cmd: &metal::CommandBufferRef,
    matrix: &Matrix,
    input: &MetalBuffer,
    output: &MetalBuffer,
    num_token: usize,
    ctx: &V7MetalContext,
) -> Option<()> {
    match extract_matrix(matrix)? {
        ExtractedMatrix::Q4K { w, k, m } => {
            encode_matmul_q4k(
                cmd,
                &ctx.matmul_q4k_squared_relu,
                &w,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
        }
        ExtractedMatrix::Int8 { w, minmax, k, m } => {
            encode_matmul_int8(
                cmd,
                &ctx.matmul_int8_squared_relu,
                &w,
                &minmax,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
        }
        ExtractedMatrix::Fp16 { w, k, m } => {
            // Fp16 doesn't have squared relu variant yet, use regular and apply activation separately
            encode_matmul_fp16(
                cmd,
                &ctx.matmul_fp16,
                &w,
                input,
                output,
                k,
                m,
                num_token,
                &ctx.ctx,
            );
            // TODO: Add squared relu activation pass
        }
    }
    Some(())
}

// Encoding helper functions
fn encode_blit(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    src: &MetalBuffer,
    dst: &MetalBuffer,
    count: u32,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(src), 0);
    encoder.set_buffer(1, Some(dst), 0);
    encoder.set_bytes(2, 4, &count as *const u32 as *const _);
    let threads = metal::MTLSize::new(((count + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_add_inplace(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    src: &MetalBuffer,
    dst: &MetalBuffer,
    count: u32,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(src), 0);
    encoder.set_buffer(1, Some(dst), 0);
    encoder.set_bytes(2, 4, &count as *const u32 as *const _);
    let threads = metal::MTLSize::new(((count + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_mul_inplace(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    src: &MetalBuffer,
    dst: &MetalBuffer,
    count: u32,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(src), 0);
    encoder.set_buffer(1, Some(dst), 0);
    encoder.set_bytes(2, 4, &count as *const u32 as *const _);
    let threads = metal::MTLSize::new(((count + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_add_activate_sigmoid(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    bias: &MetalBuffer,
    x: &MetalBuffer,
    count: u32,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(bias), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_bytes(2, 4, &count as *const u32 as *const _);
    let threads = metal::MTLSize::new(((count + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_layer_norm(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    w: &MetalBuffer,
    b: &MetalBuffer,
    x: &MetalBuffer,
    params: &MetalBuffer,
    _c: usize,
    num_token: usize,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(x), 0);
    encoder.set_buffer(3, Some(params), 0);
    encoder.set_threadgroup_memory_length(0, 256 * 4);
    let threads = metal::MTLSize::new(1, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_token_shift(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    cursors: &MetalBuffer,
    mix: &MetalBuffer,
    state: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    params: &MetalBuffer,
    c: usize,
    num_token: usize,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(cursors), 0);
    encoder.set_buffer(1, Some(mix), 0);
    encoder.set_buffer(2, Some(state), 0);
    encoder.set_buffer(3, Some(input), 0);
    encoder.set_buffer(4, Some(output), 0);
    encoder.set_buffer(5, Some(params), 0);
    let threads = metal::MTLSize::new(((c + 255) / 256) as u64, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_matmul_q4k(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    matrix: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    k: usize,
    m: usize,
    num_token: usize,
    ctx: &MetalContext,
) {
    let params = MatmulParams {
        k: k as u32,
        m: m as u32,
        t: num_token as u32,
        b: 1,
    };
    let params_slice = unsafe {
        std::slice::from_raw_parts(
            &params as *const MatmulParams as *const u8,
            std::mem::size_of::<MatmulParams>(),
        )
    };
    let metal_params =
        ctx.new_buffer_with_data(params_slice, metal::MTLResourceOptions::StorageModeShared);

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(matrix), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_buffer(3, Some(&metal_params), 0);
    encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

    let num_groups_m = (m + 3) / 4;
    let thread_group = metal::MTLSize::new(64, 1, 1);
    let grid = metal::MTLSize::new(num_groups_m as u64, num_token as u64, 1);
    encoder.dispatch_thread_groups(grid, thread_group);
    encoder.end_encoding();
}

fn encode_matmul_int8(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    matrix: &MetalBuffer,
    minmax: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    k: usize,
    m: usize,
    num_token: usize,
    ctx: &MetalContext,
) {
    let params = MatmulParams {
        k: k as u32,
        m: m as u32,
        t: num_token as u32,
        b: 1,
    };
    let params_slice = unsafe {
        std::slice::from_raw_parts(
            &params as *const MatmulParams as *const u8,
            std::mem::size_of::<MatmulParams>(),
        )
    };
    let metal_params =
        ctx.new_buffer_with_data(params_slice, metal::MTLResourceOptions::StorageModeShared);

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(matrix), 0);
    encoder.set_buffer(1, Some(minmax), 0);
    encoder.set_buffer(2, Some(input), 0);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_buffer(4, Some(&metal_params), 0);
    encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

    let num_groups_m = (m + 3) / 4;
    let thread_group = metal::MTLSize::new(64, 1, 1);
    let grid = metal::MTLSize::new(num_groups_m as u64, num_token as u64, 1);
    encoder.dispatch_thread_groups(grid, thread_group);
    encoder.end_encoding();
}

fn encode_matmul_fp16(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    matrix: &MetalBuffer,
    input: &MetalBuffer,
    output: &MetalBuffer,
    k: usize,
    m: usize,
    num_token: usize,
    ctx: &MetalContext,
) {
    let params = MatmulParams {
        k: k as u32,
        m: m as u32,
        t: num_token as u32,
        b: 1,
    };
    let params_slice = unsafe {
        std::slice::from_raw_parts(
            &params as *const MatmulParams as *const u8,
            std::mem::size_of::<MatmulParams>(),
        )
    };
    let metal_params =
        ctx.new_buffer_with_data(params_slice, metal::MTLResourceOptions::StorageModeShared);

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(matrix), 0);
    encoder.set_buffer(1, Some(input), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_buffer(3, Some(&metal_params), 0);
    encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

    let num_groups_m = (m + 3) / 4;
    let thread_group = metal::MTLSize::new(64, 1, 1);
    let grid = metal::MTLSize::new(num_groups_m as u64, num_token as u64, 1);
    encoder.dispatch_thread_groups(grid, thread_group);
    encoder.end_encoding();
}

fn encode_group_norm(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    w: &MetalBuffer,
    b: &MetalBuffer,
    x: &MetalBuffer,
    params: &MetalBuffer,
    num_head: usize,
    _head_size: usize,
    num_token: usize,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(b), 0);
    encoder.set_buffer(2, Some(x), 0);
    encoder.set_buffer(3, Some(params), 0);
    encoder.set_threadgroup_memory_length(0, 64 * 4);
    let threads = metal::MTLSize::new(num_head as u64, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(64, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_l2_norm(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    x: &MetalBuffer,
    params: &MetalBuffer,
    num_head: usize,
    _head_size: usize,
    num_token: usize,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(x), 0);
    encoder.set_buffer(1, Some(params), 0);
    encoder.set_threadgroup_memory_length(0, 64 * 4);
    let threads = metal::MTLSize::new(num_head as u64, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(64, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_channel_mix_v7(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    cursors: &MetalBuffer,
    state: &MetalBuffer,
    v: &MetalBuffer,
    x: &MetalBuffer,
    params: &MetalBuffer,
    c: usize,
    num_token: usize,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(cursors), 0);
    encoder.set_buffer(1, Some(state), 0);
    encoder.set_buffer(2, Some(v), 0);
    encoder.set_buffer(3, Some(x), 0);
    encoder.set_buffer(4, Some(params), 0);
    let threads = metal::MTLSize::new(((c + 255) / 256) as u64, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_pack_kvakk(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    k: &MetalBuffer,
    v: &MetalBuffer,
    a: &MetalBuffer,
    kk: &MetalBuffer,
    n: &MetalBuffer,
    stride: u32,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(k), 0);
    encoder.set_buffer(1, Some(v), 0);
    encoder.set_buffer(2, Some(a), 0);
    encoder.set_buffer(3, Some(kk), 0);
    encoder.set_buffer(4, Some(n), 0);
    encoder.set_bytes(5, 4, &stride as *const u32 as *const _);
    let threads = metal::MTLSize::new(((stride + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

#[allow(clippy::too_many_arguments)]
fn encode_time_mix_v7(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    cursors: &MetalBuffer,
    state: &MetalBuffer,
    r: &MetalBuffer,
    w: &MetalBuffer,
    n: &MetalBuffer,
    x: &MetalBuffer,
    num_head: usize,
    head_size: usize,
    num_token: usize,
    ctx: &MetalContext,
) {
    let stride = num_head * head_size / 4; // vec4 elements per token
    let block_size = stride; // One thread per vec4 element

    // Create time mix params
    let params = TimeMixParams {
        s: (head_size / 4) as u32,
        h: num_head as u32,
        t: num_token as u32,
        c: (num_head * head_size) as u32,
    };
    let params_slice = unsafe {
        std::slice::from_raw_parts(
            &params as *const TimeMixParams as *const u8,
            std::mem::size_of::<TimeMixParams>(),
        )
    };
    let metal_params =
        ctx.new_buffer_with_data(params_slice, metal::MTLResourceOptions::StorageModeShared);

    // Create state view params (simplified - assumes contiguous state)
    let state_view = StateView {
        shape_x: (num_head * head_size) as u32,
        shape_y: (head_size + 1) as u32,
        shape_z: 1, // batch size
        stride_x: 1,
        stride_y: (num_head * head_size) as u32,
        offset_x: 0,
        offset_y: 0,
        offset_z: 0,
    };
    let view_slice = unsafe {
        std::slice::from_raw_parts(
            &state_view as *const StateView as *const u8,
            std::mem::size_of::<StateView>(),
        )
    };
    let metal_view =
        ctx.new_buffer_with_data(view_slice, metal::MTLResourceOptions::StorageModeShared);

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(cursors), 0);
    encoder.set_buffer(1, Some(state), 0);
    encoder.set_buffer(2, Some(r), 0);
    encoder.set_buffer(3, Some(w), 0);
    encoder.set_buffer(4, Some(n), 0);
    encoder.set_buffer(5, Some(x), 0);
    encoder.set_buffer(6, Some(&metal_params), 0);
    encoder.set_buffer(7, Some(&metal_view), 0);

    // Threadgroup memory for shared arrays (5 arrays of block_size float4s)
    let shared_mem_size = block_size * 16; // 16 bytes per float4
    encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);
    encoder.set_threadgroup_memory_length(1, shared_mem_size as u64);
    encoder.set_threadgroup_memory_length(2, shared_mem_size as u64);
    encoder.set_threadgroup_memory_length(3, shared_mem_size as u64);
    encoder.set_threadgroup_memory_length(4, shared_mem_size as u64);

    let threads = metal::MTLSize::new(((stride + 31) / 32) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

#[allow(clippy::too_many_arguments)]
fn encode_time_first_v7(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    u: &MetalBuffer,
    r: &MetalBuffer,
    n: &MetalBuffer,
    x: &MetalBuffer,
    num_head: usize,
    head_size: usize,
    num_token: usize,
    ctx: &MetalContext,
) {
    let stride = num_head * head_size / 4;
    let block_size = stride;

    let params = TimeMixParams {
        s: (head_size / 4) as u32,
        h: num_head as u32,
        t: num_token as u32,
        c: (num_head * head_size) as u32,
    };
    let params_slice = unsafe {
        std::slice::from_raw_parts(
            &params as *const TimeMixParams as *const u8,
            std::mem::size_of::<TimeMixParams>(),
        )
    };
    let metal_params =
        ctx.new_buffer_with_data(params_slice, metal::MTLResourceOptions::StorageModeShared);

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(u), 0);
    encoder.set_buffer(1, Some(r), 0);
    encoder.set_buffer(2, Some(n), 0);
    encoder.set_buffer(3, Some(x), 0);
    encoder.set_buffer(4, Some(&metal_params), 0);

    // Threadgroup memory for shared_x
    let shared_mem_size = block_size * 16;
    encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

    let threads = metal::MTLSize::new(((stride + 31) / 32) as u64, num_token as u64, 1);
    let thread_group = metal::MTLSize::new(block_size as u64, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

fn encode_lerp(
    cmd: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    f: &MetalBuffer,
    x: &MetalBuffer,
    y: &MetalBuffer,
    count: u32,
    reversed: bool,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(f), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    encoder.set_bytes(3, 4, &count as *const u32 as *const _);
    let reversed_u32: u32 = if reversed { 1 } else { 0 };
    encoder.set_bytes(4, 4, &reversed_u32 as *const u32 as *const _);
    let threads = metal::MTLSize::new(((count + 255) / 256) as u64, 1, 1);
    let thread_group = metal::MTLSize::new(256, 1, 1);
    encoder.dispatch_thread_groups(threads, thread_group);
    encoder.end_encoding();
}

//! Metal operations for quantized matrix multiplication and Phase 1 simple ops.
//!
//! This module provides two API styles:
//! 1. `encode_*` - Creates a new compute encoder per operation (legacy, slower)
//! 2. `enc_*` - Uses an existing encoder reference (fast, for batched execution)
//!
//! For best performance, use the `enc_*` variants with a shared encoder across
//! multiple operations within a layer.

use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef, MTLSize};

use super::context::MetalContext;

pub struct MetalOps;

impl MetalOps {
    // ========================================================================
    // Phase 1: Simple Element-wise Operations
    // ========================================================================

    /// Add two fp16 tensors: output = input + output (in-place)
    pub fn encode_add_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Add f32 tensor to fp16 tensor: output = input + output
    pub fn encode_add_f32_to_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_f32_to_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Multiply two fp16 tensors: output = input * output (in-place)
    pub fn encode_mul_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.mul_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Copy fp16 tensor: dst = src
    pub fn encode_blit_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        src: &Buffer,
        dst: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.blit_fp16_pipeline());
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Affine transform: x = scale * x + bias (in-place)
    pub fn encode_affine_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        scale: f32,
        bias: f32,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.affine_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<f32>() as u64,
            &scale as *const f32 as *const _,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<f32>() as u64,
            &bias as *const f32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 1: Layer Normalization
    // ========================================================================

    /// Layer normalization with affine transform
    /// Each threadgroup processes one token
    pub fn encode_layer_norm_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.layer_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // One threadgroup per token, 256 threads per threadgroup (8 simdgroups)
        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    /// Group normalization (per-head normalization)
    pub fn encode_group_norm_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        head_size: u32,
        num_heads: u32,
        tokens: u32,
        eps: f32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.group_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &head_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &num_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // One threadgroup per (token, head) pair, 32 threads per threadgroup (1 simdgroup)
        encoder.dispatch_thread_groups(
            MTLSize::new(tokens as u64, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 1: Token Shift
    // ========================================================================

    /// Token shift operation for RWKV
    pub fn encode_token_shift_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        cursors: &Buffer,
        time_mix: &Buffer,
        state: &Buffer,
        input: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        time_mix_stride: u32, // 0 if shared, channels if per-token
        reversed: bool,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = if reversed {
            ctx.token_shift_reversed_fp16_pipeline()
        } else {
            ctx.token_shift_fp16_pipeline()
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(time_mix), 0);
        encoder.set_buffer(2, Some(state), 0);
        encoder.set_buffer(3, Some(input), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &time_mix_stride as *const u32 as *const _,
        );

        // 2D dispatch: channels x tokens
        let threads_x = 256u64;
        let num_tgs_x = ((channels as u64) + threads_x - 1) / threads_x;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs_x, tokens as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 1: Lerp
    // ========================================================================

    /// Linear interpolation: output = a + factor * (b - a)
    pub fn encode_lerp_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        factor: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.lerp_fp16_pipeline());
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(factor), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 2: FP16 Matrix-Vector Multiplication
    // ========================================================================

    /// FP16 matrix-vector multiplication
    /// output[M] = input[K] @ weights[K, M]
    pub fn encode_matmul_vec_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        // 4 simdgroups per threadgroup, each handles one row
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// FP16 matmul with tanh activation
    pub fn encode_matmul_vec_fp16_tanh(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_tanh_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// FP16 matmul with squared ReLU activation
    pub fn encode_matmul_vec_fp16_squared_relu(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_squared_relu_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// FP16 matmul with buffer offset (for multi-token processing)
    pub fn encode_matmul_vec_fp16_with_offset(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// FP16 matmul with tanh and buffer offset
    pub fn encode_matmul_vec_fp16_tanh_with_offset(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_tanh_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// FP16 matmul with squared ReLU and buffer offset
    pub fn encode_matmul_vec_fp16_squared_relu_with_offset(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_squared_relu_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;

        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 2: L2 and RMS Normalization
    // ========================================================================

    /// L2 normalize: x = x / ||x||
    pub fn encode_l2_norm_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.l2_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // One threadgroup per token, 256 threads per threadgroup
        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    /// RMS normalize with affine transform
    pub fn encode_rms_norm_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.rms_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 2: Activation Functions
    // ========================================================================

    /// Squared ReLU: x = max(x, 0)^2
    pub fn encode_squared_relu_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.squared_relu_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Tanh activation
    pub fn encode_tanh_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.tanh_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Sigmoid activation
    pub fn encode_sigmoid_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.sigmoid_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// SiLU (Swish) activation: x * sigmoid(x)
    pub fn encode_silu_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.silu_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 3: Time Mix V7 Operations
    // ========================================================================

    /// Pack k, v, a, kk into a single tensor n
    pub fn encode_pack_kvakk_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        k: &Buffer,
        v: &Buffer,
        a: &Buffer,
        kk: &Buffer,
        n: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.pack_kvakk_fp16_pipeline());
        encoder.set_buffer(0, Some(k), 0);
        encoder.set_buffer(1, Some(v), 0);
        encoder.set_buffer(2, Some(a), 0);
        encoder.set_buffer(3, Some(kk), 0);
        encoder.set_buffer(4, Some(n), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );

        let count = channels * tokens;
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Time first kernel: y += sum_j(u[j] * k[j] * r[j]) * v
    pub fn encode_time_first_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        u: &Buffer,
        r: &Buffer,
        k: &Buffer,
        v: &Buffer,
        x: &Buffer,
        head_size: u32,
        num_heads: u32,
        num_tokens: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.time_first_fp16_pipeline());
        encoder.set_buffer(0, Some(u), 0);
        encoder.set_buffer(1, Some(r), 0);
        encoder.set_buffer(2, Some(k), 0);
        encoder.set_buffer(3, Some(v), 0);
        encoder.set_buffer(4, Some(x), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &head_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &num_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &num_tokens as *const u32 as *const _,
        );

        // One threadgroup per (token, head), 32 threads per threadgroup
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tokens as u64, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Time mix V7 kernel - main recurrent attention
    ///
    /// This kernel implements the full RWKV-7 time mix algorithm with HEAD_SIZE × HEAD_SIZE
    /// state matrix operations per head.
    ///
    /// Parameters:
    /// - state_stride: C (num_emb), the number of elements per state row
    pub fn encode_time_mix_v7_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        cursors: &Buffer,
        state: &Buffer,
        r: &Buffer,
        w: &Buffer,
        n: &Buffer,
        x: &Buffer,
        head_size: u32,
        num_heads: u32,
        num_tokens: u32,
        state_stride: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.time_mix_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(r), 0);
        encoder.set_buffer(3, Some(w), 0);
        encoder.set_buffer(4, Some(n), 0);
        encoder.set_buffer(5, Some(x), 0);
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &head_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &num_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &num_tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &state_stride as *const u32 as *const _,
        );

        // Dispatch: one threadgroup per (token, head), HEAD_SIZE threads per threadgroup
        // Each thread handles one output element within the head
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tokens as u64, num_heads as u64, 1),
            MTLSize::new(head_size as u64, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Simplified time mix for single-token inference
    pub fn encode_time_mix_v7_single_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        state: &Buffer,
        r: &Buffer,
        w: &Buffer,
        k: &Buffer,
        v: &Buffer,
        a: &Buffer,
        kk: &Buffer,
        output: &Buffer,
        head_size: u32,
        num_heads: u32,
        batch: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.time_mix_v7_single_fp16_pipeline());
        encoder.set_buffer(0, Some(state), 0);
        encoder.set_buffer(1, Some(r), 0);
        encoder.set_buffer(2, Some(w), 0);
        encoder.set_buffer(3, Some(k), 0);
        encoder.set_buffer(4, Some(v), 0);
        encoder.set_buffer(5, Some(a), 0);
        encoder.set_buffer(6, Some(kk), 0);
        encoder.set_buffer(7, Some(output), 0);
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &head_size as *const u32 as *const _,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &num_heads as *const u32 as *const _,
        );
        encoder.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &batch as *const u32 as *const _,
        );

        // 2D dispatch: heads x (head_size/32)
        let elem_groups = (head_size + 31) / 32;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_heads as u64, elem_groups as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Control K operation for RWKV-7
    pub fn encode_control_k_v7_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        k: &Buffer,
        gate: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.control_k_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(k), 0);
        encoder.set_buffer(1, Some(gate), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 4: Channel Mix Operations
    // ========================================================================

    /// Channel mix V7: x = v (no gating)
    pub fn encode_channel_mix_v7_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        cursors: &Buffer,
        state: &Buffer,
        v: &Buffer,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.channel_mix_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(x), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );

        let threads_x = 256u64;
        let num_tgs_x = ((channels as u64) + threads_x - 1) / threads_x;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs_x, tokens as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Channel mix with sigmoid gating: x = sigmoid(r) * v
    pub fn encode_channel_mix_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        cursors: &Buffer,
        state: &Buffer,
        r: &Buffer,
        v: &Buffer,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.channel_mix_fp16_pipeline());
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(r), 0);
        encoder.set_buffer(3, Some(v), 0);
        encoder.set_buffer(4, Some(x), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );

        let threads_x = 256u64;
        let num_tgs_x = ((channels as u64) + threads_x - 1) / threads_x;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs_x, tokens as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 4: Add with Activation Operations
    // ========================================================================

    /// Add with tanh: output = tanh(input) + output
    pub fn encode_add_tanh_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_tanh_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Add with sigmoid: output = sigmoid(input) + output
    pub fn encode_add_sigmoid_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_sigmoid_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    /// Add with squared ReLU: output = squared_relu(input) + output
    pub fn encode_add_squared_relu_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_squared_relu_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &count as *const u32 as *const _,
        );

        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
        encoder.end_encoding();
    }

    // ========================================================================
    // Phase 4: Fused Operations
    // ========================================================================

    /// Fused add + layer norm: output = layer_norm(input + residual)
    pub fn encode_add_layer_norm_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        residual: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.add_layer_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(residual), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(bias), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );
        encoder.set_bytes(
            7,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    /// Softmax
    pub fn encode_softmax_fp16(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(ctx.softmax_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &channels as *const u32 as *const _,
        );
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &tokens as *const u32 as *const _,
        );

        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
    }

    // ========================================================================
    // Existing Q4K Matmul Operations
    // ========================================================================

    pub fn matmul_vec_q4k(
        ctx: &MetalContext,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_vec_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        // New kernel: 4 simdgroups per threadgroup, each simdgroup (32 threads) handles one row
        // So 128 threads per threadgroup, processing 4 rows per threadgroup
        let simdgroups_per_tg = 4u32;
        let threads_per_simdgroup = 32u32;
        let threads_per_tg = simdgroups_per_tg * threads_per_simdgroup; // 128
        let rows_per_tg = simdgroups_per_tg; // 4

        let num_threadgroups = (m + rows_per_tg - 1) / rows_per_tg;

        let thread_groups = MTLSize::new(num_threadgroups as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(threads_per_tg as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Encode matmul into an existing command buffer (for batching)
    pub fn encode_matmul_vec_q4k(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_vec_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_simdgroup = 32u32;
        let threads_per_tg = simdgroups_per_tg * threads_per_simdgroup;
        let rows_per_tg = simdgroups_per_tg;
        let num_threadgroups = (m + rows_per_tg - 1) / rows_per_tg;

        let thread_groups = MTLSize::new(num_threadgroups as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(threads_per_tg as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();
    }

    pub fn matmul_mat_q4k(
        ctx: &MetalContext,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        t: u32,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        Self::encode_matmul_mat_q4k(ctx, &command_buffer, weights, input, output, k, m, t);
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    pub fn encode_matmul_mat_q4k(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        t: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_mat_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &t as *const u32 as *const _,
        );

        let tile_m = 8u32;
        let tile_t = 8u32;

        let thread_groups = MTLSize::new(
            ((m + tile_m - 1) / tile_m) as u64,
            ((t + tile_t - 1) / tile_t) as u64,
            1,
        );
        let threads_per_threadgroup = MTLSize::new(tile_m as u64, tile_t as u64, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();
    }

    // ========================================================================
    // Fast Encoder-Reuse API (enc_* variants)
    // ========================================================================
    // These functions accept an existing ComputeCommandEncoderRef and do NOT
    // call end_encoding(). The caller is responsible for managing the encoder
    // lifecycle. This eliminates encoder creation overhead when batching
    // multiple operations.

    /// Add two fp16 tensors: output = input + output (in-place)
    pub fn enc_add_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.add_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Multiply two fp16 tensors: output = input * output (in-place)
    pub fn enc_mul_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.mul_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Copy fp16 tensor: dst = src
    pub fn enc_blit_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        src: &Buffer,
        dst: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.blit_fp16_pipeline());
        encoder.set_buffer(0, Some(src), 0);
        encoder.set_buffer(1, Some(dst), 0);
        encoder.set_bytes(2, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Affine transform: x = scale * x + bias (in-place)
    pub fn enc_affine_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        scale: f32,
        bias: f32,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.affine_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &scale as *const f32 as *const _);
        encoder.set_bytes(2, 4, &bias as *const f32 as *const _);
        encoder.set_bytes(3, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Layer normalization
    pub fn enc_layer_norm_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(ctx.layer_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(4, 4, &channels as *const u32 as *const _);
        encoder.set_bytes(5, 4, &tokens as *const u32 as *const _);
        encoder.set_bytes(6, 4, &eps as *const f32 as *const _);
        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Group normalization
    pub fn enc_group_norm_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        head_size: u32,
        num_heads: u32,
        tokens: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(ctx.group_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(bias), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(4, 4, &head_size as *const u32 as *const _);
        encoder.set_bytes(5, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(6, 4, &tokens as *const u32 as *const _);
        encoder.set_bytes(7, 4, &eps as *const f32 as *const _);
        encoder.dispatch_thread_groups(
            MTLSize::new(tokens as u64, num_heads as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    /// Token shift operation
    pub fn enc_token_shift_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        cursors: &Buffer,
        time_mix: &Buffer,
        state: &Buffer,
        input: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        time_mix_stride: u32,
        reversed: bool,
    ) {
        let pipeline = if reversed {
            ctx.token_shift_reversed_fp16_pipeline()
        } else {
            ctx.token_shift_fp16_pipeline()
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(time_mix), 0);
        encoder.set_buffer(2, Some(state), 0);
        encoder.set_buffer(3, Some(input), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_bytes(5, 4, &channels as *const u32 as *const _);
        encoder.set_bytes(6, 4, &tokens as *const u32 as *const _);
        encoder.set_bytes(7, 4, &time_mix_stride as *const u32 as *const _);
        let threads_x = 256u64;
        let num_tgs_x = ((channels as u64) + threads_x - 1) / threads_x;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs_x, tokens as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
    }

    /// Linear interpolation: output = a + factor * (b - a)
    pub fn enc_lerp_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        a: &Buffer,
        b: &Buffer,
        factor: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.lerp_fp16_pipeline());
        encoder.set_buffer(0, Some(a), 0);
        encoder.set_buffer(1, Some(b), 0);
        encoder.set_buffer(2, Some(factor), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_bytes(4, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// FP16 matrix-vector multiplication
    pub fn enc_matmul_vec_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// FP16 matmul with offset
    pub fn enc_matmul_vec_fp16_with_offset(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// FP16 matmul with tanh activation
    pub fn enc_matmul_vec_fp16_tanh(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_tanh_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// FP16 matmul with tanh and offset
    pub fn enc_matmul_vec_fp16_tanh_with_offset(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_tanh_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// FP16 matmul with squared ReLU
    pub fn enc_matmul_vec_fp16_squared_relu(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_squared_relu_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// FP16 matmul with squared ReLU and offset
    pub fn enc_matmul_vec_fp16_squared_relu_with_offset(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        input_offset: u64,
        output_offset: u64,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_fp16_squared_relu_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), input_offset);
        encoder.set_buffer(2, Some(output), output_offset);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// L2 normalize
    pub fn enc_l2_norm_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(ctx.l2_norm_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &channels as *const u32 as *const _);
        encoder.set_bytes(2, 4, &tokens as *const u32 as *const _);
        encoder.set_bytes(3, 4, &eps as *const f32 as *const _);
        encoder.dispatch_thread_groups(MTLSize::new(tokens as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Squared ReLU activation
    pub fn enc_squared_relu_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.squared_relu_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Sigmoid activation
    pub fn enc_sigmoid_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        x: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.sigmoid_fp16_pipeline());
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(1, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Pack k, v, a, kk into n
    pub fn enc_pack_kvakk_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        k: &Buffer,
        v: &Buffer,
        a: &Buffer,
        kk: &Buffer,
        n: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.pack_kvakk_fp16_pipeline());
        encoder.set_buffer(0, Some(k), 0);
        encoder.set_buffer(1, Some(v), 0);
        encoder.set_buffer(2, Some(a), 0);
        encoder.set_buffer(3, Some(kk), 0);
        encoder.set_buffer(4, Some(n), 0);
        encoder.set_bytes(5, 4, &channels as *const u32 as *const _);
        encoder.set_bytes(6, 4, &tokens as *const u32 as *const _);
        let count = channels * tokens;
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Time first kernel: y += sum_j(u[j] * k[j] * r[j]) * v
    /// Uses parallel reduction within threadgroup
    pub fn enc_time_first_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        u: &Buffer,
        r: &Buffer,
        k: &Buffer,
        v: &Buffer,
        x: &Buffer,
        head_size: u32,
        num_heads: u32,
        num_tokens: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.time_first_fp16_pipeline());
        encoder.set_buffer(0, Some(u), 0);
        encoder.set_buffer(1, Some(r), 0);
        encoder.set_buffer(2, Some(k), 0);
        encoder.set_buffer(3, Some(v), 0);
        encoder.set_buffer(4, Some(x), 0);
        encoder.set_bytes(5, 4, &head_size as *const u32 as *const _);
        encoder.set_bytes(6, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(7, 4, &num_tokens as *const u32 as *const _);
        // Dispatch: one threadgroup per (token, head), HEAD_SIZE threads for reduction
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tokens as u64, num_heads as u64, 1),
            MTLSize::new(head_size as u64, 1, 1),
        );
    }

    /// Time mix V7 kernel (encoder-reuse variant)
    ///
    /// Implements full RWKV-7 time mix with HEAD_SIZE × HEAD_SIZE state matrix.
    /// state_stride should be C (num_emb).
    pub fn enc_time_mix_v7_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        cursors: &Buffer,
        state: &Buffer,
        r: &Buffer,
        w: &Buffer,
        n: &Buffer,
        x: &Buffer,
        head_size: u32,
        num_heads: u32,
        num_tokens: u32,
        state_stride: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.time_mix_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(r), 0);
        encoder.set_buffer(3, Some(w), 0);
        encoder.set_buffer(4, Some(n), 0);
        encoder.set_buffer(5, Some(x), 0);
        encoder.set_bytes(6, 4, &head_size as *const u32 as *const _);
        encoder.set_bytes(7, 4, &num_heads as *const u32 as *const _);
        encoder.set_bytes(8, 4, &num_tokens as *const u32 as *const _);
        encoder.set_bytes(9, 4, &state_stride as *const u32 as *const _);
        // Dispatch: one threadgroup per (token, head), HEAD_SIZE threads per threadgroup
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tokens as u64, num_heads as u64, 1),
            MTLSize::new(head_size as u64, 1, 1),
        );
    }

    /// Control K operation
    pub fn enc_control_k_v7_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        k: &Buffer,
        gate: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.control_k_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(k), 0);
        encoder.set_buffer(1, Some(gate), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Channel mix V7
    pub fn enc_channel_mix_v7_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        cursors: &Buffer,
        state: &Buffer,
        v: &Buffer,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.channel_mix_v7_fp16_pipeline());
        encoder.set_buffer(0, Some(cursors), 0);
        encoder.set_buffer(1, Some(state), 0);
        encoder.set_buffer(2, Some(v), 0);
        encoder.set_buffer(3, Some(x), 0);
        encoder.set_bytes(4, 4, &channels as *const u32 as *const _);
        encoder.set_bytes(5, 4, &tokens as *const u32 as *const _);
        let threads_x = 256u64;
        let num_tgs_x = ((channels as u64) + threads_x - 1) / threads_x;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs_x, tokens as u64, 1),
            MTLSize::new(threads_x, 1, 1),
        );
    }

    /// Add with sigmoid activation
    pub fn enc_add_sigmoid_fp16(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.add_sigmoid_fp16_pipeline());
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &count as *const u32 as *const _);
        let threads_per_tg = 256u64;
        let num_tgs = ((count as u64) + threads_per_tg - 1) / threads_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs, 1, 1),
            MTLSize::new(threads_per_tg, 1, 1),
        );
    }

    /// Q4K matrix-vector multiplication
    pub fn enc_matmul_vec_q4k(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_vec_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        let simdgroups_per_tg = 4u32;
        let threads_per_tg = simdgroups_per_tg * 32;
        let num_tgs = (m + simdgroups_per_tg - 1) / simdgroups_per_tg;
        encoder.dispatch_thread_groups(
            MTLSize::new(num_tgs as u64, 1, 1),
            MTLSize::new(threads_per_tg as u64, 1, 1),
        );
    }

    /// Q4K matrix-matrix multiplication
    pub fn enc_matmul_mat_q4k(
        ctx: &MetalContext,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        t: u32,
    ) {
        encoder.set_compute_pipeline_state(ctx.matmul_mat_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &k as *const u32 as *const _);
        encoder.set_bytes(4, 4, &m as *const u32 as *const _);
        encoder.set_bytes(5, 4, &t as *const u32 as *const _);
        let tile_m = 8u32;
        let tile_t = 8u32;
        encoder.dispatch_thread_groups(
            MTLSize::new(
                ((m + tile_m - 1) / tile_m) as u64,
                ((t + tile_t - 1) / tile_t) as u64,
                1,
            ),
            MTLSize::new(tile_m as u64, tile_t as u64, 1),
        );
    }
}

//! Metal Layer Executor for pure Metal execution of RWKV layers.
//!
//! This module provides a high-level executor that runs entire RWKV layers
//! in pure Metal, synchronizing only once per layer instead of per-operation.
//! This eliminates the wgpu/Metal synchronization overhead that was causing
//! the 10 tok/s regression when using per-op Metal dispatch.

use std::sync::Arc;

use metal::{Buffer, CommandBufferRef};

use super::{MetalContext, MetalOps};

/// Configuration for a layer execution.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub num_emb: u32,
    pub num_head: u32,
    pub head_size: u32,
    pub num_tokens: u32,
}

impl LayerConfig {
    pub fn new(num_emb: usize, num_head: usize, num_tokens: usize) -> Self {
        let head_size = num_emb / num_head;
        Self {
            num_emb: num_emb as u32,
            num_head: num_head as u32,
            head_size: head_size as u32,
            num_tokens: num_tokens as u32,
        }
    }
}

/// Executes RWKV layers entirely in Metal.
///
/// Usage:
/// 1. Create executor with `new()`
/// 2. Call `begin_layer()` at the start of each layer
/// 3. Encode operations using the various `encode_*` methods
/// 4. Call `end_layer()` to commit and wait for completion
pub struct MetalLayerExecutor {
    ctx: Arc<MetalContext>,
}

impl MetalLayerExecutor {
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self { ctx }
    }

    /// Get the Metal context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Begin a new command buffer for layer execution.
    pub fn begin_command_buffer(&self) -> metal::CommandBuffer {
        self.ctx.queue().new_command_buffer().to_owned()
    }

    /// Commit and wait for a command buffer to complete.
    pub fn commit_and_wait(&self, cmd: metal::CommandBuffer) {
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ========================================================================
    // Phase 1: Simple Operations
    // ========================================================================

    /// Element-wise add: b = a + b (in-place on b)
    pub fn encode_add(&self, cmd: &CommandBufferRef, a: &Buffer, b: &Buffer, count: u32) {
        MetalOps::encode_add_fp16(&self.ctx, cmd, a, b, count);
    }

    /// Element-wise multiply: b = a * b (in-place on b)
    pub fn encode_mul(&self, cmd: &CommandBufferRef, a: &Buffer, b: &Buffer, count: u32) {
        MetalOps::encode_mul_fp16(&self.ctx, cmd, a, b, count);
    }

    /// Copy data: output = input
    pub fn encode_blit(&self, cmd: &CommandBufferRef, input: &Buffer, output: &Buffer, count: u32) {
        MetalOps::encode_blit_fp16(&self.ctx, cmd, input, output, count);
    }

    /// Affine transform: x = x * scale + bias
    pub fn encode_affine(
        &self,
        cmd: &CommandBufferRef,
        x: &Buffer,
        scale: f32,
        bias: f32,
        count: u32,
    ) {
        MetalOps::encode_affine_fp16(&self.ctx, cmd, x, scale, bias, count);
    }

    /// Layer normalization
    pub fn encode_layer_norm(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        MetalOps::encode_layer_norm_fp16(
            &self.ctx, cmd, input, weight, bias, output, channels, tokens, eps,
        );
    }

    /// Group normalization
    pub fn encode_group_norm(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        head_size: u32,
        num_heads: u32,
        tokens: u32,
        eps: f32,
    ) {
        MetalOps::encode_group_norm_fp16(
            &self.ctx, cmd, input, weight, bias, output, head_size, num_heads, tokens, eps,
        );
    }

    /// Token shift operation
    pub fn encode_token_shift(
        &self,
        cmd: &CommandBufferRef,
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
        MetalOps::encode_token_shift_fp16(
            &self.ctx,
            cmd,
            cursors,
            time_mix,
            state,
            input,
            output,
            channels,
            tokens,
            time_mix_stride,
            reversed,
        );
    }

    /// Linear interpolation: output = a + factor * (b - a)
    pub fn encode_lerp(
        &self,
        cmd: &CommandBufferRef,
        a: &Buffer,
        b: &Buffer,
        factor: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        MetalOps::encode_lerp_fp16(&self.ctx, cmd, a, b, factor, output, count);
    }

    // ========================================================================
    // Phase 2: FP16 Matmul and Normalization
    // ========================================================================

    /// FP16 matrix-vector multiplication
    pub fn encode_matmul_vec_fp16(
        &self,
        cmd: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        MetalOps::encode_matmul_vec_fp16(&self.ctx, cmd, weights, input, output, k, m);
    }

    /// FP16 matmul with tanh activation
    pub fn encode_matmul_vec_fp16_tanh(
        &self,
        cmd: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        MetalOps::encode_matmul_vec_fp16_tanh(&self.ctx, cmd, weights, input, output, k, m);
    }

    /// FP16 matmul with squared ReLU activation
    pub fn encode_matmul_vec_fp16_squared_relu(
        &self,
        cmd: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        MetalOps::encode_matmul_vec_fp16_squared_relu(&self.ctx, cmd, weights, input, output, k, m);
    }

    /// L2 normalization
    pub fn encode_l2_norm(
        &self,
        cmd: &CommandBufferRef,
        x: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        MetalOps::encode_l2_norm_fp16(&self.ctx, cmd, x, channels, tokens, eps);
    }

    /// RMS normalization with affine
    pub fn encode_rms_norm(
        &self,
        cmd: &CommandBufferRef,
        x: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        MetalOps::encode_rms_norm_fp16(&self.ctx, cmd, x, weight, bias, channels, tokens, eps);
    }

    /// Squared ReLU activation
    pub fn encode_squared_relu(&self, cmd: &CommandBufferRef, x: &Buffer, count: u32) {
        MetalOps::encode_squared_relu_fp16(&self.ctx, cmd, x, count);
    }

    /// Tanh activation
    pub fn encode_tanh(&self, cmd: &CommandBufferRef, x: &Buffer, count: u32) {
        MetalOps::encode_tanh_fp16(&self.ctx, cmd, x, count);
    }

    /// Sigmoid activation
    pub fn encode_sigmoid(&self, cmd: &CommandBufferRef, x: &Buffer, count: u32) {
        MetalOps::encode_sigmoid_fp16(&self.ctx, cmd, x, count);
    }

    /// SiLU activation
    pub fn encode_silu(&self, cmd: &CommandBufferRef, x: &Buffer, count: u32) {
        MetalOps::encode_silu_fp16(&self.ctx, cmd, x, count);
    }

    // ========================================================================
    // Phase 3: Time Mix V7
    // ========================================================================

    /// Pack k, v, a, kk into a single tensor
    pub fn encode_pack_kvakk(
        &self,
        cmd: &CommandBufferRef,
        k: &Buffer,
        v: &Buffer,
        a: &Buffer,
        kk: &Buffer,
        n: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        MetalOps::encode_pack_kvakk_fp16(&self.ctx, cmd, k, v, a, kk, n, channels, tokens);
    }

    /// Time first operation
    pub fn encode_time_first(
        &self,
        cmd: &CommandBufferRef,
        u: &Buffer,
        r: &Buffer,
        k: &Buffer,
        v: &Buffer,
        x: &Buffer,
        head_size: u32,
        num_heads: u32,
        num_tokens: u32,
    ) {
        MetalOps::encode_time_first_fp16(
            &self.ctx, cmd, u, r, k, v, x, head_size, num_heads, num_tokens,
        );
    }

    /// Time mix V7 operation
    pub fn encode_time_mix_v7(
        &self,
        cmd: &CommandBufferRef,
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
        MetalOps::encode_time_mix_v7_fp16(
            &self.ctx,
            cmd,
            cursors,
            state,
            r,
            w,
            n,
            x,
            head_size,
            num_heads,
            num_tokens,
            state_stride,
        );
    }

    /// Simplified time mix for single-token inference
    pub fn encode_time_mix_v7_single(
        &self,
        cmd: &CommandBufferRef,
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
        MetalOps::encode_time_mix_v7_single_fp16(
            &self.ctx, cmd, state, r, w, k, v, a, kk, output, head_size, num_heads, batch,
        );
    }

    /// Control K operation
    pub fn encode_control_k_v7(
        &self,
        cmd: &CommandBufferRef,
        k: &Buffer,
        gate: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        MetalOps::encode_control_k_v7_fp16(&self.ctx, cmd, k, gate, output, count);
    }

    // ========================================================================
    // Phase 4: Channel Mix and Fused Operations
    // ========================================================================

    /// Channel mix V7
    pub fn encode_channel_mix_v7(
        &self,
        cmd: &CommandBufferRef,
        cursors: &Buffer,
        state: &Buffer,
        v: &Buffer,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        MetalOps::encode_channel_mix_v7_fp16(
            &self.ctx, cmd, cursors, state, v, x, channels, tokens,
        );
    }

    /// Channel mix with sigmoid gating
    pub fn encode_channel_mix(
        &self,
        cmd: &CommandBufferRef,
        cursors: &Buffer,
        state: &Buffer,
        r: &Buffer,
        v: &Buffer,
        x: &Buffer,
        channels: u32,
        tokens: u32,
    ) {
        MetalOps::encode_channel_mix_fp16(
            &self.ctx, cmd, cursors, state, r, v, x, channels, tokens,
        );
    }

    /// Add with tanh activation
    pub fn encode_add_tanh(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        MetalOps::encode_add_tanh_fp16(&self.ctx, cmd, input, output, count);
    }

    /// Add with sigmoid activation
    pub fn encode_add_sigmoid(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        MetalOps::encode_add_sigmoid_fp16(&self.ctx, cmd, input, output, count);
    }

    /// Add with squared ReLU activation
    pub fn encode_add_squared_relu(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        MetalOps::encode_add_squared_relu_fp16(&self.ctx, cmd, input, output, count);
    }

    /// Fused add + layer norm
    pub fn encode_add_layer_norm(
        &self,
        cmd: &CommandBufferRef,
        input: &Buffer,
        residual: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        output: &Buffer,
        channels: u32,
        tokens: u32,
        eps: f32,
    ) {
        MetalOps::encode_add_layer_norm_fp16(
            &self.ctx, cmd, input, residual, weight, bias, output, channels, tokens, eps,
        );
    }

    /// Softmax
    pub fn encode_softmax(&self, cmd: &CommandBufferRef, x: &Buffer, channels: u32, tokens: u32) {
        MetalOps::encode_softmax_fp16(&self.ctx, cmd, x, channels, tokens);
    }

    // ========================================================================
    // Q4K Matmul (existing)
    // ========================================================================

    /// Q4K matrix-vector multiplication
    pub fn encode_matmul_vec_q4k(
        &self,
        cmd: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        MetalOps::encode_matmul_vec_q4k(&self.ctx, cmd, weights, input, output, k, m);
    }
}

impl std::fmt::Debug for MetalLayerExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalLayerExecutor").finish()
    }
}

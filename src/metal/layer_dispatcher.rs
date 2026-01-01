//! Metal Layer Dispatcher for pure Metal layer execution.
//!
//! This module provides the core dispatcher that encodes an entire RWKV layer
//! into a single Metal command buffer, eliminating per-operation synchronization
//! overhead that was causing the 10 tok/s regression.

use std::sync::Arc;

use metal::{Buffer, CommandBufferRef, ComputeCommandEncoderRef};

use super::layer_buffers::{MetalLayerBuffers, MetalLayerState};
use super::layer_weights::{
    MetalAttWeights, MetalFfnWeights, MetalLayerWeights, MetalMatrixWeights,
};
use super::{MetalContext, MetalOps};

/// Constants matching the RWKV model.
const LN_EPS: f32 = 1.0e-5;
const GN_EPS: f32 = 64.0e-5;
const L2_EPS: f32 = 1.0e-12;

/// Dispatches entire RWKV layers in pure Metal.
///
/// This dispatcher encodes all operations for a layer into a single Metal
/// command buffer, synchronizing only once at the end of the layer instead
/// of after each operation.
pub struct MetalLayerDispatcher {
    ctx: Arc<MetalContext>,
}

impl MetalLayerDispatcher {
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

    /// Dispatch an entire RWKV layer in pure Metal using a shared encoder.
    ///
    /// The encoder is passed in from the caller so that ALL layers can share
    /// a single encoder, eliminating encoder creation overhead entirely.
    pub fn dispatch_layer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        layer: &MetalLayerWeights,
        buffers: &MetalLayerBuffers,
        state: &MetalLayerState,
        layer_index: usize,
        rescale: usize,
    ) {
        // Encode attention block
        self.enc_attention(encoder, &layer.att, buffers, state, layer_index);

        // Encode FFN block
        self.enc_ffn(encoder, &layer.ffn, buffers, state);

        // Apply rescale if needed
        if (layer_index + 1) % rescale == 0 {
            MetalOps::enc_affine_fp16(
                &self.ctx,
                encoder,
                &buffers.x,
                0.5,
                0.0,
                buffers.main_count(),
            );
        }
    }

    /// Encode the attention block operations using a shared encoder.
    fn enc_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        att: &MetalAttWeights,
        buffers: &MetalLayerBuffers,
        state: &MetalLayerState,
        layer_index: usize,
    ) {
        let c = buffers.num_emb;
        let t = buffers.num_token;
        let h = buffers.num_head;
        let s = buffers.head_size;
        let count = c * t;

        // 1. Copy x to att_x
        MetalOps::enc_blit_fp16(&self.ctx, encoder, &buffers.x, &buffers.att_x, count);

        // 2. Layer norm on att_x
        MetalOps::enc_layer_norm_fp16(
            &self.ctx,
            encoder,
            &buffers.att_x,
            &att.ln.w,
            &att.ln.b,
            &buffers.att_x,
            c,
            t,
            LN_EPS,
        );

        // 3. Token shifts (6x) - all use the same state but different mix factors
        self.enc_token_shift(
            encoder,
            &att.x_r,
            state.att(),
            &buffers.att_x,
            &buffers.att_rx,
            buffers,
        );
        self.enc_token_shift(
            encoder,
            &att.x_w,
            state.att(),
            &buffers.att_x,
            &buffers.att_wx,
            buffers,
        );
        self.enc_token_shift(
            encoder,
            &att.x_k,
            state.att(),
            &buffers.att_x,
            &buffers.att_kx,
            buffers,
        );
        self.enc_token_shift(
            encoder,
            &att.x_v,
            state.att(),
            &buffers.att_x,
            &buffers.att_vx,
            buffers,
        );
        self.enc_token_shift(
            encoder,
            &att.x_a,
            state.att(),
            &buffers.att_x,
            &buffers.att_ax,
            buffers,
        );
        self.enc_token_shift(
            encoder,
            &att.x_g,
            state.att(),
            &buffers.att_x,
            &buffers.att_gx,
            buffers,
        );

        // 4. Linear projections for r, k, v
        self.enc_matmul(encoder, &att.w_r, &buffers.att_rx, &buffers.att_r, t, false);
        self.enc_matmul(encoder, &att.w_k, &buffers.att_kx, &buffers.att_k, t, false);
        self.enc_matmul(encoder, &att.w_v, &buffers.att_vx, &buffers.att_v, t, false);

        // 5. Time decay LoRA: w = w2(tanh(w1(wx))) + w0
        self.enc_matmul_fp16_tanh(encoder, &att.w1, &buffers.att_wx, &buffers.aux_w, t);
        self.enc_matmul_fp16(encoder, &att.w2, &buffers.aux_w, &buffers.att_w, t);
        MetalOps::enc_add_fp16(&self.ctx, encoder, &att.w0, &buffers.att_w, count);

        // 6. Bonus A LoRA: a = sigmoid(a2(a1(ax)) + a0)
        self.enc_matmul_fp16(encoder, &att.a1, &buffers.att_ax, &buffers.aux_a, t);
        self.enc_matmul_fp16(encoder, &att.a2, &buffers.aux_a, &buffers.att_a, t);
        MetalOps::enc_add_sigmoid_fp16(&self.ctx, encoder, &att.a0, &buffers.att_a, count);

        // 7. Gate LoRA: g = g2(sigmoid(g1(gx)))
        self.enc_matmul_fp16_sigmoid(encoder, &att.g1, &buffers.att_gx, &buffers.aux_g, t);
        self.enc_matmul_fp16(encoder, &att.g2, &buffers.aux_g, &buffers.att_g, t);

        // 8. Control operations
        // kk = k * k_k, then L2 normalize
        MetalOps::enc_blit_fp16(&self.ctx, encoder, &buffers.att_k, &buffers.att_kk, count);
        MetalOps::enc_mul_fp16(&self.ctx, encoder, &att.k_k, &buffers.att_kk, count);
        MetalOps::enc_l2_norm_fp16(&self.ctx, encoder, &buffers.att_kk, s, h * t, L2_EPS);

        // k = control_k(k, sigmoid(a) * k_a)
        MetalOps::enc_control_k_v7_fp16(
            &self.ctx,
            encoder,
            &att.k_a,
            &buffers.att_a,
            &buffers.att_k,
            count,
        );

        // 9. Value residual (layer 0 just copies, others use LoRA)
        if layer_index == 0 {
            MetalOps::enc_blit_fp16(&self.ctx, encoder, &buffers.att_v, &buffers.att_v0, count);
        } else {
            // vv = sigmoid(v2(v1(vx)) + v0)
            self.enc_matmul_fp16(encoder, &att.v1, &buffers.att_vx, &buffers.aux_v, t);
            self.enc_matmul_fp16(encoder, &att.v2, &buffers.aux_v, &buffers.att_vv, t);
            MetalOps::enc_add_sigmoid_fp16(&self.ctx, encoder, &att.v0, &buffers.att_vv, count);
            // v = lerp(v0, v, vv)
            MetalOps::enc_lerp_fp16(
                &self.ctx,
                encoder,
                &buffers.att_v0,
                &buffers.att_v,
                &buffers.att_vv,
                &buffers.att_v,
                count,
            );
        }

        // 10. Pack k, v, a, kk into n for time mix
        MetalOps::enc_pack_kvakk_fp16(
            &self.ctx,
            encoder,
            &buffers.att_k,
            &buffers.att_v,
            &buffers.att_a,
            &buffers.att_kk,
            &buffers.att_n,
            c,
            t,
        );

        // 11. Time mix V7 - the main recurrent attention
        MetalOps::enc_time_mix_v7_fp16(
            &self.ctx,
            encoder,
            &buffers.cursors,
            state.att(),
            &buffers.att_r,
            &buffers.att_w,
            &buffers.att_n,
            &buffers.att_x,
            s,
            h,
            t,
            state.state_stride,
        );

        // 12. Group norm on att_x
        MetalOps::enc_group_norm_fp16(
            &self.ctx,
            encoder,
            &buffers.att_x,
            &att.gn.w,
            &att.gn.b,
            &buffers.att_x,
            s,
            h,
            t,
            GN_EPS,
        );

        // 13. Time first: x += sum(r_k * r * kk) * v
        MetalOps::enc_time_first_fp16(
            &self.ctx,
            encoder,
            &att.r_k,
            &buffers.att_r,
            &buffers.att_kk,
            &buffers.att_v,
            &buffers.att_x,
            s,
            h,
            t,
        );

        // 14. Gate: x = x * g
        MetalOps::enc_mul_fp16(&self.ctx, encoder, &buffers.att_g, &buffers.att_x, count);

        // 15. Output projection
        self.enc_matmul(encoder, &att.w_o, &buffers.att_x, &buffers.att_o, t, false);

        // 16. Residual add: x = x + att_o
        MetalOps::enc_add_fp16(&self.ctx, encoder, &buffers.att_o, &buffers.x, count);
    }

    /// Encode the FFN block operations using a shared encoder.
    fn enc_ffn(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ffn: &MetalFfnWeights,
        buffers: &MetalLayerBuffers,
        state: &MetalLayerState,
    ) {
        let c = buffers.num_emb;
        let t = buffers.num_token;
        let count = c * t;

        // 1. Copy x to ffn_x
        MetalOps::enc_blit_fp16(&self.ctx, encoder, &buffers.x, &buffers.ffn_x, count);

        // 2. Layer norm on ffn_x
        MetalOps::enc_layer_norm_fp16(
            &self.ctx,
            encoder,
            &buffers.ffn_x,
            &ffn.ln.w,
            &ffn.ln.b,
            &buffers.ffn_x,
            c,
            t,
            LN_EPS,
        );

        // 3. Token shift
        self.enc_token_shift(
            encoder,
            &ffn.x_k,
            state.ffn(),
            &buffers.ffn_x,
            &buffers.ffn_kx,
            buffers,
        );

        // 4. Key projection with squared ReLU activation
        self.enc_matmul_squared_relu(encoder, &ffn.w_k, &buffers.ffn_kx, &buffers.ffn_k, t);

        // 5. Value projection (sparse - uses optimized path)
        self.enc_matmul(encoder, &ffn.w_v, &buffers.ffn_k, &buffers.ffn_v, t, true);

        // 6. Channel mix V7: updates state and outputs v
        MetalOps::enc_channel_mix_v7_fp16(
            &self.ctx,
            encoder,
            &buffers.cursors,
            state.ffn(),
            &buffers.ffn_v,
            &buffers.ffn_x,
            c,
            t,
        );

        // 7. Residual add: x = x + ffn_x
        MetalOps::enc_add_fp16(&self.ctx, encoder, &buffers.ffn_x, &buffers.x, count);
    }

    /// Encode token shift operation using shared encoder.
    fn enc_token_shift(
        &self,
        encoder: &ComputeCommandEncoderRef,
        time_mix: &Buffer,
        state: &Buffer,
        input: &Buffer,
        output: &Buffer,
        buffers: &MetalLayerBuffers,
    ) {
        MetalOps::enc_token_shift_fp16(
            &self.ctx,
            encoder,
            &buffers.cursors,
            time_mix,
            state,
            input,
            output,
            buffers.num_emb,
            buffers.num_token,
            0,     // time_mix_stride = 0 for shared mix factors
            false, // not reversed
        );
    }

    /// Encode matrix multiplication (Q4K or FP16) using shared encoder.
    fn enc_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &MetalMatrixWeights,
        input: &Buffer,
        output: &Buffer,
        num_tokens: u32,
        _sparse: bool,
    ) {
        match weights {
            MetalMatrixWeights::Fp16 { w, k, m } => {
                if num_tokens == 1 {
                    MetalOps::enc_matmul_vec_fp16(&self.ctx, encoder, w, input, output, *k, *m);
                } else {
                    for t in 0..num_tokens {
                        let input_offset = (t * *k) as u64 * 2;
                        let output_offset = (t * *m) as u64 * 2;
                        MetalOps::enc_matmul_vec_fp16_with_offset(
                            &self.ctx,
                            encoder,
                            w,
                            input,
                            output,
                            *k,
                            *m,
                            input_offset,
                            output_offset,
                        );
                    }
                }
            }
            MetalMatrixWeights::Q4K { w, k, m } => {
                if num_tokens == 1 {
                    MetalOps::enc_matmul_vec_q4k(&self.ctx, encoder, w, input, output, *k, *m);
                } else {
                    MetalOps::enc_matmul_mat_q4k(
                        &self.ctx, encoder, w, input, output, *k, *m, num_tokens,
                    );
                }
            }
        }
    }

    /// Encode FP16 matmul (for LoRA matrices) using shared encoder.
    fn enc_matmul_fp16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        num_tokens: u32,
    ) {
        let input_size = input.length() as u32 / 2 / num_tokens;
        let output_size = output.length() as u32 / 2 / num_tokens;

        if num_tokens == 1 {
            MetalOps::enc_matmul_vec_fp16(
                &self.ctx,
                encoder,
                weights,
                input,
                output,
                input_size,
                output_size,
            );
        } else {
            for t in 0..num_tokens {
                let input_offset = (t * input_size) as u64 * 2;
                let output_offset = (t * output_size) as u64 * 2;
                MetalOps::enc_matmul_vec_fp16_with_offset(
                    &self.ctx,
                    encoder,
                    weights,
                    input,
                    output,
                    input_size,
                    output_size,
                    input_offset,
                    output_offset,
                );
            }
        }
    }

    /// Encode FP16 matmul with tanh activation using shared encoder.
    fn enc_matmul_fp16_tanh(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        num_tokens: u32,
    ) {
        let input_size = input.length() as u32 / 2 / num_tokens;
        let output_size = output.length() as u32 / 2 / num_tokens;

        if num_tokens == 1 {
            MetalOps::enc_matmul_vec_fp16_tanh(
                &self.ctx,
                encoder,
                weights,
                input,
                output,
                input_size,
                output_size,
            );
        } else {
            for t in 0..num_tokens {
                let input_offset = (t * input_size) as u64 * 2;
                let output_offset = (t * output_size) as u64 * 2;
                MetalOps::enc_matmul_vec_fp16_tanh_with_offset(
                    &self.ctx,
                    encoder,
                    weights,
                    input,
                    output,
                    input_size,
                    output_size,
                    input_offset,
                    output_offset,
                );
            }
        }
    }

    /// Encode FP16 matmul with sigmoid activation using shared encoder.
    fn enc_matmul_fp16_sigmoid(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        num_tokens: u32,
    ) {
        let input_size = input.length() as u32 / 2 / num_tokens;
        let output_size = output.length() as u32 / 2 / num_tokens;

        if num_tokens == 1 {
            MetalOps::enc_matmul_vec_fp16(
                &self.ctx,
                encoder,
                weights,
                input,
                output,
                input_size,
                output_size,
            );
            MetalOps::enc_sigmoid_fp16(&self.ctx, encoder, output, output_size);
        } else {
            for t in 0..num_tokens {
                let input_offset = (t * input_size) as u64 * 2;
                let output_offset = (t * output_size) as u64 * 2;
                MetalOps::enc_matmul_vec_fp16_with_offset(
                    &self.ctx,
                    encoder,
                    weights,
                    input,
                    output,
                    input_size,
                    output_size,
                    input_offset,
                    output_offset,
                );
            }
            MetalOps::enc_sigmoid_fp16(&self.ctx, encoder, output, output_size * num_tokens);
        }
    }

    /// Encode matmul with squared ReLU activation (for FFN key) using shared encoder.
    fn enc_matmul_squared_relu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &MetalMatrixWeights,
        input: &Buffer,
        output: &Buffer,
        num_tokens: u32,
    ) {
        match weights {
            MetalMatrixWeights::Fp16 { w, k, m } => {
                if num_tokens == 1 {
                    MetalOps::enc_matmul_vec_fp16_squared_relu(
                        &self.ctx, encoder, w, input, output, *k, *m,
                    );
                } else {
                    for t in 0..num_tokens {
                        let input_offset = (t * *k) as u64 * 2;
                        let output_offset = (t * *m) as u64 * 2;
                        MetalOps::enc_matmul_vec_fp16_squared_relu_with_offset(
                            &self.ctx,
                            encoder,
                            w,
                            input,
                            output,
                            *k,
                            *m,
                            input_offset,
                            output_offset,
                        );
                    }
                }
            }
            MetalMatrixWeights::Q4K { w, k, m } => {
                if num_tokens == 1 {
                    MetalOps::enc_matmul_vec_q4k(&self.ctx, encoder, w, input, output, *k, *m);
                    MetalOps::enc_squared_relu_fp16(&self.ctx, encoder, output, *m);
                } else {
                    MetalOps::enc_matmul_mat_q4k(
                        &self.ctx, encoder, w, input, output, *k, *m, num_tokens,
                    );
                    MetalOps::enc_squared_relu_fp16(&self.ctx, encoder, output, *m * num_tokens);
                }
            }
        }
    }
}

impl std::fmt::Debug for MetalLayerDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalLayerDispatcher").finish()
    }
}

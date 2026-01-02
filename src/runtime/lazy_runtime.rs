//! Lazy runtime utilities for deferred tensor evaluation.
//!
//! This module provides utilities for using lazy tensors in inference,
//! enabling better memory planning and potential operation fusion.
//!
//! ## V7 Integration
//!
//! The `LazyV7Att` and `LazyV7Ffn` structs provide lazy wrappers for V7 attention
//! and FFN blocks. These build compute graphs that can be executed efficiently
//! with buffer reuse and potential operation fusion.

use half::f16;

use crate::{
    context::Context,
    tensor::{
        kind::ReadWrite, lazy_tensor::LazyTensor, matrix::Matrix, ops::Activation, TensorGpu,
    },
};

/// A lazy layer that wraps layer normalization parameters.
pub struct LazyLayerNorm {
    pub weight: LazyTensor<f16>,
    pub bias: LazyTensor<f16>,
}

impl LazyLayerNorm {
    /// Create a lazy layer norm from GPU tensors.
    pub fn from_gpu(weight: &TensorGpu<f16, ReadWrite>, bias: &TensorGpu<f16, ReadWrite>) -> Self {
        Self {
            weight: LazyTensor::from_gpu(weight),
            bias: LazyTensor::from_gpu(bias),
        }
    }

    /// Apply layer normalization to the input.
    pub fn forward(&self, input: &LazyTensor<f16>, eps: f32) -> LazyTensor<f16> {
        input.layer_norm(&self.weight, &self.bias, eps)
    }
}

/// A lazy linear layer (matrix multiplication).
pub struct LazyLinear<'a> {
    pub matrix: &'a Matrix,
    pub activation: Activation,
    pub turbo: bool,
}

impl<'a> LazyLinear<'a> {
    /// Create a new lazy linear layer.
    pub fn new(matrix: &'a Matrix, activation: Activation, turbo: bool) -> Self {
        Self {
            matrix,
            activation,
            turbo,
        }
    }

    /// Apply the linear transformation to the input.
    pub fn forward(&self, input: &LazyTensor<f16>) -> LazyTensor<f16> {
        input.matmul(self.matrix, self.activation, self.turbo)
    }
}

/// A lazy FFN (feed-forward network) block.
///
/// This represents a simplified FFN without state management,
/// suitable for demonstrating lazy evaluation.
pub struct LazyFfn<'a> {
    pub ln: LazyLayerNorm,
    pub w_k: &'a Matrix,
    pub w_v: &'a Matrix,
}

impl<'a> LazyFfn<'a> {
    /// Forward pass through the FFN block.
    ///
    /// Note: This is a simplified version without token shift or channel mix.
    /// Full implementation would require cursor-free variants of those ops.
    pub fn forward(&self, x: &LazyTensor<f16>, eps: f32, turbo: bool) -> LazyTensor<f16> {
        // Layer norm
        let normed = self.ln.forward(x, eps);

        // Key projection with squared relu
        let k = normed.matmul(self.w_k, Activation::SquaredRelu, turbo);

        // Value projection
        let v = k.matmul(self.w_v, Activation::None, turbo);

        // Residual connection
        x.add(&v)
    }
}

/// Statistics about lazy graph execution.
#[derive(Debug, Clone, Default)]
pub struct LazyStats {
    pub nodes_created: usize,
    pub nodes_executed: usize,
    pub buffers_reused: usize,
}

impl LazyStats {
    /// Get stats from the context's compute graph.
    pub fn from_context(context: &Context) -> Self {
        let graph = context.graph();
        Self {
            nodes_created: graph.len(),
            nodes_executed: 0, // Would need tracking in resolve()
            buffers_reused: 0, // Would need tracking in plan_buffers()
        }
    }
}

/// Clear all cached buffers in the compute graph.
///
/// This is useful for reducing memory pressure between inference batches.
pub fn clear_graph_caches(context: &Context) {
    context.graph().clear_caches();
}

/// Lazy V7 attention layer parameters.
///
/// This wraps the V7 attention layer weights for lazy evaluation.
/// The forward pass builds a compute graph that can be executed efficiently.
pub struct LazyV7Att<'a> {
    pub ln: LazyLayerNorm,
    pub x_r: LazyTensor<f16>,
    pub x_w: LazyTensor<f16>,
    pub x_k: LazyTensor<f16>,
    pub x_v: LazyTensor<f16>,
    pub x_a: LazyTensor<f16>,
    pub x_g: LazyTensor<f16>,
    pub w0: LazyTensor<f16>,
    pub a0: LazyTensor<f16>,
    pub v0: LazyTensor<f16>,
    pub w1: &'a Matrix,
    pub w2: &'a Matrix,
    pub a1: &'a Matrix,
    pub a2: &'a Matrix,
    pub g1: &'a Matrix,
    pub g2: &'a Matrix,
    pub v1: &'a Matrix,
    pub v2: &'a Matrix,
    pub r_k: LazyTensor<f16>,
    pub k_k: LazyTensor<f16>,
    pub k_a: LazyTensor<f16>,
    pub w_k: &'a Matrix,
    pub w_v: &'a Matrix,
    pub w_r: &'a Matrix,
    pub w_o: &'a Matrix,
    pub gn_w: LazyTensor<f16>,
    pub gn_b: LazyTensor<f16>,
}

impl<'a> LazyV7Att<'a> {
    /// Forward pass through the V7 attention block.
    ///
    /// This builds a lazy compute graph for the attention computation.
    /// The actual GPU execution happens when `materialize()` is called on the result.
    ///
    /// Note: This is a simplified version that doesn't include the time_mix_v7 kernel
    /// (which requires cursor-based state updates). The full V7 attention would need
    /// cursor-free variants of time_mix_v7 and time_first_v7.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &LazyTensor<f16>,
        state: &LazyTensor<f32>,
        layer_index: usize,
        eps: f32,
        gn_eps: f32,
        l2_eps: f32,
        turbo: bool,
    ) -> LazyV7AttOutput<'a> {
        let att_x = x.blit();
        let normed = self.ln.forward(&att_x, eps);

        let att_rx = normed.token_shift(state, &self.x_r);
        let att_wx = normed.token_shift(state, &self.x_w);
        let att_kx = normed.token_shift(state, &self.x_k);
        let att_vx = normed.token_shift(state, &self.x_v);
        let att_ax = normed.token_shift(state, &self.x_a);
        let att_gx = normed.token_shift(state, &self.x_g);

        let att_r = att_rx.matmul(self.w_r, Activation::None, turbo);
        let att_k = att_kx.matmul(self.w_k, Activation::None, turbo);
        let att_v = att_vx.matmul(self.w_v, Activation::None, turbo);

        let aux_w = att_wx.matmul(self.w1, Activation::Tanh, turbo);
        let att_w = aux_w.matmul(self.w2, Activation::None, turbo);
        let att_w = self.w0.add(&att_w);

        let aux_a = att_ax.matmul(self.a1, Activation::None, turbo);
        let att_a = aux_a.matmul(self.a2, Activation::None, turbo);
        let att_a = self.a0.add_activate(
            &att_a,
            Activation::None,
            Activation::None,
            Activation::Sigmoid,
        );

        let aux_g = att_gx.matmul(self.g1, Activation::Sigmoid, turbo);
        let att_g = aux_g.matmul(self.g2, Activation::None, turbo);

        let att_kk = att_k.blit();
        let att_kk = self.k_k.mul(&att_kk);
        let att_kk = att_kk.l2_norm(l2_eps);

        let att_k = att_k.control_k_v7(&self.k_a, &att_a);

        let att_vv = if layer_index == 0 {
            att_v.blit()
        } else {
            let aux_v = att_vx.matmul(self.v1, Activation::None, turbo);
            let vv = aux_v.matmul(self.v2, Activation::None, turbo);
            let vv =
                self.v0
                    .add_activate(&vv, Activation::None, Activation::None, Activation::Sigmoid);
            att_v.lerp(&att_v, &vv, true)
        };

        let att_n = LazyTensor::pack_kvakk(&att_k, &att_vv, &att_a, &att_kk);

        LazyV7AttOutput {
            r: att_r,
            w: att_w,
            n: att_n,
            g: att_g,
            x: normed,
            gn_w: self.gn_w.clone(),
            gn_b: self.gn_b.clone(),
            r_k: self.r_k.clone(),
            w_o: self.w_o,
            gn_eps,
            turbo,
        }
    }
}

/// Output from lazy V7 attention forward pass.
///
/// Contains intermediate tensors needed for time_mix and output projection.
/// Call `time_mix()` to apply the WKV kernel, then `finalize()` to complete.
pub struct LazyV7AttOutput<'a> {
    pub r: LazyTensor<f16>,
    pub w: LazyTensor<f16>,
    pub n: LazyTensor<f16>,
    pub g: LazyTensor<f16>,
    pub x: LazyTensor<f16>,
    pub gn_w: LazyTensor<f16>,
    pub gn_b: LazyTensor<f16>,
    pub r_k: LazyTensor<f16>,
    pub w_o: &'a Matrix,
    pub gn_eps: f32,
    pub turbo: bool,
}

impl<'a> LazyV7AttOutput<'a> {
    /// Apply the V7 time_mix (WKV) kernel.
    ///
    /// This is the core attention computation that updates state and produces output.
    /// Returns the output tensor after time_mix.
    pub fn time_mix(
        &self,
        state: &LazyTensor<f32>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<f16> {
        LazyTensor::time_mix_v7(
            &self.r,
            &self.w,
            &self.n,
            &self.x,
            state,
            batch_index,
            num_tokens,
        )
    }

    /// Apply the V7 time_first operation.
    ///
    /// This adds the "first token bonus" to the time_mix output.
    pub fn time_first(&self, u: &LazyTensor<f16>, att_x: &LazyTensor<f16>) -> LazyTensor<f16> {
        LazyTensor::time_first_v7(u, &self.r, &self.n, att_x)
    }

    /// Finalize the attention output after time_mix and time_first.
    ///
    /// This applies group normalization, gating, and output projection.
    pub fn finalize(&self, att_x: &LazyTensor<f16>, residual: &LazyTensor<f16>) -> LazyTensor<f16> {
        let normed = att_x.group_norm(&self.gn_w, &self.gn_b, self.gn_eps);

        let gated = self.g.mul(&normed);

        let att_o = gated.matmul(self.w_o, Activation::None, self.turbo);

        residual.add(&att_o)
    }

    /// Complete the full attention computation in one call.
    ///
    /// This combines time_mix, time_first, and finalize into a single method.
    pub fn complete(
        &self,
        u: &LazyTensor<f16>,
        state: &LazyTensor<f32>,
        residual: &LazyTensor<f16>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<f16> {
        let att_x = self.time_mix(state, batch_index, num_tokens);
        let att_x = self.time_first(u, &att_x);
        self.finalize(&att_x, residual)
    }
}

/// Lazy V7 FFN layer parameters.
pub struct LazyV7Ffn<'a> {
    pub ln: LazyLayerNorm,
    pub x_k: LazyTensor<f16>,
    pub w_k: &'a Matrix,
    pub w_v: &'a Matrix,
}

impl<'a> LazyV7Ffn<'a> {
    /// Forward pass through the V7 FFN block.
    ///
    /// This builds a lazy compute graph for the FFN computation.
    /// Note: channel_mix_v7 is not yet implemented as a lazy op.
    pub fn forward(
        &self,
        x: &LazyTensor<f16>,
        state: &LazyTensor<f32>,
        eps: f32,
        turbo: bool,
    ) -> LazyV7FfnOutput {
        let ffn_x = x.blit();
        let normed = self.ln.forward(&ffn_x, eps);

        let ffn_kx = normed.token_shift(state, &self.x_k);

        let ffn_k = ffn_kx.matmul(self.w_k, Activation::SquaredRelu, turbo);

        let ffn_v = ffn_k.matmul(self.w_v, Activation::None, turbo);

        LazyV7FfnOutput {
            v: ffn_v,
            x: normed,
            residual: x.clone(),
        }
    }
}

/// Output from lazy V7 FFN forward pass.
pub struct LazyV7FfnOutput {
    pub v: LazyTensor<f16>,
    pub x: LazyTensor<f16>,
    pub residual: LazyTensor<f16>,
}

impl LazyV7FfnOutput {
    /// Apply the V7 channel_mix operation.
    ///
    /// This updates the FFN state and produces the output.
    pub fn channel_mix(
        &self,
        state: &LazyTensor<f32>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<f16> {
        LazyTensor::channel_mix_v7(&self.v, &self.x, state, batch_index, num_tokens)
    }

    /// Finalize the FFN output after channel_mix.
    pub fn finalize(&self, ffn_x: &LazyTensor<f16>) -> LazyTensor<f16> {
        self.residual.add(ffn_x)
    }

    /// Complete the full FFN computation in one call.
    ///
    /// This combines channel_mix and finalize into a single method.
    pub fn complete(
        &self,
        state: &LazyTensor<f32>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<f16> {
        let ffn_x = self.channel_mix(state, batch_index, num_tokens);
        self.finalize(&ffn_x)
    }
}

/// Configuration for lazy V7 inference.
#[derive(Debug, Clone)]
pub struct LazyV7Config {
    pub ln_eps: f32,
    pub gn_eps: f32,
    pub l2_eps: f32,
    pub rescale: usize,
}

impl Default for LazyV7Config {
    fn default() -> Self {
        Self {
            ln_eps: 1.0e-5,
            gn_eps: 64.0e-5,
            l2_eps: 1.0e-12,
            rescale: 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: These tests require GPU context and would be run as integration tests
    // See examples/test_lazy_tensor.rs for integration tests
}

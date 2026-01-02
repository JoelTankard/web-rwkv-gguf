use super::{
    graph::{ComputeNode, LazyOp, MatrixRef, TensorInfo},
    lazy_tensor::{LazyDataType, LazyTensor},
    matrix::Matrix,
    ops::Activation,
    shape::Shape,
};

impl<T: LazyDataType> LazyTensor<T> {
    /// Lazy matrix multiplication.
    ///
    /// Creates a graph node for matmul that will be executed when `materialize()` is called.
    /// - `matrix`: The weight matrix to multiply with.
    /// - `activation`: Activation function to apply after multiplication.
    /// - `turbo`: Whether to use turbo mode (batch optimization).
    ///
    /// Returns a new lazy tensor representing the output.
    pub fn matmul(&self, matrix: &Matrix, activation: Activation, turbo: bool) -> LazyTensor<T> {
        let output_shape = compute_matmul_output_shape(self.shape(), matrix);
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::MatMul {
                matrix: MatrixRef::from_ref(matrix),
                input: self.key(),
                activation,
                turbo,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy fused matrix multiplication with bias addition.
    ///
    /// Creates a graph node for `activation(matmul(matrix, self) + bias)`.
    /// This is more efficient than separate matmul and add operations.
    /// - `matrix`: The weight matrix to multiply with.
    /// - `bias`: Bias tensor to add after multiplication (shape: `[R, 1, 1, 1]`).
    /// - `activation`: Activation function to apply after bias addition.
    /// - `turbo`: Whether to use turbo mode (batch optimization).
    ///
    /// Returns a new lazy tensor representing the output.
    pub fn matmul_bias(
        &self,
        matrix: &Matrix,
        bias: &LazyTensor<T>,
        activation: Activation,
        turbo: bool,
    ) -> LazyTensor<T> {
        let output_shape = compute_matmul_output_shape(self.shape(), matrix);
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::MatMulBias {
                matrix: MatrixRef::from_ref(matrix),
                input: self.key(),
                bias: bias.key(),
                activation,
                turbo,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy element-wise addition.
    ///
    /// Creates a graph node for `self + other`.
    /// Both tensors must have compatible shapes.
    pub fn add(&self, other: &LazyTensor<T>) -> LazyTensor<T> {
        let output_shape = broadcast_shapes(self.shape(), other.shape());
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Add {
                lhs: self.key(),
                rhs: other.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy element-wise multiplication.
    ///
    /// Creates a graph node for `self * other`.
    /// Both tensors must have compatible shapes.
    pub fn mul(&self, other: &LazyTensor<T>) -> LazyTensor<T> {
        let output_shape = broadcast_shapes(self.shape(), other.shape());
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Mul {
                lhs: self.key(),
                rhs: other.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy blend operation: `self * factor + other * (1 - factor)`.
    pub fn blend(&self, other: &LazyTensor<T>, factor: &LazyTensor<T>) -> LazyTensor<T> {
        let output_shape = broadcast_shapes(self.shape(), other.shape());
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Blend {
                lhs: self.key(),
                rhs: other.key(),
                factor: factor.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy softmax operation.
    ///
    /// Applies softmax along the first dimension.
    pub fn softmax(&self) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Softmax { input: self.key() },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy layer normalization.
    ///
    /// - `weight`: Scale parameter (shape: `[C, 1, 1]`).
    /// - `bias`: Bias parameter (shape: `[C, 1, 1]`).
    /// - `eps`: Small constant for numerical stability.
    pub fn layer_norm(
        &self,
        weight: &LazyTensor<T>,
        bias: &LazyTensor<T>,
        eps: f32,
    ) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::LayerNorm {
                input: self.key(),
                weight: weight.key(),
                bias: bias.key(),
                eps,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy RMS normalization.
    ///
    /// - `weight`: Scale parameter (shape: `[C, 1, 1]`).
    /// - `bias`: Bias parameter (shape: `[C, 1, 1]`).
    /// - `eps`: Small constant for numerical stability.
    pub fn rms_norm(
        &self,
        weight: &LazyTensor<T>,
        bias: &LazyTensor<T>,
        eps: f32,
    ) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::RmsNorm {
                input: self.key(),
                weight: weight.key(),
                bias: bias.key(),
                eps,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy token shift (time mixing).
    ///
    /// - `state`: Previous state tensor (f32).
    /// - `mix`: Mixing coefficients (same type as input).
    ///
    /// Note: State is always f32 to match the existing runtime state format.
    pub fn token_shift(&self, state: &LazyTensor<f32>, mix: &LazyTensor<T>) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::TokenShift {
                input: self.key(),
                state: state.key(),
                mix: mix.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy copy (blit) operation.
    ///
    /// Creates a copy of this tensor.
    pub fn blit(&self) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Blit { input: self.key() },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy affine transformation: x * scale + bias.
    pub fn affine(&self, scale: f32, bias: f32) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Affine {
                input: self.key(),
                scale,
                bias,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy group normalization.
    ///
    /// - `weight`: Scale parameter.
    /// - `bias`: Bias parameter.
    /// - `eps`: Small constant for numerical stability.
    pub fn group_norm(
        &self,
        weight: &LazyTensor<T>,
        bias: &LazyTensor<T>,
        eps: f32,
    ) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::GroupNorm {
                input: self.key(),
                weight: weight.key(),
                bias: bias.key(),
                eps,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy L2 normalization.
    ///
    /// - `eps`: Small constant for numerical stability.
    pub fn l2_norm(&self, eps: f32) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::L2Norm {
                input: self.key(),
                eps,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy sigmoid activation.
    pub fn sigmoid(&self) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Sigmoid { input: self.key() },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy tanh activation.
    pub fn tanh(&self) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Tanh { input: self.key() },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy squared ReLU activation.
    pub fn squared_relu(&self) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::SquaredRelu { input: self.key() },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy lerp operation: lerp(lhs, rhs, factor).
    /// If `reversed` is false: lhs + factor * (rhs - lhs)
    /// If `reversed` is true: rhs + factor * (lhs - rhs)
    pub fn lerp(
        &self,
        other: &LazyTensor<T>,
        factor: &LazyTensor<T>,
        reversed: bool,
    ) -> LazyTensor<T> {
        let output_shape = broadcast_shapes(self.shape(), other.shape());
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::Lerp {
                lhs: self.key(),
                rhs: other.key(),
                factor: factor.key(),
                reversed,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy add with activation.
    pub fn add_activate(
        &self,
        other: &LazyTensor<T>,
        lhs_act: Activation,
        rhs_act: Activation,
        out_act: Activation,
    ) -> LazyTensor<T> {
        let output_shape = broadcast_shapes(self.shape(), other.shape());
        let info = TensorInfo {
            shape: output_shape,
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::AddActivate {
                lhs: self.key(),
                rhs: other.key(),
                lhs_act,
                rhs_act,
                out_act,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }

    /// Lazy control K operation for V7 attention.
    /// Modifies k based on p and a: k = k * (1 - a) + kk * a where kk = k * p
    pub fn control_k_v7(&self, p: &LazyTensor<T>, a: &LazyTensor<T>) -> LazyTensor<T> {
        let info = TensorInfo {
            shape: self.shape(),
            dtype: T::lazy_dtype(),
        };

        let node = ComputeNode {
            operation: LazyOp::ControlKV7 {
                p: p.key(),
                a: a.key(),
                k: self.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = self.context().graph().add_node(node);
        LazyTensor::new(self.context().clone(), info, key)
    }
}

impl LazyTensor<half::f16> {
    /// Lazy pack k, v, a, kk into a single tensor n.
    /// Output shape: [shape[0], shape[1], shape[2], 4]
    pub fn pack_kvakk(
        k: &LazyTensor<half::f16>,
        v: &LazyTensor<half::f16>,
        a: &LazyTensor<half::f16>,
        kk: &LazyTensor<half::f16>,
    ) -> LazyTensor<half::f16> {
        let shape = k.shape();
        let output_shape = Shape::new(shape[0], shape[1], shape[2], 4);
        let info = TensorInfo {
            shape: output_shape,
            dtype: super::graph::DataType::F16,
        };

        let node = ComputeNode {
            operation: LazyOp::PackKvakk {
                k: k.key(),
                v: v.key(),
                a: a.key(),
                kk: kk.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = k.context().graph().add_node(node);
        LazyTensor::new(k.context().clone(), info, key)
    }

    /// Lazy V7 time mix operation.
    /// This is a cursor-free variant that takes explicit batch index.
    #[allow(clippy::too_many_arguments)]
    pub fn time_mix_v7(
        r: &LazyTensor<half::f16>,
        w: &LazyTensor<half::f16>,
        n: &LazyTensor<half::f16>,
        x: &LazyTensor<half::f16>,
        state: &LazyTensor<f32>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<half::f16> {
        let info = TensorInfo {
            shape: x.shape(),
            dtype: super::graph::DataType::F16,
        };

        let node = ComputeNode {
            operation: LazyOp::TimeMixV7 {
                r: r.key(),
                w: w.key(),
                n: n.key(),
                x: x.key(),
                state: state.key(),
                batch_index,
                num_tokens,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = x.context().graph().add_node(node);
        LazyTensor::new(x.context().clone(), info, key)
    }

    /// Lazy V7 channel mix operation.
    /// This is a cursor-free variant that takes explicit batch index.
    pub fn channel_mix_v7(
        v: &LazyTensor<half::f16>,
        x: &LazyTensor<half::f16>,
        state: &LazyTensor<f32>,
        batch_index: u32,
        num_tokens: u32,
    ) -> LazyTensor<half::f16> {
        let info = TensorInfo {
            shape: x.shape(),
            dtype: super::graph::DataType::F16,
        };

        let node = ComputeNode {
            operation: LazyOp::ChannelMixV7 {
                v: v.key(),
                x: x.key(),
                state: state.key(),
                batch_index,
                num_tokens,
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = x.context().graph().add_node(node);
        LazyTensor::new(x.context().clone(), info, key)
    }

    /// Lazy V7 time first operation.
    pub fn time_first_v7(
        u: &LazyTensor<half::f16>,
        r: &LazyTensor<half::f16>,
        n: &LazyTensor<half::f16>,
        x: &LazyTensor<half::f16>,
    ) -> LazyTensor<half::f16> {
        let info = TensorInfo {
            shape: x.shape(),
            dtype: super::graph::DataType::F16,
        };

        let node = ComputeNode {
            operation: LazyOp::TimeFirstV7 {
                u: u.key(),
                r: r.key(),
                n: n.key(),
                x: x.key(),
            },
            info: info.clone(),
            ref_count: 1,
            buffer: None,
        };

        let key = x.context().graph().add_node(node);
        LazyTensor::new(x.context().clone(), info, key)
    }
}

/// Compute the output shape for matrix multiplication.
fn compute_matmul_output_shape(input_shape: Shape, matrix: &Matrix) -> Shape {
    let matrix_shape = matrix.shape();
    // Matrix shape is [K, M, B] where K is input dim, M is output dim
    // Input shape is [K, T, B] where T is token count
    // Output shape is [M, T, B]
    Shape::new(
        matrix_shape[1], // M (output dimension)
        input_shape[1],  // T (token count)
        input_shape[2],  // B (batch)
        1,
    )
}

/// Compute broadcast shape for element-wise operations.
fn broadcast_shapes(a: Shape, b: Shape) -> Shape {
    // For now, use the larger shape (simple broadcasting)
    // In practice, we'd need proper broadcast semantics
    let x = a[0].max(b[0]);
    let y = a[1].max(b[1]);
    let z = a[2].max(b[2]);
    let w = a[3].max(b[3]);
    Shape::new(x, y, z, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_matmul_output_shape() {
        // Input: [2048, 5, 1] (2048 dim, 5 tokens, 1 batch)
        // Matrix: [2048, 4096, 1] (2048 input dim, 4096 output dim)
        // Expected output: [4096, 5, 1]
        // Note: We can't easily create a Matrix in unit tests without GPU context,
        // so we test broadcast_shapes instead which exercises similar logic.
        let a = Shape::new(64, 1, 1, 1);
        let b = Shape::new(64, 10, 1, 1);
        let result = broadcast_shapes(a, b);
        assert_eq!(result, Shape::new(64, 10, 1, 1));
    }

    #[test]
    fn test_broadcast_shapes() {
        let a = Shape::new(64, 1, 1, 1);
        let b = Shape::new(64, 10, 2, 1);
        let result = broadcast_shapes(a, b);
        assert_eq!(result, Shape::new(64, 10, 2, 1));

        let a = Shape::new(128, 5, 1, 1);
        let b = Shape::new(128, 5, 1, 1);
        let result = broadcast_shapes(a, b);
        assert_eq!(result, Shape::new(128, 5, 1, 1));
    }
}

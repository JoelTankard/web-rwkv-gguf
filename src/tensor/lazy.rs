use std::sync::Arc;

use half::f16;

use super::{
    kind::ReadWrite, matrix::Matrix, shape::Shape, TensorCpu, TensorError, TensorGpu, TensorInit,
    TensorInto,
};
use crate::context::Context;

/// Lazy matrix that defers GPU upload until first use.
/// This is the main optimization for fast model loading.
#[derive(Debug, Clone)]
pub enum LazyMatrix {
    /// Already loaded to GPU (for eager loading path or after first access).
    Loaded(Matrix),
    /// CPU-resident data waiting to be uploaded.
    Pending(LazyMatrixData),
}

/// CPU-resident matrix data waiting for GPU upload.
#[derive(Debug, Clone)]
pub struct LazyMatrixData {
    pub context: Context,
    pub variant: LazyMatrixVariant,
}

/// The different matrix formats that can be lazily loaded.
#[derive(Debug, Clone)]
pub enum LazyMatrixVariant {
    Fp16(TensorCpu<f16>),
    Int8 {
        w: Vec<u8>,
        m: Vec<f16>,
        shape: Shape,
    },
    Q8_0 {
        raw_data: Arc<[u8]>,
        shape: Shape,
    },
}

impl LazyMatrix {
    /// Create a lazy matrix from an already-loaded GPU matrix.
    pub fn loaded(matrix: Matrix) -> Self {
        LazyMatrix::Loaded(matrix)
    }

    /// Create a lazy Fp16 matrix from CPU data.
    pub fn fp16_pending(context: Context, tensor: TensorCpu<f16>) -> Self {
        LazyMatrix::Pending(LazyMatrixData {
            context,
            variant: LazyMatrixVariant::Fp16(tensor),
        })
    }

    /// Create a lazy Q8_0 matrix from raw GGUF data.
    pub fn q8_0_pending(context: Context, raw_data: Arc<[u8]>, shape: Shape) -> Self {
        LazyMatrix::Pending(LazyMatrixData {
            context,
            variant: LazyMatrixVariant::Q8_0 { raw_data, shape },
        })
    }

    /// Get the loaded matrix, uploading to GPU if necessary.
    /// This consumes the lazy matrix and returns the loaded version.
    pub fn ensure_loaded(self) -> Result<Matrix, TensorError> {
        match self {
            LazyMatrix::Loaded(m) => Ok(m),
            LazyMatrix::Pending(data) => data.upload(),
        }
    }

    /// Check if the matrix is already loaded to GPU.
    pub fn is_loaded(&self) -> bool {
        matches!(self, LazyMatrix::Loaded(_))
    }
}

impl LazyMatrixData {
    /// Upload the CPU data to GPU and create the appropriate Matrix variant.
    pub fn upload(self) -> Result<Matrix, TensorError> {
        let context = &self.context;
        match self.variant {
            LazyMatrixVariant::Fp16(tensor) => {
                let gpu_tensor = tensor.to(context);
                Ok(Matrix::Fp16(gpu_tensor))
            }
            LazyMatrixVariant::Int8 { w, m, shape } => {
                let w_tensor: TensorGpu<u8, ReadWrite> =
                    TensorCpu::from_data(shape, w)?.to(context);
                let m_shape = Shape::new(m.len(), 1, 1, 1);
                let m_tensor: TensorGpu<f16, ReadWrite> =
                    TensorCpu::from_data(m_shape, m)?.to(context);
                Ok(Matrix::Int8 {
                    w: w_tensor,
                    m: m_tensor,
                })
            }
            LazyMatrixVariant::Q8_0 { raw_data, shape } => {
                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorGpu::from_data_u8(context, block_data_shape, &raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q8_0 { w, s })
            }
        }
    }
}

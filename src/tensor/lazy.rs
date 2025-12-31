//! Lazy matrix implementation with interior mutability for deferred GPU upload.

use std::fmt;
use std::sync::{Arc, RwLock};

use half::f16;

use super::{
    kind::ReadWrite, matrix::Matrix, ops::Activation, shape::Shape, TensorCpu, TensorError,
    TensorGpu, TensorGpuView, TensorInit, TensorInto,
};
use crate::context::Context;
use crate::num::Float;

/// Type alias for shared byte data that can be used for zero-copy loading.
pub type SharedBytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// Lazy matrix that defers GPU upload until first use.
#[derive(Clone)]
pub struct LazyMatrix {
    inner: Arc<RwLock<LazyMatrixInner>>,
}

enum LazyMatrixInner {
    Loaded(Matrix),
    Pending(LazyMatrixData),
}

struct LazyMatrixData {
    context: Context,
    variant: LazyMatrixVariant,
}

enum LazyMatrixVariant {
    Fp16(TensorCpu<f16>),
    Int8 { w: Vec<u8>, m: Vec<f16>, shape: Shape },
    Q4K { raw_data: Arc<[u8]>, shape: Shape },
    Q5K { raw_data: Arc<[u8]>, shape: Shape },
    Q6K { raw_data: Arc<[u8]>, shape: Shape },
    Q8_0 { raw_data: Arc<[u8]>, shape: Shape },
    Q4KZeroCopy { shared: SharedBytes, offset: usize, len: usize, shape: Shape },
    Q5KZeroCopy { shared: SharedBytes, offset: usize, len: usize, shape: Shape },
    Q6KZeroCopy { shared: SharedBytes, offset: usize, len: usize, shape: Shape },
    Q8_0ZeroCopy { shared: SharedBytes, offset: usize, len: usize, shape: Shape },
    Fp16ZeroCopy { shared: SharedBytes, offset: usize, len: usize, shape: Shape },
}

impl fmt::Debug for LazyMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.read().unwrap();
        match &*inner {
            LazyMatrixInner::Loaded(m) => f.debug_tuple("LazyMatrix::Loaded").field(m).finish(),
            LazyMatrixInner::Pending(data) => f.debug_struct("LazyMatrix::Pending").field("variant", &data.variant).finish(),
        }
    }
}

impl fmt::Debug for LazyMatrixVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp16(t) => f.debug_tuple("Fp16").field(t).finish(),
            Self::Int8 { shape, .. } => f.debug_struct("Int8").field("shape", shape).finish(),
            Self::Q4K { shape, .. } => f.debug_struct("Q4K").field("shape", shape).finish(),
            Self::Q5K { shape, .. } => f.debug_struct("Q5K").field("shape", shape).finish(),
            Self::Q6K { shape, .. } => f.debug_struct("Q6K").field("shape", shape).finish(),
            Self::Q8_0 { shape, .. } => f.debug_struct("Q8_0").field("shape", shape).finish(),
            Self::Q4KZeroCopy { offset, len, shape, .. } => f.debug_struct("Q4KZeroCopy").field("offset", offset).field("len", len).field("shape", shape).finish(),
            Self::Q5KZeroCopy { offset, len, shape, .. } => f.debug_struct("Q5KZeroCopy").field("offset", offset).field("len", len).field("shape", shape).finish(),
            Self::Q6KZeroCopy { offset, len, shape, .. } => f.debug_struct("Q6KZeroCopy").field("offset", offset).field("len", len).field("shape", shape).finish(),
            Self::Q8_0ZeroCopy { offset, len, shape, .. } => f.debug_struct("Q8_0ZeroCopy").field("offset", offset).field("len", len).field("shape", shape).finish(),
            Self::Fp16ZeroCopy { offset, len, shape, .. } => f.debug_struct("Fp16ZeroCopy").field("offset", offset).field("len", len).field("shape", shape).finish(),
        }
    }
}

impl LazyMatrix {
    pub fn loaded(matrix: Matrix) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Loaded(matrix))) }
    }

    pub fn fp16_pending(context: Context, tensor: TensorCpu<f16>) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Fp16(tensor) }))) }
    }

    pub fn q4k_pending(context: Context, raw_data: Arc<[u8]>, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q4K { raw_data, shape } }))) }
    }

    pub fn q5k_pending(context: Context, raw_data: Arc<[u8]>, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q5K { raw_data, shape } }))) }
    }

    pub fn q6k_pending(context: Context, raw_data: Arc<[u8]>, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q6K { raw_data, shape } }))) }
    }

    pub fn q8_0_pending(context: Context, raw_data: Arc<[u8]>, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q8_0 { raw_data, shape } }))) }
    }

    pub fn q4k_zero_copy(context: Context, shared: SharedBytes, offset: usize, len: usize, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q4KZeroCopy { shared, offset, len, shape } }))) }
    }

    pub fn q5k_zero_copy(context: Context, shared: SharedBytes, offset: usize, len: usize, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q5KZeroCopy { shared, offset, len, shape } }))) }
    }

    pub fn q6k_zero_copy(context: Context, shared: SharedBytes, offset: usize, len: usize, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q6KZeroCopy { shared, offset, len, shape } }))) }
    }

    pub fn q8_0_zero_copy(context: Context, shared: SharedBytes, offset: usize, len: usize, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Q8_0ZeroCopy { shared, offset, len, shape } }))) }
    }

    pub fn fp16_zero_copy(context: Context, shared: SharedBytes, offset: usize, len: usize, shape: Shape) -> Self {
        LazyMatrix { inner: Arc::new(RwLock::new(LazyMatrixInner::Pending(LazyMatrixData { context, variant: LazyMatrixVariant::Fp16ZeroCopy { shared, offset, len, shape } }))) }
    }

    pub fn is_loaded(&self) -> bool {
        let inner = self.inner.read().unwrap();
        matches!(&*inner, LazyMatrixInner::Loaded(_))
    }

    pub fn ensure_loaded_matrix(&self) -> Result<Matrix, TensorError> {
        self.ensure_loaded()?;
        let inner = self.inner.read().unwrap();
        match &*inner {
            LazyMatrixInner::Loaded(m) => Ok(m.clone()),
            LazyMatrixInner::Pending(_) => unreachable!(),
        }
    }

    fn ensure_loaded(&self) -> Result<(), TensorError> {
        {
            let inner = self.inner.read().unwrap();
            if matches!(&*inner, LazyMatrixInner::Loaded(_)) {
                return Ok(());
            }
        }
        let mut inner = self.inner.write().unwrap();
        if let LazyMatrixInner::Pending(data) = &*inner {
            let ctx = data.context.clone();
            drop(inner);
            let mut inner = self.inner.write().unwrap();
            if let LazyMatrixInner::Pending(data) = std::mem::replace(&mut *inner, LazyMatrixInner::Pending(LazyMatrixData {
                context: ctx,
                variant: LazyMatrixVariant::Fp16(TensorCpu::from_data(Shape::new(1,1,1,1), vec![f16::ZERO]).unwrap()),
            })) {
                let uploaded = data.upload()?;
                *inner = LazyMatrixInner::Loaded(uploaded);
            }
        }
        Ok(())
    }

    pub fn matmul_op<'a, 'b, F0: Float, F1: Float>(
        &self,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
        turbo: bool,
    ) -> Result<super::ops::TensorOp, TensorError> {
        self.ensure_loaded()?;
        let inner = self.inner.read().unwrap();
        match &*inner {
            LazyMatrixInner::Loaded(matrix) => matrix.matmul_op(input, output, act, turbo),
            LazyMatrixInner::Pending(_) => unreachable!(),
        }
    }

    pub fn matmul_op_sparse<'a, 'b, F0: Float, F1: Float>(
        &self,
        input: impl Into<TensorGpuView<'a, F0>>,
        output: impl Into<TensorGpuView<'b, F1>>,
        act: Activation,
        turbo: bool,
    ) -> Result<super::ops::TensorOp, TensorError> {
        self.ensure_loaded()?;
        let inner = self.inner.read().unwrap();
        match &*inner {
            LazyMatrixInner::Loaded(matrix) => matrix.matmul_op_sparse(input, output, act, turbo),
            LazyMatrixInner::Pending(_) => unreachable!(),
        }
    }
}

impl LazyMatrixData {
    fn upload(self) -> Result<Matrix, TensorError> {
        let context = &self.context;
        match self.variant {
            LazyMatrixVariant::Fp16(tensor) => Ok(Matrix::Fp16(tensor.to(context))),
            LazyMatrixVariant::Int8 { w, m, shape } => {
                let w_tensor: TensorGpu<u8, ReadWrite> = TensorCpu::from_data(shape, w)?.to(context);
                let m_shape = Shape::new(m.len(), 1, 1, 1);
                let m_tensor: TensorGpu<f16, ReadWrite> = TensorCpu::from_data(m_shape, m)?.to(context);
                Ok(Matrix::Int8 { w: w_tensor, m: m_tensor })
            }
            LazyMatrixVariant::Q4K { raw_data, shape } => {
                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, &raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q4K { w, s })
            }
            LazyMatrixVariant::Q5K { raw_data, shape } => {
                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, &raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q5K { w, s })
            }
            LazyMatrixVariant::Q6K { raw_data, shape } => {
                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, &raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q6K { w, s })
            }
            LazyMatrixVariant::Q8_0 { raw_data, shape } => {
                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, &raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q8_0 { w, s })
            }
            LazyMatrixVariant::Q4KZeroCopy { shared, offset, len, shape } => {
                let data: &[u8] = (*shared).as_ref();
                let raw_data = &data[offset..offset + len];
                let block_data_shape = Shape::new(len, 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q4K { w, s })
            }
            LazyMatrixVariant::Q5KZeroCopy { shared, offset, len, shape } => {
                let data: &[u8] = (*shared).as_ref();
                let raw_data = &data[offset..offset + len];
                let block_data_shape = Shape::new(len, 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q5K { w, s })
            }
            LazyMatrixVariant::Q6KZeroCopy { shared, offset, len, shape } => {
                let data: &[u8] = (*shared).as_ref();
                let raw_data = &data[offset..offset + len];
                let block_data_shape = Shape::new(len, 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q6K { w, s })
            }
            LazyMatrixVariant::Q8_0ZeroCopy { shared, offset, len, shape } => {
                let data: &[u8] = (*shared).as_ref();
                let raw_data = &data[offset..offset + len];
                let block_data_shape = Shape::new(len, 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> = TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);
                Ok(Matrix::Q8_0 { w, s })
            }
            LazyMatrixVariant::Fp16ZeroCopy { shared, offset, len, shape } => {
                let data: &[u8] = (*shared).as_ref();
                let raw_data = &data[offset..offset + len];
                let f16_data: &[f16] = bytemuck::cast_slice(raw_data);
                let tensor: TensorCpu<f16> = TensorCpu::from_data(shape, f16_data.to_vec())?;
                let gpu_tensor = tensor.to(context);
                Ok(Matrix::Fp16(gpu_tensor))
            }
        }
    }
}

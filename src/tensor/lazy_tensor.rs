use std::{marker::PhantomData, sync::Arc};

use wgpu::Buffer;

use super::{
    graph::{ComputeNode, DataType, LazyOp, NodeIndex, TensorInfo},
    kind::ReadWrite,
    shape::Shape,
    TensorError, TensorGpu, TensorShape,
};
use crate::{context::Context, num::Float};

/// Extension trait to get the lazy DataType for a Float type.
pub trait LazyDataType: Float {
    fn lazy_dtype() -> DataType;
}

impl LazyDataType for half::f16 {
    fn lazy_dtype() -> DataType {
        DataType::F16
    }
}

impl LazyDataType for f32 {
    fn lazy_dtype() -> DataType {
        DataType::F32
    }
}

/// A tensor that defers computation until materialized.
///
/// `LazyTensor` represents a node in a compute graph. Operations on lazy tensors
/// don't execute immediately - instead they build up a graph of operations.
/// The actual GPU computation only happens when `materialize()` is called.
///
/// This enables:
/// - Better memory planning (know full graph before allocating)
/// - Automatic intermediate buffer reuse
/// - Potential for operation fusion
/// - Reduced memory pressure during inference
#[derive(Debug)]
pub struct LazyTensor<T: LazyDataType> {
    context: Context,
    info: TensorInfo,
    key: NodeIndex,
    _phantom: PhantomData<T>,
}

impl<T: LazyDataType> LazyTensor<T> {
    /// Create a new lazy tensor from a node in the compute graph.
    pub(crate) fn new(context: Context, info: TensorInfo, key: NodeIndex) -> Self {
        Self {
            context,
            info,
            key,
            _phantom: PhantomData,
        }
    }

    /// Create a lazy tensor from an existing GPU tensor.
    /// This wraps the tensor as an Input node in the compute graph.
    pub fn from_gpu(tensor: &TensorGpu<T, ReadWrite>) -> Self {
        let context = tensor.context().clone();
        let shape = tensor.shape();
        let dtype = T::lazy_dtype();

        let node = ComputeNode {
            operation: LazyOp::Input {
                buffer: tensor.buffer.clone(),
            },
            info: TensorInfo { shape, dtype },
            ref_count: 1,
            buffer: Some(tensor.buffer.clone()),
        };

        let key = context.graph().add_node(node);

        Self {
            context,
            info: TensorInfo { shape, dtype },
            key,
            _phantom: PhantomData,
        }
    }

    /// Get the shape of this tensor without materializing.
    pub fn shape(&self) -> Shape {
        self.info.shape
    }

    /// Get the data type of this tensor.
    pub fn dtype(&self) -> DataType {
        self.info.dtype
    }

    /// Get the node index in the compute graph.
    pub fn key(&self) -> NodeIndex {
        self.key
    }

    /// Get a reference to the context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Execute the compute graph and return the GPU buffer.
    ///
    /// This performs a topological sort of all operations needed to compute
    /// this tensor, allocates buffers, and executes the operations.
    pub fn materialize(&self) -> Result<Arc<Buffer>, TensorError> {
        self.context.graph().resolve(self.key, &self.context)
    }

    /// Check if this tensor's result is already cached.
    pub fn is_cached(&self) -> bool {
        self.context
            .graph()
            .get(self.key)
            .map(|node| node.buffer.is_some())
            .unwrap_or(false)
    }
}

impl<T: LazyDataType> Clone for LazyTensor<T> {
    fn clone(&self) -> Self {
        self.context.graph().add_reference(self.key);
        Self {
            context: self.context.clone(),
            info: self.info.clone(),
            key: self.key,
            _phantom: PhantomData,
        }
    }
}

impl<T: LazyDataType> Drop for LazyTensor<T> {
    fn drop(&mut self) {
        self.context.graph().remove_reference(self.key);
    }
}

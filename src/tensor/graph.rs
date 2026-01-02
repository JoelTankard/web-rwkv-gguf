use std::{collections::HashMap, sync::Arc};

use half::f16;
use wgpu::{Buffer, BufferUsages};

use super::{
    kind::ReadWrite,
    matrix::Matrix,
    ops::{Activation, TensorOp},
    shape::Shape,
    TensorError, TensorErrorKind, TensorGpu, TensorInitContext,
};
use crate::context::Context;

/// Index into the compute graph's node storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIndex(pub(crate) usize);

impl NodeIndex {
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Metadata about a tensor's shape and data type.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Shape,
    pub dtype: DataType,
}

/// Supported data types for lazy tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F16,
    F32,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::F16 => 2,
            DataType::F32 => 4,
        }
    }
}

/// A reference to a matrix for lazy operations.
/// We use Arc to allow cheap cloning in the graph.
#[derive(Debug, Clone)]
pub enum MatrixRef {
    Owned(Arc<Matrix>),
    Borrowed(*const Matrix),
}

// SAFETY: Matrix is Send+Sync, and we only use the pointer during graph execution
// when the original Matrix is guaranteed to be alive.
unsafe impl Send for MatrixRef {}
unsafe impl Sync for MatrixRef {}

impl MatrixRef {
    pub fn from_ref(matrix: &Matrix) -> Self {
        MatrixRef::Borrowed(matrix as *const Matrix)
    }

    pub fn from_owned(matrix: Matrix) -> Self {
        MatrixRef::Owned(Arc::new(matrix))
    }

    /// Get a reference to the underlying matrix.
    ///
    /// # Safety
    /// For borrowed references, the caller must ensure the original Matrix
    /// is still alive.
    pub unsafe fn get(&self) -> &Matrix {
        match self {
            MatrixRef::Owned(arc) => arc.as_ref(),
            MatrixRef::Borrowed(ptr) => &**ptr,
        }
    }
}

/// Operations that can be lazily evaluated.
#[derive(Debug, Clone)]
pub enum LazyOp {
    /// Input tensor - already has a GPU buffer.
    Input { buffer: Arc<Buffer> },

    /// Matrix multiplication.
    MatMul {
        matrix: MatrixRef,
        input: NodeIndex,
        activation: Activation,
        turbo: bool,
    },

    /// Element-wise addition: lhs + rhs.
    Add { lhs: NodeIndex, rhs: NodeIndex },

    /// Element-wise multiplication: lhs * rhs.
    Mul { lhs: NodeIndex, rhs: NodeIndex },

    /// Blend: lhs * factor + rhs * (1 - factor).
    Blend {
        lhs: NodeIndex,
        rhs: NodeIndex,
        factor: NodeIndex,
    },

    /// Layer normalization.
    LayerNorm {
        input: NodeIndex,
        weight: NodeIndex,
        bias: NodeIndex,
        eps: f32,
    },

    /// RMS normalization.
    RmsNorm {
        input: NodeIndex,
        weight: NodeIndex,
        bias: NodeIndex,
        eps: f32,
    },

    /// Softmax.
    Softmax { input: NodeIndex },

    /// Token shift (time mixing).
    TokenShift {
        input: NodeIndex,
        state: NodeIndex,
        mix: NodeIndex,
    },

    /// Copy tensor data (blit).
    Blit { input: NodeIndex },

    /// Affine transformation: x * scale + bias.
    Affine {
        input: NodeIndex,
        scale: f32,
        bias: f32,
    },

    /// Group normalization.
    GroupNorm {
        input: NodeIndex,
        weight: NodeIndex,
        bias: NodeIndex,
        eps: f32,
    },

    /// L2 normalization.
    L2Norm { input: NodeIndex, eps: f32 },

    /// Sigmoid activation.
    Sigmoid { input: NodeIndex },

    /// Tanh activation.
    Tanh { input: NodeIndex },

    /// Squared ReLU activation.
    SquaredRelu { input: NodeIndex },

    /// V7 Time Mix operation (WKV kernel).
    /// This is a cursor-free variant that operates on explicit batch index.
    /// - `r`: Receptance tensor [head_size, num_head, num_token, 1]
    /// - `w`: Time decay tensor [head_size, num_head, num_token, 1]
    /// - `n`: Packed k,v,a,kk tensor [head_size, num_head, num_token, 4]
    /// - `x`: Input/output tensor [head_size, num_head, num_token, 1]
    /// - `state`: State tensor [num_emb, head_size+1, num_batch, 1] (f32)
    /// - `batch_index`: Explicit batch index for state access
    /// - `num_tokens`: Number of tokens in this batch
    TimeMixV7 {
        r: NodeIndex,
        w: NodeIndex,
        n: NodeIndex,
        x: NodeIndex,
        state: NodeIndex,
        batch_index: u32,
        num_tokens: u32,
    },

    /// V7 Channel Mix operation.
    /// This is a cursor-free variant that operates on explicit batch index.
    /// - `v`: Value tensor after FFN key projection [num_emb, num_token, 1, 1]
    /// - `x`: Input tensor (will be modified) [num_emb, num_token, 1, 1]
    /// - `state`: FFN state tensor [num_emb, 1, num_batch, 1] (f32)
    /// - `batch_index`: Explicit batch index for state access
    /// - `num_tokens`: Number of tokens in this batch
    ChannelMixV7 {
        v: NodeIndex,
        x: NodeIndex,
        state: NodeIndex,
        batch_index: u32,
        num_tokens: u32,
    },

    /// Lerp operation: lerp(a, b, factor) = a + factor * (b - a)
    /// If `reversed` is true: lerp(b, a, factor) = b + factor * (a - b)
    Lerp {
        lhs: NodeIndex,
        rhs: NodeIndex,
        factor: NodeIndex,
        reversed: bool,
    },

    /// Add with activation: applies activation to each input before adding.
    AddActivate {
        lhs: NodeIndex,
        rhs: NodeIndex,
        lhs_act: Activation,
        rhs_act: Activation,
        out_act: Activation,
    },

    /// Control K operation for V7 attention.
    ControlKV7 {
        p: NodeIndex,
        a: NodeIndex,
        k: NodeIndex,
    },

    /// Pack k, v, a, kk into a single tensor n.
    PackKvakk {
        k: NodeIndex,
        v: NodeIndex,
        a: NodeIndex,
        kk: NodeIndex,
    },

    /// Time first operation for V7.
    TimeFirstV7 {
        u: NodeIndex,
        r: NodeIndex,
        n: NodeIndex,
        x: NodeIndex,
    },

    /// Fused matrix multiplication with bias addition.
    /// Combines: output = activation(matmul(matrix, input) + bias)
    /// This is an optimization that fuses MatMul → Add into a single kernel.
    MatMulBias {
        matrix: MatrixRef,
        input: NodeIndex,
        bias: NodeIndex,
        activation: Activation,
        turbo: bool,
    },
}

impl LazyOp {
    /// Get the input node indices this operation depends on.
    pub fn inputs(&self) -> Vec<NodeIndex> {
        match self {
            LazyOp::Input { .. } => vec![],
            LazyOp::MatMul { input, .. } => vec![*input],
            LazyOp::Add { lhs, rhs } => vec![*lhs, *rhs],
            LazyOp::Mul { lhs, rhs } => vec![*lhs, *rhs],
            LazyOp::Blend { lhs, rhs, factor } => vec![*lhs, *rhs, *factor],
            LazyOp::LayerNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            LazyOp::RmsNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            LazyOp::Softmax { input } => vec![*input],
            LazyOp::TokenShift { input, state, mix } => vec![*input, *state, *mix],
            LazyOp::Blit { input } => vec![*input],
            LazyOp::Affine { input, .. } => vec![*input],
            LazyOp::GroupNorm {
                input,
                weight,
                bias,
                ..
            } => vec![*input, *weight, *bias],
            LazyOp::L2Norm { input, .. } => vec![*input],
            LazyOp::Sigmoid { input } => vec![*input],
            LazyOp::Tanh { input } => vec![*input],
            LazyOp::SquaredRelu { input } => vec![*input],
            LazyOp::TimeMixV7 {
                r, w, n, x, state, ..
            } => vec![*r, *w, *n, *x, *state],
            LazyOp::ChannelMixV7 { v, x, state, .. } => vec![*v, *x, *state],
            LazyOp::Lerp {
                lhs, rhs, factor, ..
            } => vec![*lhs, *rhs, *factor],
            LazyOp::AddActivate { lhs, rhs, .. } => vec![*lhs, *rhs],
            LazyOp::ControlKV7 { p, a, k } => vec![*p, *a, *k],
            LazyOp::PackKvakk { k, v, a, kk } => vec![*k, *v, *a, *kk],
            LazyOp::TimeFirstV7 { u, r, n, x } => vec![*u, *r, *n, *x],
            LazyOp::MatMulBias { input, bias, .. } => vec![*input, *bias],
        }
    }
}

/// A node in the compute graph.
#[derive(Debug)]
pub struct ComputeNode {
    /// The operation that produces this node's output.
    pub operation: LazyOp,
    /// Shape and dtype of the output.
    pub info: TensorInfo,
    /// Reference count - how many LazyTensors point to this node.
    pub ref_count: usize,
    /// Cached GPU buffer (populated after materialization).
    pub buffer: Option<Arc<Buffer>>,
}

/// The compute graph for lazy tensor evaluation.
#[derive(Debug, Default)]
pub struct ComputeGraph {
    /// All nodes in the graph.
    nodes: Vec<Option<ComputeNode>>,
    /// Free list for reusing node slots after nodes are garbage collected.
    free_list: Vec<usize>,
    /// Generation counter for cache invalidation.
    generation: u64,
}

/// Buffer allocation plan for graph execution.
/// Maps node indices to pre-allocated or reused buffers.
#[derive(Debug, Default)]
pub struct BufferPlan {
    /// Assigned buffers for each node index.
    assignments: HashMap<usize, Arc<Buffer>>,
    /// Pool of available buffers keyed by (size, usage).
    available: HashMap<(u64, BufferUsages), Vec<Arc<Buffer>>>,
}

impl BufferPlan {
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            available: HashMap::new(),
        }
    }

    /// Assign a buffer to a node.
    pub fn assign(&mut self, idx: NodeIndex, buffer: Arc<Buffer>) {
        self.assignments.insert(idx.0, buffer);
    }

    /// Get the assigned buffer for a node.
    pub fn get(&self, idx: NodeIndex) -> Option<Arc<Buffer>> {
        self.assignments.get(&idx.0).cloned()
    }

    /// Return a buffer to the pool for potential reuse.
    pub fn release(&mut self, buffer: Arc<Buffer>) {
        let key = (buffer.size(), buffer.usage());
        self.available.entry(key).or_default().push(buffer);
    }

    /// Try to get a reusable buffer from the pool.
    pub fn try_reuse(&mut self, size: u64, usage: BufferUsages) -> Option<Arc<Buffer>> {
        self.available.get_mut(&(size, usage)).and_then(|v| v.pop())
    }
}

impl ComputeGraph {
    /// Create a new empty compute graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            generation: 0,
        }
    }

    /// Add a new node to the graph and return its index.
    pub fn add_node(&mut self, node: ComputeNode) -> NodeIndex {
        // Increment ref counts for all inputs
        for input_idx in node.operation.inputs() {
            self.add_reference(input_idx);
        }

        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.nodes[free_idx] = Some(node);
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            idx
        };

        NodeIndex(idx)
    }

    /// Get a reference to a node.
    pub fn get(&self, idx: NodeIndex) -> Option<&ComputeNode> {
        self.nodes.get(idx.0).and_then(|n| n.as_ref())
    }

    /// Get a mutable reference to a node.
    pub fn get_mut(&mut self, idx: NodeIndex) -> Option<&mut ComputeNode> {
        self.nodes.get_mut(idx.0).and_then(|n| n.as_mut())
    }

    /// Increment the reference count for a node.
    pub fn add_reference(&mut self, idx: NodeIndex) {
        if let Some(node) = self.get_mut(idx) {
            node.ref_count += 1;
        }
    }

    /// Decrement the reference count for a node.
    /// If it reaches zero, the node is garbage collected.
    pub fn remove_reference(&mut self, idx: NodeIndex) {
        let should_gc = if let Some(node) = self.get_mut(idx) {
            node.ref_count = node.ref_count.saturating_sub(1);
            node.ref_count == 0
        } else {
            false
        };

        if should_gc {
            self.garbage_collect(idx);
        }
    }

    /// Garbage collect a node and recursively decrement input ref counts.
    fn garbage_collect(&mut self, idx: NodeIndex) {
        if let Some(node) = self.nodes[idx.0].take() {
            // Recursively decrement ref counts for inputs
            for input_idx in node.operation.inputs() {
                self.remove_reference(input_idx);
            }
            // Add slot to free list
            self.free_list.push(idx.0);
            self.generation += 1;
        }
    }

    /// Get the current generation (for cache invalidation).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get the number of active nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_some()).count()
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached buffers (useful for memory pressure).
    pub fn clear_caches(&mut self) {
        for node in self.nodes.iter_mut().flatten() {
            // Don't clear Input nodes - they own their buffers
            if !matches!(node.operation, LazyOp::Input { .. }) {
                node.buffer = None;
            }
        }
        self.generation += 1;
    }

    /// Perform topological sort starting from the target node.
    /// Returns nodes in execution order (dependencies first).
    pub fn topological_sort(&self, target: NodeIndex) -> Vec<NodeIndex> {
        let mut visited = vec![false; self.nodes.len()];
        let mut result = Vec::new();

        self.topo_visit(target, &mut visited, &mut result);

        result
    }

    fn topo_visit(&self, idx: NodeIndex, visited: &mut [bool], result: &mut Vec<NodeIndex>) {
        if visited[idx.0] {
            return;
        }
        visited[idx.0] = true;

        if let Some(node) = self.get(idx) {
            // Visit all inputs first
            for input_idx in node.operation.inputs() {
                self.topo_visit(input_idx, visited, result);
            }
            // Then add this node
            result.push(idx);
        }
    }

    /// Optimize the graph by fusing operations.
    /// Currently supports:
    /// - MatMul → Add (bias) fusion: Combines matmul with bias addition into a single kernel.
    ///
    /// Returns the number of fusions performed.
    pub fn optimize(&mut self) -> usize {
        let mut fusions = 0;

        // Collect fusion candidates: (add_node_idx, matmul_node_idx, bias_node_idx)
        let mut candidates: Vec<(NodeIndex, NodeIndex, NodeIndex)> = Vec::new();

        for idx in 0..self.nodes.len() {
            let node_idx = NodeIndex(idx);
            if let Some(node) = self.get(node_idx) {
                // Look for Add nodes
                if let LazyOp::Add { lhs, rhs } = &node.operation {
                    // Check if lhs is a MatMul with no activation
                    if let Some(lhs_node) = self.get(*lhs) {
                        if let LazyOp::MatMul {
                            activation: Activation::None,
                            ..
                        } = &lhs_node.operation
                        {
                            // Check if rhs is a bias (Input node with shape [R, 1, 1, 1])
                            if let Some(rhs_node) = self.get(*rhs) {
                                if matches!(rhs_node.operation, LazyOp::Input { .. }) {
                                    let bias_shape = rhs_node.info.shape;
                                    // Bias should be 1D (broadcast across tokens and batches)
                                    if bias_shape[1] == 1 && bias_shape[2] == 1 {
                                        // Check that the matmul output matches bias dimension
                                        if lhs_node.info.shape[0] == bias_shape[0] {
                                            // Only fuse if matmul is only used by this Add
                                            // ref_count == 2 means: 1 from initial creation + 1 from Add's dependency
                                            if lhs_node.ref_count <= 2 {
                                                candidates.push((node_idx, *lhs, *rhs));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Also check the reverse: rhs is MatMul, lhs is bias
                    if let Some(rhs_node) = self.get(*rhs) {
                        if let LazyOp::MatMul {
                            activation: Activation::None,
                            ..
                        } = &rhs_node.operation
                        {
                            if let Some(lhs_node) = self.get(*lhs) {
                                if matches!(lhs_node.operation, LazyOp::Input { .. }) {
                                    let bias_shape = lhs_node.info.shape;
                                    if bias_shape[1] == 1 && bias_shape[2] == 1 {
                                        if rhs_node.info.shape[0] == bias_shape[0] {
                                            // ref_count <= 2: 1 from initial + 1 from Add's dependency
                                            if rhs_node.ref_count <= 2 {
                                                candidates.push((node_idx, *rhs, *lhs));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply fusions
        for (add_idx, matmul_idx, bias_idx) in candidates {
            if let Some(matmul_node) = self.get(matmul_idx) {
                if let LazyOp::MatMul {
                    matrix,
                    input,
                    turbo,
                    ..
                } = &matmul_node.operation
                {
                    let matrix = matrix.clone();
                    let input = *input;
                    let turbo = *turbo;
                    let output_info = matmul_node.info.clone();

                    // Replace the Add node's operation with MatMulBias
                    if let Some(add_node) = self.get_mut(add_idx) {
                        add_node.operation = LazyOp::MatMulBias {
                            matrix,
                            input,
                            bias: bias_idx,
                            activation: Activation::None,
                            turbo,
                        };
                        add_node.info = output_info;
                        fusions += 1;
                    }
                }
            }
        }

        fusions
    }

    /// Resolve a node, executing all dependencies and returning the result buffer.
    ///
    /// This is the core execution method for lazy evaluation:
    /// 1. Returns cached buffer if already computed
    /// 2. Performs topological sort to get execution order
    /// 3. Plans buffer allocations with reuse
    /// 4. Executes operations in order
    /// 5. Caches and returns the result
    pub fn resolve(
        &mut self,
        target: NodeIndex,
        context: &Context,
    ) -> Result<Arc<Buffer>, TensorError> {
        // Return cached buffer if already computed
        if let Some(node) = self.get(target) {
            if let Some(buffer) = &node.buffer {
                return Ok(buffer.clone());
            }
        } else {
            return Err(TensorErrorKind::Deduce.into());
        }

        // Apply graph optimizations before execution
        let fusions = self.optimize();
        if fusions > 0 {
            log::debug!("Graph optimization: {} fusions applied", fusions);
        }

        // Get execution order via topological sort
        let execution_order = self.topological_sort(target);

        // Plan buffer allocations with reuse optimization
        let mut buffer_plan = self.plan_buffers(&execution_order, context);

        // Execute each node in order
        for node_idx in execution_order {
            // Skip if already has a buffer (Input nodes or cached results)
            if self
                .get(node_idx)
                .map(|n| n.buffer.is_some())
                .unwrap_or(true)
            {
                continue;
            }

            // Execute this node with the buffer plan
            self.execute_node_with_plan(node_idx, context, &mut buffer_plan)?;
        }

        // Return the target's buffer
        self.get(target)
            .and_then(|n| n.buffer.clone())
            .ok_or_else(|| TensorErrorKind::Deduce.into())
    }

    /// Plan buffer allocations for the execution order.
    /// Attempts to reuse buffers when their producing nodes are no longer needed.
    fn plan_buffers(&self, execution_order: &[NodeIndex], context: &Context) -> BufferPlan {
        let mut plan = BufferPlan::new();

        // Compute last use index for each node
        let mut last_use: HashMap<usize, usize> = HashMap::new();
        for (exec_idx, &node_idx) in execution_order.iter().enumerate() {
            if let Some(node) = self.get(node_idx) {
                for input_idx in node.operation.inputs() {
                    last_use.insert(input_idx.0, exec_idx);
                }
            }
        }

        // Allocate or reuse buffers
        for (exec_idx, &node_idx) in execution_order.iter().enumerate() {
            let node = match self.get(node_idx) {
                Some(n) => n,
                None => continue,
            };

            // Skip Input nodes - they already have buffers
            if matches!(node.operation, LazyOp::Input { .. }) {
                continue;
            }

            // Skip if already has a cached buffer
            if node.buffer.is_some() {
                continue;
            }

            // Calculate required buffer size
            let size = (node.info.shape.len() * node.info.dtype.size()) as u64;
            let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

            // Try to reuse an available buffer
            let buffer = plan.try_reuse(size, usage).unwrap_or_else(|| {
                // Allocate new buffer
                context.checkout_buffer(size as usize, usage)
            });

            plan.assign(node_idx, buffer);

            // Release input buffers that are no longer needed after this operation
            for input_idx in node.operation.inputs() {
                if last_use.get(&input_idx.0) == Some(&exec_idx) {
                    // This is the last use of this input
                    if let Some(input_node) = self.get(input_idx) {
                        // Only release non-Input buffers (Input buffers are owned externally)
                        if !matches!(input_node.operation, LazyOp::Input { .. }) {
                            if let Some(buf) = plan.get(input_idx) {
                                plan.release(buf);
                            }
                        }
                    }
                }
            }
        }

        plan
    }

    /// Execute a node using the buffer plan for allocation.
    fn execute_node_with_plan(
        &mut self,
        idx: NodeIndex,
        context: &Context,
        buffer_plan: &mut BufferPlan,
    ) -> Result<(), TensorError> {
        let node = self.get(idx).ok_or(TensorErrorKind::Deduce)?;
        let output_shape = node.info.shape;

        // Get the pre-planned buffer for this node
        let planned_buffer = buffer_plan.get(idx);

        match &node.operation {
            LazyOp::Input { buffer } => {
                let buffer = buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(buffer);
                }
                Ok(())
            }

            LazyOp::MatMul {
                matrix,
                input,
                activation,
                turbo,
            } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let matrix = unsafe { matrix.get() };
                let activation = *activation;
                let turbo = *turbo;

                // Use planned buffer or allocate new one
                let output: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };
                let input_tensor = self.wrap_buffer_as_tensor(context, input_buffer, input_shape);

                // Try Metal dispatch for Q4K matrices on macOS
                #[cfg(all(target_os = "macos", feature = "metal-acceleration"))]
                if let super::matrix::Matrix::Q4K { w, .. } = matrix {
                    if let Some(output_buf) = self.execute_metal_q4k_matmul(
                        context,
                        w,
                        &input_tensor,
                        &output,
                        activation,
                    ) {
                        if let Some(node) = self.get_mut(idx) {
                            node.buffer = Some(output_buf);
                        }
                        return Ok(());
                    }
                    // Fall through to WebGPU if Metal fails
                }

                let op = matrix.matmul_op(&input_tensor, &output, activation, turbo)?;
                context.queue.submit(context.encode(&op));

                let output_buffer = output.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }

            LazyOp::Add { lhs, rhs } => {
                let lhs_buffer = self
                    .get(*lhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*lhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let rhs_buffer = self
                    .get(*rhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*rhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let lhs_shape = self.get(*lhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let rhs_shape = self.get(*rhs).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let lhs_tensor = self.wrap_buffer_as_tensor(context, lhs_buffer, lhs_shape);
                let output_tensor =
                    self.wrap_buffer_as_tensor(context, rhs_buffer.clone(), rhs_shape);

                let op = TensorOp::add(&lhs_tensor, &output_tensor)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(rhs_buffer);
                }
                Ok(())
            }

            LazyOp::Mul { lhs, rhs } => {
                let lhs_buffer = self
                    .get(*lhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*lhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let rhs_buffer = self
                    .get(*rhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*rhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let lhs_shape = self.get(*lhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let rhs_shape = self.get(*rhs).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let lhs_tensor = self.wrap_buffer_as_tensor(context, lhs_buffer, lhs_shape);
                let output_tensor =
                    self.wrap_buffer_as_tensor(context, rhs_buffer.clone(), rhs_shape);

                let op = TensorOp::mul(&lhs_tensor, &output_tensor)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(rhs_buffer);
                }
                Ok(())
            }

            LazyOp::Softmax { input } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::softmax(&tensor)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::LayerNorm {
                input,
                weight,
                bias,
                eps,
            } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let weight_buffer = self
                    .get(*weight)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*weight))
                    .ok_or(TensorErrorKind::Deduce)?;
                let bias_buffer = self
                    .get(*bias)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*bias))
                    .ok_or(TensorErrorKind::Deduce)?;

                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let weight_shape = self.get(*weight).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let bias_shape = self.get(*bias).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let eps = *eps;

                let x_tensor =
                    self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);
                let w_tensor = self.wrap_buffer_as_tensor(context, weight_buffer, weight_shape);
                let b_tensor = self.wrap_buffer_as_tensor(context, bias_buffer, bias_shape);

                let op = TensorOp::layer_norm(&w_tensor, &b_tensor, &x_tensor, eps)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::RmsNorm {
                input,
                weight,
                bias,
                eps,
            } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let weight_buffer = self
                    .get(*weight)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*weight))
                    .ok_or(TensorErrorKind::Deduce)?;
                let bias_buffer = self
                    .get(*bias)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*bias))
                    .ok_or(TensorErrorKind::Deduce)?;

                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let weight_shape = self.get(*weight).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let bias_shape = self.get(*bias).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let eps = *eps;

                let x_tensor =
                    self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);
                let w_tensor = self.wrap_buffer_as_tensor(context, weight_buffer, weight_shape);
                let b_tensor = self.wrap_buffer_as_tensor(context, bias_buffer, bias_shape);

                let op = TensorOp::rms_norm(&w_tensor, &b_tensor, &x_tensor, eps)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::Blend { lhs, rhs, factor } => {
                let lhs_buffer = self
                    .get(*lhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*lhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let rhs_buffer = self
                    .get(*rhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*rhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let factor_buffer = self
                    .get(*factor)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*factor))
                    .ok_or(TensorErrorKind::Deduce)?;

                let lhs_shape = self.get(*lhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let rhs_shape = self.get(*rhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let factor_shape = self.get(*factor).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let lhs_tensor = self.wrap_buffer_as_tensor(context, lhs_buffer, lhs_shape);
                let rhs_tensor = self.wrap_buffer_as_tensor(context, rhs_buffer, rhs_shape);
                let factor_tensor =
                    self.wrap_buffer_as_tensor(context, factor_buffer, factor_shape);

                let output_tensor: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };

                let op = TensorOp::blend_tensor(
                    &lhs_tensor,
                    &rhs_tensor,
                    &factor_tensor,
                    &output_tensor,
                )?;
                context.queue.submit(context.encode(&op));

                let output_buffer = output_tensor.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }

            LazyOp::TokenShift { input, state, mix } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let state_buffer = self
                    .get(*state)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*state))
                    .ok_or(TensorErrorKind::Deduce)?;
                let mix_buffer = self
                    .get(*mix)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*mix))
                    .ok_or(TensorErrorKind::Deduce)?;

                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let state_shape = self.get(*state).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let mix_shape = self.get(*mix).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let input_tensor =
                    self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);
                let state_tensor =
                    self.wrap_buffer_as_tensor_f32(context, state_buffer, state_shape);
                let mix_tensor = self.wrap_buffer_as_tensor(context, mix_buffer, mix_shape);

                let output_tensor: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };

                let op = TensorOp::token_shift_lazy(
                    &mix_tensor,
                    &state_tensor,
                    &input_tensor,
                    &output_tensor,
                    false,
                )?;
                context.queue.submit(context.encode(&op));

                let output_buffer = output_tensor.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }

            LazyOp::Blit { input } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let input_tensor = self.wrap_buffer_as_tensor(context, input_buffer, input_shape);
                let output_tensor: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };

                let op = TensorOp::blit(&input_tensor, &output_tensor)?;
                context.queue.submit(context.encode(&op));

                let output_buffer = output_tensor.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }

            LazyOp::Affine { input, scale, bias } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let scale = *scale;
                let bias = *bias;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::affine(&tensor, scale, bias)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::GroupNorm {
                input,
                weight,
                bias,
                eps,
            } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let weight_buffer = self
                    .get(*weight)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*weight))
                    .ok_or(TensorErrorKind::Deduce)?;
                let bias_buffer = self
                    .get(*bias)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*bias))
                    .ok_or(TensorErrorKind::Deduce)?;

                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let weight_shape = self.get(*weight).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let bias_shape = self.get(*bias).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let eps = *eps;

                let x_tensor =
                    self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);
                let w_tensor = self.wrap_buffer_as_tensor(context, weight_buffer, weight_shape);
                let b_tensor = self.wrap_buffer_as_tensor(context, bias_buffer, bias_shape);

                let op = TensorOp::group_norm(&w_tensor, &b_tensor, &x_tensor, eps)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::L2Norm { input, eps } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let eps = *eps;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::l2_norm(&tensor, eps)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::Sigmoid { input } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::activate(&tensor, Activation::Sigmoid)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::Tanh { input } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::activate(&tensor, Activation::Tanh)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::SquaredRelu { input } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let tensor = self.wrap_buffer_as_tensor(context, input_buffer.clone(), input_shape);

                let op = TensorOp::activate(&tensor, Activation::SquaredRelu)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(input_buffer);
                }
                Ok(())
            }

            LazyOp::Lerp {
                lhs,
                rhs,
                factor,
                reversed,
            } => {
                let lhs_buffer = self
                    .get(*lhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*lhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let rhs_buffer = self
                    .get(*rhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*rhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let factor_buffer = self
                    .get(*factor)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*factor))
                    .ok_or(TensorErrorKind::Deduce)?;

                let lhs_shape = self.get(*lhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let rhs_shape = self.get(*rhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let factor_shape = self.get(*factor).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let reversed = *reversed;

                let lhs_tensor = self.wrap_buffer_as_tensor(context, lhs_buffer, lhs_shape);
                let rhs_tensor = self.wrap_buffer_as_tensor(context, rhs_buffer.clone(), rhs_shape);
                let factor_tensor =
                    self.wrap_buffer_as_tensor(context, factor_buffer, factor_shape);

                let op = TensorOp::lerp(&lhs_tensor, &rhs_tensor, &factor_tensor, reversed)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(rhs_buffer);
                }
                Ok(())
            }

            LazyOp::AddActivate {
                lhs,
                rhs,
                lhs_act,
                rhs_act,
                out_act,
            } => {
                let lhs_buffer = self
                    .get(*lhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*lhs))
                    .ok_or(TensorErrorKind::Deduce)?;
                let rhs_buffer = self
                    .get(*rhs)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*rhs))
                    .ok_or(TensorErrorKind::Deduce)?;

                let lhs_shape = self.get(*lhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let rhs_shape = self.get(*rhs).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let lhs_act = *lhs_act;
                let rhs_act = *rhs_act;
                let out_act = *out_act;

                let lhs_tensor = self.wrap_buffer_as_tensor(context, lhs_buffer, lhs_shape);
                let rhs_tensor = self.wrap_buffer_as_tensor(context, rhs_buffer.clone(), rhs_shape);

                let op =
                    TensorOp::add_activate(&lhs_tensor, &rhs_tensor, lhs_act, rhs_act, out_act)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(rhs_buffer);
                }
                Ok(())
            }

            LazyOp::ControlKV7 { p, a, k } => {
                let p_buffer = self
                    .get(*p)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*p))
                    .ok_or(TensorErrorKind::Deduce)?;
                let a_buffer = self
                    .get(*a)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*a))
                    .ok_or(TensorErrorKind::Deduce)?;
                let k_buffer = self
                    .get(*k)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*k))
                    .ok_or(TensorErrorKind::Deduce)?;

                let p_shape = self.get(*p).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let a_shape = self.get(*a).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let k_shape = self.get(*k).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let p_tensor = self.wrap_buffer_as_tensor(context, p_buffer, p_shape);
                let a_tensor = self.wrap_buffer_as_tensor(context, a_buffer, a_shape);
                let k_tensor = self.wrap_buffer_as_tensor(context, k_buffer.clone(), k_shape);

                let op = TensorOp::control_k_v7(&p_tensor, &a_tensor, &k_tensor)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(k_buffer);
                }
                Ok(())
            }

            LazyOp::PackKvakk { k, v, a, kk } => {
                let k_buffer = self
                    .get(*k)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*k))
                    .ok_or(TensorErrorKind::Deduce)?;
                let v_buffer = self
                    .get(*v)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*v))
                    .ok_or(TensorErrorKind::Deduce)?;
                let a_buffer = self
                    .get(*a)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*a))
                    .ok_or(TensorErrorKind::Deduce)?;
                let kk_buffer = self
                    .get(*kk)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*kk))
                    .ok_or(TensorErrorKind::Deduce)?;

                let k_shape = self.get(*k).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let v_shape = self.get(*v).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let a_shape = self.get(*a).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let kk_shape = self.get(*kk).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let k_tensor = self.wrap_buffer_as_tensor(context, k_buffer, k_shape);
                let v_tensor = self.wrap_buffer_as_tensor(context, v_buffer, v_shape);
                let a_tensor = self.wrap_buffer_as_tensor(context, a_buffer, a_shape);
                let kk_tensor = self.wrap_buffer_as_tensor(context, kk_buffer, kk_shape);

                let n_tensor: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };

                let op =
                    TensorOp::pack_kvakk(&k_tensor, &v_tensor, &a_tensor, &kk_tensor, &n_tensor)?;
                context.queue.submit(context.encode(&op));

                let output_buffer = n_tensor.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }

            LazyOp::TimeMixV7 {
                r,
                w,
                n,
                x,
                state,
                batch_index,
                num_tokens,
            } => {
                let r_buffer = self
                    .get(*r)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*r))
                    .ok_or(TensorErrorKind::Deduce)?;
                let w_buffer = self
                    .get(*w)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*w))
                    .ok_or(TensorErrorKind::Deduce)?;
                let n_buffer = self
                    .get(*n)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*n))
                    .ok_or(TensorErrorKind::Deduce)?;
                let x_buffer = self
                    .get(*x)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*x))
                    .ok_or(TensorErrorKind::Deduce)?;
                let state_buffer = self
                    .get(*state)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*state))
                    .ok_or(TensorErrorKind::Deduce)?;

                let r_shape = self.get(*r).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let w_shape = self.get(*w).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let n_shape = self.get(*n).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let x_shape = self.get(*x).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let state_shape = self.get(*state).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let batch_index = *batch_index;
                let num_tokens = *num_tokens;

                let r_tensor = self.wrap_buffer_as_tensor(context, r_buffer, r_shape);
                let w_tensor = self.wrap_buffer_as_tensor(context, w_buffer, w_shape);
                let n_tensor = self.wrap_buffer_as_tensor(context, n_buffer, n_shape);
                let x_tensor = self.wrap_buffer_as_tensor(context, x_buffer.clone(), x_shape);
                let state_tensor =
                    self.wrap_buffer_as_tensor_f32(context, state_buffer, state_shape);

                let op = TensorOp::time_mix_v7_lazy(
                    &state_tensor,
                    &r_tensor,
                    &w_tensor,
                    &n_tensor,
                    &x_tensor,
                    batch_index,
                    num_tokens,
                )?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(x_buffer);
                }
                Ok(())
            }

            LazyOp::ChannelMixV7 {
                v,
                x,
                state,
                batch_index,
                num_tokens,
            } => {
                let v_buffer = self
                    .get(*v)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*v))
                    .ok_or(TensorErrorKind::Deduce)?;
                let x_buffer = self
                    .get(*x)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*x))
                    .ok_or(TensorErrorKind::Deduce)?;
                let state_buffer = self
                    .get(*state)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*state))
                    .ok_or(TensorErrorKind::Deduce)?;

                let v_shape = self.get(*v).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let x_shape = self.get(*x).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let state_shape = self.get(*state).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let batch_index = *batch_index;
                let num_tokens = *num_tokens;

                let v_tensor = self.wrap_buffer_as_tensor(context, v_buffer, v_shape);
                let x_tensor = self.wrap_buffer_as_tensor(context, x_buffer.clone(), x_shape);
                let state_tensor =
                    self.wrap_buffer_as_tensor_f32(context, state_buffer, state_shape);

                let op = TensorOp::channel_mix_v7_lazy(
                    &state_tensor,
                    &v_tensor,
                    &x_tensor,
                    batch_index,
                    num_tokens,
                )?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(x_buffer);
                }
                Ok(())
            }

            LazyOp::TimeFirstV7 { u, r, n, x } => {
                let u_buffer = self
                    .get(*u)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*u))
                    .ok_or(TensorErrorKind::Deduce)?;
                let r_buffer = self
                    .get(*r)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*r))
                    .ok_or(TensorErrorKind::Deduce)?;
                let n_buffer = self
                    .get(*n)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*n))
                    .ok_or(TensorErrorKind::Deduce)?;
                let x_buffer = self
                    .get(*x)
                    .and_then(|node| node.buffer.clone())
                    .or_else(|| buffer_plan.get(*x))
                    .ok_or(TensorErrorKind::Deduce)?;

                let u_shape = self.get(*u).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let r_shape = self.get(*r).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let n_shape = self.get(*n).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let x_shape = self.get(*x).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let _u_tensor = self.wrap_buffer_as_tensor(context, u_buffer, u_shape);
                let r_tensor = self.wrap_buffer_as_tensor(context, r_buffer, r_shape);
                let n_tensor = self.wrap_buffer_as_tensor(context, n_buffer, n_shape);
                let x_tensor = self.wrap_buffer_as_tensor(context, x_buffer.clone(), x_shape);

                let op = TensorOp::time_first_v7_lazy(&r_tensor, &n_tensor, &x_tensor)?;
                context.queue.submit(context.encode(&op));

                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(x_buffer);
                }
                Ok(())
            }

            LazyOp::MatMulBias {
                matrix,
                input,
                bias,
                activation,
                turbo,
            } => {
                let input_buffer = self
                    .get(*input)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*input))
                    .ok_or(TensorErrorKind::Deduce)?;
                let bias_buffer = self
                    .get(*bias)
                    .and_then(|n| n.buffer.clone())
                    .or_else(|| buffer_plan.get(*bias))
                    .ok_or(TensorErrorKind::Deduce)?;
                let input_shape = self.get(*input).ok_or(TensorErrorKind::Deduce)?.info.shape;
                let bias_shape = self.get(*bias).ok_or(TensorErrorKind::Deduce)?.info.shape;

                let matrix = unsafe { matrix.get() };
                let activation = *activation;
                let turbo = *turbo;

                let output: TensorGpu<f16, ReadWrite> = match planned_buffer {
                    Some(buf) => self.wrap_buffer_as_tensor(context, buf, output_shape),
                    None => TensorInitContext::init(context, output_shape),
                };
                let input_tensor = self.wrap_buffer_as_tensor(context, input_buffer, input_shape);
                let bias_tensor = self.wrap_buffer_as_tensor(context, bias_buffer, bias_shape);

                // Try Metal dispatch for supported matrix types on macOS
                #[cfg(all(target_os = "macos", feature = "metal-acceleration"))]
                {
                    // Metal Q4K path - use native SIMD operations
                    if let super::matrix::Matrix::Q4K { w, .. } = matrix {
                        if let Some(output_buf) = self.execute_metal_q4k_matmul_bias(
                            context,
                            w,
                            &input_tensor,
                            &bias_tensor,
                            &output,
                            activation,
                        ) {
                            if let Some(node) = self.get_mut(idx) {
                                node.buffer = Some(output_buf);
                            }
                            return Ok(());
                        }
                        // Fall through to WebGPU if Metal fails
                    }

                    // Metal Int8 path
                    if let super::matrix::Matrix::Int8 { w, m } = matrix {
                        if let Some(output_buf) = self.execute_metal_int8_matmul_bias(
                            context,
                            w,
                            m,
                            &input_tensor,
                            &bias_tensor,
                            &output,
                            activation,
                        ) {
                            if let Some(node) = self.get_mut(idx) {
                                node.buffer = Some(output_buf);
                            }
                            return Ok(());
                        }
                        // Fall through to WebGPU if Metal fails
                    }
                }

                let op = TensorOp::matmul_vec_fp16_bias(
                    matrix,
                    &input_tensor,
                    &bias_tensor,
                    &output,
                    activation,
                    turbo,
                )?;
                context.queue.submit(context.encode(&op));

                let output_buffer = output.buffer.clone();
                if let Some(node) = self.get_mut(idx) {
                    node.buffer = Some(output_buffer);
                }
                Ok(())
            }
        }
    }

    /// Helper to wrap an existing buffer as a TensorGpu<f16> for use with TensorOp.
    fn wrap_buffer_as_tensor(
        &self,
        context: &Context,
        buffer: Arc<Buffer>,
        shape: Shape,
    ) -> TensorGpu<f16, ReadWrite> {
        let meta = context.checkout_shape_uniform(shape);
        TensorGpu {
            shape,
            data: super::TensorGpuData {
                context: context.clone(),
                meta,
                buffer,
            },
            id: uid::Id::new(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Helper to wrap an existing buffer as a TensorGpu<f32> for use with TensorOp.
    fn wrap_buffer_as_tensor_f32(
        &self,
        context: &Context,
        buffer: Arc<Buffer>,
        shape: Shape,
    ) -> TensorGpu<f32, ReadWrite> {
        let meta = context.checkout_shape_uniform(shape);
        TensorGpu {
            shape,
            data: super::TensorGpuData {
                context: context.clone(),
                meta,
                buffer,
            },
            id: uid::Id::new(),
            phantom: std::marker::PhantomData,
        }
    }

    /// Execute Int8 matmul+bias using native Metal SIMD operations.
    /// Returns the output buffer if successful, None if Metal execution failed.
    #[cfg(all(target_os = "macos", feature = "metal-acceleration"))]
    #[allow(clippy::too_many_arguments)]
    fn execute_metal_int8_matmul_bias(
        &self,
        _context: &Context,
        matrix: &TensorGpu<u8, ReadWrite>,
        scales: &TensorGpu<f16, ReadWrite>,
        input: &TensorGpu<f16, ReadWrite>,
        bias: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<f16, ReadWrite>,
        activation: Activation,
    ) -> Option<Arc<Buffer>> {
        use crate::metal::kernels::{kernel_name_for_activation, MATMUL_INT8_BIAS_SOURCE};
        use crate::metal::{BufferBridge, MetalContext};

        // Extract Metal buffers from wgpu buffers (zero-copy)
        let metal_matrix = unsafe { BufferBridge::extract_from_arc(&matrix.buffer)? };
        let metal_scales = unsafe { BufferBridge::extract_from_arc(&scales.buffer)? };
        let metal_input = unsafe { BufferBridge::extract_from_arc(&input.buffer)? };
        let metal_bias = unsafe { BufferBridge::extract_from_arc(&bias.buffer)? };
        let metal_output = unsafe { BufferBridge::extract_from_arc(&output.buffer)? };

        // Create Metal context and compile shader
        let mut metal_ctx = MetalContext::new().ok()?;
        metal_ctx.compile_library(MATMUL_INT8_BIAS_SOURCE).ok()?;

        // Get the appropriate kernel for the activation
        let kernel_name = kernel_name_for_activation(activation);
        let pipeline = metal_ctx.get_pipeline(kernel_name).ok()?.clone();

        // Create params buffer
        let [k, m, _, _] = matrix.shape.into();
        let [_, t, b, _] = input.shape.into();
        let params: [u32; 4] = [k as u32, m as u32, t as u32, b as u32];
        let metal_params =
            metal_ctx.new_buffer_with_data(&params, metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let cmd = metal_ctx.queue().new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&metal_matrix), 0);
        encoder.set_buffer(1, Some(&metal_scales), 0);
        encoder.set_buffer(2, Some(&metal_input), 0);
        encoder.set_buffer(3, Some(&metal_bias), 0);
        encoder.set_buffer(4, Some(&metal_output), 0);
        encoder.set_buffer(5, Some(&metal_params), 0);

        // Dispatch threads - one thread per output element
        let thread_group_size = metal::MTLSize::new(64, 1, 1);
        let grid_size = metal::MTLSize::new(m as u64, t as u64, b as u64);
        encoder.dispatch_threads(grid_size, thread_group_size);

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        log::debug!(
            "Metal Int8 matmul+bias executed: {}x{} @ {} tokens",
            k,
            m,
            t
        );

        // Return the output buffer
        Some(output.buffer.clone())
    }

    /// Execute Q4K matmul+bias using native Metal operations.
    /// Returns the output buffer if successful, None if Metal execution failed.
    #[cfg(all(target_os = "macos", feature = "metal-acceleration"))]
    #[allow(clippy::too_many_arguments)]
    fn execute_metal_q4k_matmul_bias(
        &self,
        _context: &Context,
        matrix: &TensorGpu<u8, ReadWrite>,
        input: &TensorGpu<f16, ReadWrite>,
        bias: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<f16, ReadWrite>,
        activation: Activation,
    ) -> Option<Arc<Buffer>> {
        use crate::metal::kernels::{kernel_name_q4k_for_activation, MATMUL_Q4K_BIAS_SOURCE};
        use crate::metal::{BufferBridge, MetalContext};

        // Extract Metal buffers from wgpu buffers (zero-copy)
        let metal_matrix = unsafe { BufferBridge::extract_from_arc(&matrix.buffer)? };
        let metal_input = unsafe { BufferBridge::extract_from_arc(&input.buffer)? };
        let metal_bias = unsafe { BufferBridge::extract_from_arc(&bias.buffer)? };
        let metal_output = unsafe { BufferBridge::extract_from_arc(&output.buffer)? };

        // Create Metal context and compile shader
        let mut metal_ctx = MetalContext::new().ok()?;
        metal_ctx.compile_library(MATMUL_Q4K_BIAS_SOURCE).ok()?;

        // Get the appropriate kernel for the activation
        let kernel_name = kernel_name_q4k_for_activation(activation);
        let pipeline = metal_ctx.get_pipeline(kernel_name).ok()?.clone();

        // Q4K matrix: raw bytes, 144 bytes per 256 elements (36 u32 per block)
        // Get K from input shape (input is [K, T, B, 1])
        // Get M from output shape (output is [M, T, B, 1])
        let [k, t, b, _] = input.shape.into();
        let [m, _, _, _] = output.shape.into();

        let params: [u32; 4] = [k as u32, m as u32, t as u32, b as u32];
        let metal_params =
            metal_ctx.new_buffer_with_data(&params, metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let cmd = metal_ctx.queue().new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&metal_matrix), 0);
        encoder.set_buffer(1, Some(&metal_input), 0);
        encoder.set_buffer(2, Some(&metal_bias), 0);
        encoder.set_buffer(3, Some(&metal_output), 0);
        encoder.set_buffer(4, Some(&metal_params), 0);

        // Allocate threadgroup memory for reduction (64 threads * 4 floats)
        encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

        // Dispatch: each thread group handles 4 output rows
        let num_groups_m = (m + 3) / 4;
        let thread_group_size = metal::MTLSize::new(64, 1, 1);
        let grid_size = metal::MTLSize::new(num_groups_m as u64, t as u64, b as u64);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        log::debug!("Metal Q4K matmul+bias executed: {}x{} @ {} tokens", k, m, t);

        // Return the output buffer
        Some(output.buffer.clone())
    }

    /// Execute Q4K matmul (without bias) using native Metal operations.
    /// Returns the output buffer if successful, None if Metal execution failed.
    #[cfg(all(target_os = "macos", feature = "metal-acceleration"))]
    fn execute_metal_q4k_matmul(
        &self,
        _context: &Context,
        matrix: &TensorGpu<u8, ReadWrite>,
        input: &TensorGpu<f16, ReadWrite>,
        output: &TensorGpu<f16, ReadWrite>,
        activation: Activation,
    ) -> Option<Arc<Buffer>> {
        use crate::metal::kernels::{kernel_name_q4k_for_activation, MATMUL_Q4K_BIAS_SOURCE};
        use crate::metal::{BufferBridge, MetalContext};

        // Extract Metal buffers from wgpu buffers (zero-copy)
        let metal_matrix = unsafe { BufferBridge::extract_from_arc(&matrix.buffer)? };
        let metal_input = unsafe { BufferBridge::extract_from_arc(&input.buffer)? };
        let metal_output = unsafe { BufferBridge::extract_from_arc(&output.buffer)? };

        // Create Metal context and compile shader
        let mut metal_ctx = MetalContext::new().ok()?;
        metal_ctx.compile_library(MATMUL_Q4K_BIAS_SOURCE).ok()?;

        // Get the appropriate kernel for the activation
        let kernel_name = kernel_name_q4k_for_activation(activation);
        let pipeline = metal_ctx.get_pipeline(kernel_name).ok()?.clone();

        // Get dimensions from tensor shapes
        use super::TensorShape;
        let [k, t, b, _] = input.shape().into();
        let [m, _, _, _] = output.shape().into();

        let params: [u32; 4] = [k as u32, m as u32, t as u32, b as u32];
        let metal_params =
            metal_ctx.new_buffer_with_data(&params, metal::MTLResourceOptions::StorageModeShared);

        // Create a zero bias buffer (no bias for this op)
        let zero_bias: Vec<f16> = vec![f16::from_f32(0.0); m];
        let metal_bias = metal_ctx
            .new_buffer_with_data(&zero_bias, metal::MTLResourceOptions::StorageModeShared);

        // Create command buffer and encoder
        let cmd = metal_ctx.queue().new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&metal_matrix), 0);
        encoder.set_buffer(1, Some(&metal_input), 0);
        encoder.set_buffer(2, Some(&metal_bias), 0);
        encoder.set_buffer(3, Some(&metal_output), 0);
        encoder.set_buffer(4, Some(&metal_params), 0);

        // Allocate threadgroup memory for reduction (64 threads * 4 floats)
        encoder.set_threadgroup_memory_length(0, 64 * 4 * 4);

        // Dispatch: each thread group handles 4 output rows
        let num_groups_m = (m + 3) / 4;
        let thread_group_size = metal::MTLSize::new(64, 1, 1);
        let grid_size = metal::MTLSize::new(num_groups_m as u64, t as u64, b as u64);
        encoder.dispatch_thread_groups(grid_size, thread_group_size);

        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        log::debug!("Metal Q4K matmul executed: {}x{} @ {} tokens", k, m, t);

        // Return the output buffer
        Some(output.buffer.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_node(shape: Shape) -> ComputeNode {
        ComputeNode {
            operation: LazyOp::Softmax {
                input: NodeIndex(0), // dummy, won't be used in these tests
            },
            info: TensorInfo {
                shape,
                dtype: DataType::F32,
            },
            ref_count: 1,
            buffer: None,
        }
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = ComputeGraph::new();

        let node = make_test_node(Shape::new(64, 1, 1, 1));
        let idx = graph.add_node(node);
        assert_eq!(idx.0, 0);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_graph_reference_counting() {
        let mut graph = ComputeGraph::new();

        // Create a node (using Softmax which has a dummy input)
        let node = make_test_node(Shape::new(64, 1, 1, 1));
        let node_idx = graph.add_node(node);

        // Initial ref_count is 1, but add_node increments input refs
        // For Softmax with input NodeIndex(0), it tries to add_reference to node 0
        // which is itself, so ref_count becomes 2
        let initial_ref = graph.get(node_idx).unwrap().ref_count;

        // Add a reference
        graph.add_reference(node_idx);
        assert_eq!(graph.get(node_idx).unwrap().ref_count, initial_ref + 1);

        // Remove references
        graph.remove_reference(node_idx);
        assert_eq!(graph.get(node_idx).unwrap().ref_count, initial_ref);
    }

    #[test]
    fn test_graph_topological_sort() {
        let mut graph = ComputeGraph::new();

        // Create a chain: node0 -> node1 -> node2
        // node0 uses Add with lhs=rhs=itself (self-referential, but fine for testing structure)
        let idx0 = {
            let node0 = ComputeNode {
                operation: LazyOp::Add {
                    lhs: NodeIndex(0),
                    rhs: NodeIndex(0),
                },
                info: TensorInfo {
                    shape: Shape::new(64, 1, 1, 1),
                    dtype: DataType::F32,
                },
                ref_count: 1,
                buffer: None,
            };
            graph.add_node(node0)
        };

        let idx1 = {
            let node1 = ComputeNode {
                operation: LazyOp::Softmax { input: idx0 },
                info: TensorInfo {
                    shape: Shape::new(64, 1, 1, 1),
                    dtype: DataType::F32,
                },
                ref_count: 1,
                buffer: None,
            };
            graph.add_node(node1)
        };

        let idx2 = {
            let node2 = ComputeNode {
                operation: LazyOp::Softmax { input: idx1 },
                info: TensorInfo {
                    shape: Shape::new(64, 1, 1, 1),
                    dtype: DataType::F32,
                },
                ref_count: 1,
                buffer: None,
            };
            graph.add_node(node2)
        };

        // Topological sort from node2 should give [node0, node1, node2]
        let order = graph.topological_sort(idx2);
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], idx0);
        assert_eq!(order[1], idx1);
        assert_eq!(order[2], idx2);
    }

    #[test]
    fn test_buffer_plan_reuse() {
        let mut plan = BufferPlan::new();

        // Create a mock buffer key
        let size = 1024u64;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        // Initially no buffers available for reuse
        assert!(plan.try_reuse(size, usage).is_none());

        // Simulate releasing a buffer (we can't create real buffers without GPU context)
        // Just test the HashMap logic
        assert!(plan.available.is_empty());

        // Test assignment
        // Note: We can't test with real buffers without GPU context,
        // but we can verify the structure works
        assert!(plan.get(NodeIndex(0)).is_none());
    }

    #[test]
    fn test_lazy_op_inputs() {
        // Test that LazyOp::inputs() returns correct dependencies for non-Input ops
        let add_op = LazyOp::Add {
            lhs: NodeIndex(0),
            rhs: NodeIndex(1),
        };
        let inputs = add_op.inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0], NodeIndex(0));
        assert_eq!(inputs[1], NodeIndex(1));

        let softmax_op = LazyOp::Softmax {
            input: NodeIndex(3),
        };
        let inputs = softmax_op.inputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0], NodeIndex(3));

        let blend_op = LazyOp::Blend {
            lhs: NodeIndex(0),
            rhs: NodeIndex(1),
            factor: NodeIndex(2),
        };
        let inputs = blend_op.inputs();
        assert_eq!(inputs.len(), 3);

        let layer_norm_op = LazyOp::LayerNorm {
            input: NodeIndex(0),
            weight: NodeIndex(1),
            bias: NodeIndex(2),
            eps: 1e-5,
        };
        let inputs = layer_norm_op.inputs();
        assert_eq!(inputs.len(), 3);

        let token_shift_op = LazyOp::TokenShift {
            input: NodeIndex(0),
            state: NodeIndex(1),
            mix: NodeIndex(2),
        };
        let inputs = token_shift_op.inputs();
        assert_eq!(inputs.len(), 3);
    }
}

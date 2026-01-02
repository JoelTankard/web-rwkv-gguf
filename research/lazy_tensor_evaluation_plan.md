# Lazy Tensor Evaluation Implementation Plan

> Based on fusor-ml/core patterns adapted for web-rwkv-gguf

## Executive Summary

**Goal:** Defer GPU buffer allocation and computation until results are actually needed, enabling:

1. Better memory planning (know full graph before allocating)
2. Automatic intermediate buffer reuse
3. Potential for operation fusion
4. Reduced memory pressure during inference

**Current State:** We already have `LazyMatrix` for deferring weight uploads. This plan extends that concept to _all_ tensor operations.

---

## How Fusor-ML Achieves This

### Core Concept: Compute Graph with Reference Counting

```rust
// Tensors don't hold GPU buffers - they hold a NodeIndex into a graph
pub(crate) struct LazyTensorData {
    device: Device,
    info: TensorInfo,      // Shape, dtype, etc.
    key: NodeIndex,        // Reference into compute graph
}

// The compute graph stores operations as nodes
pub struct ComputeGraph {
    nodes: Vec<ComputeNode>,
    // Each node has: operation, inputs (NodeIndex), output info, ref_count
}

// Clone/Drop manage reference counts
impl Clone for LazyTensorData {
    fn clone(&self) -> Self {
        self.device.compute_graph().add_reference(self.key);
        // ...
    }
}

impl Drop for LazyTensorData {
    fn drop(&mut self) {
        self.device.compute_graph().remove_reference(self.key);
        // When ref_count hits 0, node can be garbage collected
    }
}
```

### Materialization

```rust
impl Tensor {
    // Actually execute the graph and get GPU buffer
    pub fn materialize(&self) -> GpuBuffer {
        self.device.compute_graph().resolve(self.key)
    }
}

impl ComputeGraph {
    fn resolve(&mut self, target: NodeIndex) -> GpuBuffer {
        // 1. Topological sort from target back to inputs
        // 2. Allocate buffers (reusing where ref_count allows)
        // 3. Execute operations in order
        // 4. Return target buffer
    }
}
```

---

## Implementation Plan for web-rwkv-gguf

### Phase 1: Core Infrastructure (Foundation) - ✅ Done

#### 1.1 Define the Compute Graph

```rust
// src/tensor/graph.rs (new file)

use petgraph::graph::{DiGraph, NodeIndex};
use std::sync::{Arc, RwLock};

/// A node in the compute graph
#[derive(Debug)]
pub struct ComputeNode {
    /// The operation that produces this node's output
    pub operation: LazyOp,
    /// Shape/dtype of the output
    pub info: TensorInfo,
    /// Reference count (how many LazyTensors point here)
    pub ref_count: usize,
    /// Cached GPU buffer (populated after materialization)
    pub buffer: Option<Arc<Buffer>>,
}

/// Operations that can be lazily evaluated
#[derive(Debug, Clone)]
pub enum LazyOp {
    /// Input tensor (already has data)
    Input { buffer: Arc<Buffer> },
    /// Constant value
    Constant { data: Vec<u8>, shape: Shape },
    /// Matrix multiplication
    MatMul {
        matrix: MatrixRef,  // Reference to weight matrix
        input: NodeIndex,
        activation: Activation,
    },
    /// Element-wise operations
    Add { lhs: NodeIndex, rhs: NodeIndex },
    Mul { lhs: NodeIndex, rhs: NodeIndex },
    // ... other ops
}

/// The compute graph itself
pub struct ComputeGraph {
    nodes: Vec<ComputeNode>,
    /// Free list for reusing node slots
    free_list: Vec<NodeIndex>,
}
```

#### 1.2 Define LazyTensor

```rust
// src/tensor/lazy_tensor.rs (new file)

/// A tensor that defers computation until materialized
#[derive(Debug)]
pub struct LazyTensor<T: Scalar> {
    context: Context,
    info: TensorInfo,
    key: NodeIndex,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> Clone for LazyTensor<T> {
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

impl<T: Scalar> Drop for LazyTensor<T> {
    fn drop(&mut self) {
        self.context.graph().remove_reference(self.key);
    }
}

impl<T: Scalar> LazyTensor<T> {
    /// Execute the graph and return the GPU buffer
    pub fn materialize(&self) -> Result<TensorGpu<T, ReadWrite>, TensorError> {
        self.context.graph().resolve(self.key)
    }

    /// Get shape without materializing
    pub fn shape(&self) -> Shape {
        self.info.shape
    }
}
```

#### 1.3 Integrate Graph into Context

```rust
// Modify src/context.rs

pub struct Context {
    // ... existing fields ...

    /// Compute graph for lazy evaluation
    graph: Arc<RwLock<ComputeGraph>>,
}

impl Context {
    pub fn graph(&self) -> impl DerefMut<Target = ComputeGraph> {
        self.graph.write().unwrap()
    }
}
```

### Phase 2: Lazy Operations - ✅ Done

#### 2.1 Lazy Operation Builders

```rust
// src/tensor/lazy_ops.rs (new file)

impl<T: Scalar> LazyTensor<T> {
    /// Lazy matrix multiplication
    pub fn matmul(
        &self,
        matrix: &Matrix,
        activation: Activation,
    ) -> LazyTensor<T> {
        let output_shape = compute_matmul_shape(self.shape(), matrix.shape());
        let node = ComputeNode {
            operation: LazyOp::MatMul {
                matrix: matrix.as_ref(),
                input: self.key,
                activation,
            },
            info: TensorInfo { shape: output_shape, dtype: T::dtype() },
            ref_count: 1,
            buffer: None,
        };
        let key = self.context.graph().add_node(node);
        LazyTensor {
            context: self.context.clone(),
            info: node.info,
            key,
            _phantom: PhantomData,
        }
    }

    /// Lazy addition
    pub fn add(&self, other: &LazyTensor<T>) -> LazyTensor<T> {
        // Similar pattern...
    }

    // ... other lazy ops
}
```

### Phase 3: Graph Resolution (Execution) - ✅ Done

#### 3.1 Topological Sort and Execution

```rust
// src/tensor/graph.rs

impl ComputeGraph {
    /// Resolve a node, executing all dependencies first
    pub fn resolve(&mut self, target: NodeIndex, context: &Context)
        -> Result<Arc<Buffer>, TensorError>
    {
        // Return cached buffer if already computed
        if let Some(buffer) = &self.nodes[target].buffer {
            return Ok(buffer.clone());
        }

        // Topological sort to get execution order
        let execution_order = self.topological_sort(target);

        // Buffer allocation planning
        let buffer_plan = self.plan_buffers(&execution_order);

        // Execute operations
        let mut encoder = context.device.create_command_encoder(&Default::default());
        for node_idx in execution_order {
            self.execute_node(node_idx, &buffer_plan, &mut encoder, context)?;
        }
        context.queue.submit(Some(encoder.finish()));

        Ok(self.nodes[target].buffer.clone().unwrap())
    }

    fn execute_node(
        &mut self,
        idx: NodeIndex,
        buffer_plan: &BufferPlan,
        encoder: &mut CommandEncoder,
        context: &Context,
    ) -> Result<(), TensorError> {
        let node = &self.nodes[idx];
        match &node.operation {
            LazyOp::MatMul { matrix, input, activation } => {
                let input_buffer = self.nodes[*input].buffer.as_ref().unwrap();
                let output_buffer = buffer_plan.get_buffer(idx);

                // Create and dispatch the matmul operation
                let op = matrix.matmul_op(input_buffer, output_buffer, *activation, turbo)?;
                context.encode_to_encoder(&op, encoder);

                self.nodes[idx].buffer = Some(output_buffer);
            }
            // ... handle other ops
        }
        Ok(())
    }
}
```

#### 3.2 Buffer Reuse Planning

```rust
/// Plan buffer allocations, reusing buffers where possible
fn plan_buffers(&self, execution_order: &[NodeIndex]) -> BufferPlan {
    let mut plan = BufferPlan::new();
    let mut available_buffers: HashMap<(usize, BufferUsages), Vec<Arc<Buffer>>> = HashMap::new();

    for &idx in execution_order {
        let node = &self.nodes[idx];
        let size = node.info.shape.size() * node.info.dtype.size();
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        // Try to reuse an available buffer
        let buffer = available_buffers
            .get_mut(&(size, usage))
            .and_then(|v| v.pop())
            .unwrap_or_else(|| self.allocate_buffer(size, usage));

        plan.assign(idx, buffer.clone());

        // Mark input buffers as available if ref_count allows
        for input_idx in self.get_inputs(idx) {
            let input_node = &self.nodes[input_idx];
            if input_node.ref_count == 1 {
                // This is the last use, buffer can be reused
                if let Some(buf) = &input_node.buffer {
                    let key = (buf.size() as usize, buf.usage());
                    available_buffers.entry(key).or_default().push(buf.clone());
                }
            }
        }
    }

    plan
}
```

### Phase 4: Migration Strategy

#### 4.1 Parallel API (Non-Breaking)

Keep existing `TensorGpu` and `TensorOp` working. Add `LazyTensor` as an alternative:

```rust
// Users can choose:

// Option A: Eager (current behavior)
let output = TensorOp::matmul(&matrix, &input, activation)?;
context.queue.submit(context.encode(&output));

// Option B: Lazy (new)
let lazy_input = LazyTensor::from_gpu(&input);
let lazy_output = lazy_input.matmul(&matrix, activation);
let output = lazy_output.materialize()?;  // Executes here
```

#### 4.2 Gradual Migration

1. **Phase 4a:** Implement lazy versions of core ops (matmul, add, mul, activation)
2. **Phase 4b:** Add lazy inference path in runtime (v7.rs, etc.)
3. **Phase 4c:** Benchmark and optimize
4. **Phase 4d:** Make lazy the default, deprecate eager path

### Phase 5: Advanced Optimizations

#### 5.1 Operation Fusion

Once we have the graph, we can fuse operations:

```rust
fn optimize_graph(&mut self) {
    // Pattern: MatMul -> Add (bias) -> Activation
    // Fuse into: MatMul with fused bias and activation

    for idx in self.nodes.indices() {
        if let LazyOp::Activation { input, act } = &self.nodes[idx].operation {
            if let LazyOp::Add { lhs, rhs } = &self.nodes[*input].operation {
                if let LazyOp::MatMul { .. } = &self.nodes[*lhs].operation {
                    // Fuse!
                    self.fuse_matmul_bias_activation(idx, *input, *lhs, *rhs, *act);
                }
            }
        }
    }
}
```

#### 5.2 Memory Planning

With the full graph known upfront:

```rust
fn optimize_memory(&mut self, execution_order: &[NodeIndex]) {
    // Compute liveness intervals for each buffer
    let liveness = self.compute_liveness(execution_order);

    // Use graph coloring to minimize total memory
    let allocation = graph_color_allocation(&liveness);

    // Pre-allocate a memory pool
    let pool_size = allocation.total_size();
    let pool = self.allocate_pool(pool_size);

    // Assign offsets within pool
    for (idx, offset) in allocation.offsets() {
        self.nodes[idx].buffer_offset = offset;
    }
}
```

---

## File Structure

```text
src/tensor/
├── mod.rs              # Add: pub mod graph; pub mod lazy_tensor; pub mod lazy_ops;
├── graph.rs            # NEW: ComputeGraph, ComputeNode, LazyOp
├── lazy_tensor.rs      # NEW: LazyTensor<T>
├── lazy_ops.rs         # NEW: Lazy operation implementations
├── lazy.rs             # EXISTING: LazyMatrix (keep for now, migrate later)
├── ops.rs              # EXISTING: TensorOp (keep for eager path)
└── ...
```

---

## Implementation Order

| Step | Description                                                    | Effort | Dependencies |
| ---- | -------------------------------------------------------------- | ------ | ------------ |
| 1    | Create `graph.rs` with `ComputeGraph`, `ComputeNode`, `LazyOp` | Medium | None         |
| 2    | Create `lazy_tensor.rs` with `LazyTensor<T>`                   | Medium | Step 1       |
| 3    | Integrate `ComputeGraph` into `Context`                        | Low    | Step 1       |
| 4    | Implement `resolve()` with topological sort                    | Medium | Steps 1-3    |
| 5    | Implement lazy `matmul` operation                              | Medium | Step 4       |
| 6    | Implement lazy `add`, `mul`, `activation`                      | Medium | Step 4       |
| 7    | Add buffer reuse in `plan_buffers()`                           | Medium | Step 4       |
| 8    | Create lazy inference path in `v7.rs`                          | High   | Steps 5-7    |
| 9    | Benchmark and tune                                             | Medium | Step 8       |
| 10   | Operation fusion optimization                                  | High   | Step 8       |

---

## Expected Benefits

| Benefit                    | Impact                     | When Realized |
| -------------------------- | -------------------------- | ------------- |
| Reduced memory allocations | 10-20% faster inference    | Phase 3       |
| Buffer reuse               | 20-40% less memory         | Phase 3       |
| Operation fusion           | 5-15% faster inference     | Phase 5       |
| Better memory planning     | Enables larger batch sizes | Phase 5       |

---

## Risks and Mitigations

| Risk                         | Mitigation                                        |
| ---------------------------- | ------------------------------------------------- |
| Breaking existing API        | Parallel API approach (Phase 4.1)                 |
| Performance regression       | Benchmark at each phase, keep eager fallback      |
| Complexity increase          | Clear separation of concerns, good documentation  |
| Graph overhead for small ops | Skip graph for single operations, batch threshold |

---

## Decision Points

1. **Graph storage:** `petgraph` vs custom `Vec<Node>` with indices?

    - Recommendation: Start with `Vec<Node>` for simplicity, migrate to `petgraph` if needed

2. **Thread safety:** `RwLock<ComputeGraph>` vs per-thread graphs?

    - Recommendation: Single `RwLock` graph, operations are GPU-bound anyway

3. **When to materialize:** Explicit `materialize()` vs implicit on read?
    - Recommendation: Explicit for control, with helper for common patterns

---

## Implementation Progress

### Phase 1-3: Core Infrastructure ✅ Complete

-   `src/tensor/graph.rs`: `ComputeGraph`, `ComputeNode`, `LazyOp`, `BufferPlan`, topological sort, `resolve()`
-   `src/tensor/lazy_tensor.rs`: `LazyTensor<T>` with `materialize()`, ref counting via Clone/Drop
-   `src/tensor/lazy_ops.rs`: Lazy operation builders
-   Context integration with `graph()` method

### Phase 4: Migration Strategy 🔄 In Progress

#### 4a: Core Lazy Operations ✅ Complete

Implemented lazy operations:

-   `MatMul` - matrix multiplication with activation
-   `Add`, `Mul` - element-wise operations
-   `LayerNorm`, `RmsNorm`, `GroupNorm` - normalization
-   `Softmax` - softmax activation
-   `Blit` - tensor copy
-   `Affine` - scale + bias transformation
-   `L2Norm` - L2 normalization
-   `Sigmoid`, `Tanh`, `SquaredRelu` - activation functions
-   `TokenShift` - cursor-free time mixing ✅ NEW
-   `Blend` - tensor-based blending ✅ NEW

#### 4b: Lazy Runtime Utilities ✅ Complete

Created `src/runtime/lazy_runtime.rs` with:

-   `LazyLayerNorm` - lazy layer normalization wrapper
-   `LazyLinear` - lazy matrix multiplication wrapper
-   `LazyFfn` - simplified lazy FFN block
-   `LazyStats` - graph statistics
-   `clear_graph_caches()` - memory management

#### 4c: Cursor-Free Operations ✅ Complete

Implemented cursor-free variants for lazy evaluation:

**TokenShift (cursor-free):**

-   New shader: `src/shaders/token_shift_lazy.wgsl`
-   New op: `TensorOp::token_shift_lazy()` - operates on explicit tensors without cursor-based indexing
-   State tensor is f32 (matching runtime state format)
-   Mix tensor broadcasts across tokens if shape is [C, 1, 1]

**Blend (tensor-based):**

-   New shader: `src/shaders/blend_tensor.wgsl`
-   New op: `TensorOp::blend_tensor()` - uses tensor factor instead of scalar
-   Formula: `lhs * factor + rhs * (1 - factor)`
-   Factor broadcasts if dimensions are 1

**Tests:**

-   `examples/test_lazy_tensor.rs` - 8 tests covering all lazy operations
-   All tests pass ✅

### Next Steps for Phase 4

1. ~~Implement cursor-free `TokenShift` variant for lazy evaluation~~ ✅
2. ~~Implement tensor-based `Blend` operation~~ ✅
3. Add lazy versions of v7-specific ops (time_mix_v7, channel_mix_v7) - **complex, requires significant shader work**
4. Create lazy inference path in v7.rs (parallel to eager path)
5. Benchmark lazy vs eager performance

### Phase 4d: V7 Integration ✅ Complete

The v7-specific operations (`time_mix_v7`, `channel_mix_v7`) are complex stateful operations that:

-   Use cursors for batch/token tracking
-   Update state in-place during execution
-   Have multiple input/output tensors with complex dependencies

**Implemented:**

1. **New cursor-free WGSL shaders:**

    - `src/shaders/time_mix_v7_lazy.wgsl` - Cursor-free WKV kernel with explicit batch_info uniform
    - `src/shaders/channel_mix_v7_lazy.wgsl` - Cursor-free channel mix with explicit batch_info uniform

2. **New TensorOp implementations in `src/tensor/ops.rs`:**

    - `TensorOp::time_mix_v7_lazy()` - Cursor-free time mix with batch_index/num_tokens params
    - `TensorOp::time_first_v7_lazy()` - Cursor-free time first operation
    - `TensorOp::channel_mix_v7_lazy()` - Cursor-free channel mix with batch_index/num_tokens params

3. **New LazyOp variants in `src/tensor/graph.rs`:**

    - `TimeMixV7` - V7 WKV kernel with batch_index and num_tokens
    - `ChannelMixV7` - V7 channel mix with batch_index and num_tokens
    - `TimeFirstV7` - Time first operation
    - `Lerp` - Linear interpolation with reversed option
    - `AddActivate` - Add with per-input and output activations
    - `ControlKV7` - Control K operation for V7 attention
    - `PackKvakk` - Pack k, v, a, kk into single tensor

4. **New lazy operation builders in `src/tensor/lazy_ops.rs`:**

    - `LazyTensor::time_mix_v7()` - Lazy time mix operation
    - `LazyTensor::channel_mix_v7()` - Lazy channel mix operation
    - `LazyTensor::time_first_v7()` - Lazy time first operation
    - `lerp()` - Lazy lerp with reversed option
    - `add_activate()` - Lazy add with activations
    - `control_k_v7()` - Lazy control K for V7
    - `pack_kvakk()` - Lazy pack operation (f16 only)

5. **Full execution handlers in `src/tensor/graph.rs`:**

    - `TimeMixV7` - Calls `TensorOp::time_mix_v7_lazy()`
    - `ChannelMixV7` - Calls `TensorOp::channel_mix_v7_lazy()`
    - `TimeFirstV7` - Calls `TensorOp::time_first_v7_lazy()`
    - `Lerp`, `AddActivate`, `ControlKV7`, `PackKvakk` - All fully implemented

6. **V7 lazy runtime utilities in `src/runtime/lazy_runtime.rs`:**
    - `LazyV7Att` - Lazy V7 attention layer wrapper with `forward()` method
    - `LazyV7AttOutput` - With `time_mix()`, `time_first()`, `finalize()`, and `complete()` methods
    - `LazyV7Ffn` - Lazy V7 FFN layer wrapper with `forward()` method
    - `LazyV7FfnOutput` - With `channel_mix()`, `finalize()`, and `complete()` methods
    - `LazyV7Config` - Configuration struct with default eps values

**Key Design Decisions:**

1. **Cursor-free shaders:** Created new shader variants that take `batch_info` as a uniform
   buffer containing `[batch_index, token_offset, num_tokens, 0]` instead of reading from
   a cursor array. This enables pure lazy evaluation without cursor dependencies.

2. **Explicit batch parameters:** The lazy operations take `batch_index` and `num_tokens`
   as explicit parameters, allowing the lazy graph to be built without knowing the
   cursor state at graph construction time.

3. **Complete methods:** Both `LazyV7AttOutput` and `LazyV7FfnOutput` provide `complete()`
   methods that chain all operations together for convenience.

**Usage Example:**

```rust
// Build lazy attention graph
let att_output = lazy_att.forward(&x, &state, layer_index, eps, gn_eps, l2_eps, turbo);

// Complete the attention with time_mix, time_first, and output projection
let x = att_output.complete(&u, &att_state, &residual, batch_index, num_tokens);

// Build lazy FFN graph
let ffn_output = lazy_ffn.forward(&x, &ffn_state, eps, turbo);

// Complete the FFN with channel_mix
let x = ffn_output.complete(&ffn_state, batch_index, num_tokens);

// Materialize the final result
let result = x.materialize()?;
```

### Phase 5: Advanced Optimizations ✅ Complete (MatMul+Bias Fusion)

#### 5.1 Operation Fusion (MatMul → Add → Activation)

**Implemented:**

1. **New `LazyOp::MatMulBias` variant** in `src/tensor/graph.rs`:

    - Fuses `matmul(matrix, input) + bias` into a single kernel
    - Supports activation function application after bias addition
    - Fields: `matrix`, `input`, `bias`, `activation`, `turbo`

2. **Fused shader** `src/shaders/matmul_vec_fp16_bias.wgsl`:

    - Based on `matmul_vec_fp16.wgsl` with added bias binding
    - Adds bias vector in the output step: `output = ACT(matmul_result + bias)`
    - Bias is broadcast across tokens and batches

3. **`TensorOp::matmul_vec_fp16_bias()`** in `src/tensor/ops.rs`:

    - Creates the fused matmul+bias compute pipeline
    - Currently supports `Matrix::Fp16` only (other quant types fall back to separate ops)

4. **Automatic fusion optimization** in `ComputeGraph::optimize()`:

    - Detects `MatMul → Add` patterns where:
        - MatMul has `activation: None`
        - Add's other operand is an Input node (bias tensor)
        - Bias shape is `[R, 1, 1, 1]` (broadcastable)
        - MatMul is only used by this Add (ref_count <= 2)
    - Rewrites the Add node to `MatMulBias`, bypassing the intermediate MatMul

5. **Lazy operation builder** `LazyTensor::matmul_bias()` in `src/tensor/lazy_ops.rs`:

    - Allows explicit creation of fused matmul+bias operations
    - Useful when the user knows they want fusion upfront

6. **Tests** in `examples/test_lazy_tensor.rs`:
    - `test_matmul_bias_fusion()` - Tests automatic fusion detection
    - `test_explicit_matmul_bias()` - Tests explicit `matmul_bias()` API

**Usage:**

```rust
// Automatic fusion (optimizer detects and fuses)
let lazy_matmul = lazy_input.matmul(&matrix, Activation::None, false);
let lazy_result = lazy_matmul.add(&lazy_bias);
let buffer = lazy_result.materialize()?; // Fusion happens here

// Explicit fusion (user creates fused op directly)
let lazy_result = lazy_input.matmul_bias(&matrix, &lazy_bias, Activation::None, false);
let buffer = lazy_result.materialize()?;
```

**Benefits:**

-   Eliminates intermediate buffer allocation for matmul output
-   Reduces kernel launch overhead (1 kernel instead of 2)
-   Expected 5-15% speedup for linear layers with bias

#### 5.2 Memory Planning with Graph Coloring

-   Not yet implemented
-   Future work: Use liveness analysis and graph coloring to minimize total memory usage

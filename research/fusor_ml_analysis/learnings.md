# Learnings from Floneum's fusor-ml/core

> Analysis of <https://github.com/floneum/floneum/tree/main/fusor-ml/core>
> Note: This crate is incomplete/in-development but contains valuable architectural patterns.

## Overview

fusor-ml/core is a WebGPU-based ML tensor library in Rust with:

-   Lazy evaluation via compute graphs
-   Native quantized type support (Q4_0, Q5_0, Q8_0, Q4K, Q6K)
-   Compile-time rank checking via const generics
-   Automatic kernel fusion potential via MIR (Mid-level IR)

---

## Key Architectural Patterns

### 1. Lazy Tensor Evaluation with Reference Counting

```rust
pub(crate) struct LazyTensorData {
    device: Device,
    info: TensorInfo,
    key: NodeIndex,  // Reference into compute graph
}

impl Clone for LazyTensorData {
    fn clone(&self) -> Self {
        self.device.compute_graph().add_reference(self.key);
        // ...
    }
}

impl Drop for LazyTensorData {
    fn drop(&mut self) {
        self.device.compute_graph().remove_reference(self.key);
    }
}
```

**Insight:** Tensors don't hold data directly—they hold a `NodeIndex` into a compute graph. Reference counting on graph nodes enables automatic cleanup of unused intermediate results.

**Applicability to web-rwkv-gguf:** Could defer GPU buffer allocation until `materialize()` is called, enabling better memory planning.

---

### 2. Compile-Time Rank Safety

```rust
pub struct Tensor<const R: usize, D> {
    data: LazyTensorData,
    datatype: PhantomData<D>,
}
```

Operations enforce rank at compile time:

```rust
impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self { ... }
}
```

**Insight:** Const generic `R` prevents rank mismatches at compile time rather than runtime panics.

**Applicability:** Our current `TensorGpu<T>` could be extended to `TensorGpu<const R: usize, T>` for safer shape handling.

---

### 3. Extensive Caching Strategy

```rust
struct DeviceInner {
    // Multiple LRU caches for different resource types
    bind_group_layout_cache: RwLock<LruCache<Vec<BindGroupLayoutEntry>, BindGroupLayout>>,
    pipeline_layout_cache: RwLock<LruCache<BindGroupLayout, PipelineLayout>>,
    shader_module_cache: RwLock<LruCache<String, ShaderModule>>,
    compute_pipeline_cache: RwLock<LruCache<(PipelineLayout, ShaderModule), ComputePipeline>>,
    buffer_allocation_cache: RwLock<LruCache<(u64, BufferUsages), Vec<CachedBuffer>>>,
}
```

**Key details:**

-   Cache size: 2048 entries for pipelines, 128 for buffers
-   Uses `FxBuildHasher` (fast non-cryptographic hash)
-   Buffer cache tracks initialization state to avoid redundant writes

**Applicability:** We already cache some pipelines, but fusor's buffer allocation cache with size+usage keying could reduce allocation overhead during inference.

---

### 4. Operation Trait for Kernel Generation

```rust
pub(crate) trait Operation: Debug {
    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints;
    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3];
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex));
    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue>;
    fn output(&self, nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue;
    fn build_kernel(&self, ..., kernel: &mut GenericKernel);
    fn name(&self) -> String;
}
```

**Insight:** Each operation type (MatMul, ElementWise, Reduce, etc.) implements this trait. The `build_kernel` method generates WGSL code into a `GenericKernel` builder.

**Applicability:** Our `TensorOp` enum could evolve toward this pattern for cleaner separation of concerns.

---

### 5. Generic Kernel Builder

```rust
pub(crate) struct GenericKernel {
    workgroup_size: [u32; 3],
    inputs: Vec<KernelInput>,
    functions: Vec<Function>,
    globals: Vec<KernelGlobal>,
    enabled_builtins: EnumSet<EnabledBuiltins>,
    quantized_type_definitions: EnumSet<GgmlType>,
    body: String,
    // ...
}
```

Methods like:

-   `add_tensor_input()` - registers a tensor binding
-   `add_q_matrix_input()` - registers a quantized matrix binding
-   `add_function()` - adds a helper function
-   `global_id()`, `subgroup_size()` - enables builtins on-demand

**Insight:** Kernel code is built programmatically, enabling:

1. Conditional feature enablement (f16, subgroups)
2. Automatic binding management
3. Quantized type definitions only when needed

**Applicability:** Could replace our manual WGSL string templates with a builder pattern for more maintainable shader generation.

---

### 6. Quantized Type Handling

```rust
pub struct QMatrix {
    device: Device,
    shape: Box<[usize]>,
    buffer: Arc<wgpu::Buffer>,
    datatype: GgmlType,
}
```

Dequantization is done via code generation:

```rust
pub(crate) fn dequantize_block<W: Write>(
    kernel: &mut W,
    ty: GgmlType,
    chunk: String,
    datatype: DataTypeEnum,
    mut process_element: impl FnMut(String, String, &mut W),
) {
    match ty {
        GgmlType::Q4_0 => BlockQ4_0::dequantize_block(chunk, datatype, process_element, kernel),
        GgmlType::Q4K => BlockQ4K::dequantize_block(chunk, datatype, process_element, kernel),
        // ...
    }
}
```

**Insight:** Each quantized type implements `WgslQuantizedType` trait with methods:

-   `dequantize_block` - loop-based dequantization
-   `unrolled_dequantize_block` - fully unrolled for small blocks
-   `dequantize_vec4_block` - vectorized 4-element dequantization
-   `dequantize_4x4_block` - matrix tile dequantization

**Applicability:** Our Q4K shaders use similar patterns but could benefit from the trait-based organization for supporting multiple quant types.

---

### 7. Adaptive MatMul Selection

```rust
pub fn get_optimal_params(m: usize, n: usize, k: usize) -> MatMulParams {
    match (m, n, k) {
        (_, 0..=64, _) => MatMulParams::Vector(gemv_parameters(m, n, k)),
        (_, _, _) => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}
```

**Insight:** Automatically selects between GEMV (vector) and GEMM (matrix) based on output dimensions. N ≤ 64 triggers vector path.

**Applicability:** We have separate `matmul_vec` and `matmul_mat` but selection is manual. Could add automatic dispatch.

---

### 8. Element-wise Operation Fusion

```rust
pub(crate) struct MatMulOperation {
    // ...
    pub(crate) pre_element_wise: [ElementWiseFunctions; 2],  // Applied to inputs
    pub(crate) post_element_wise: ElementWiseFunctions,      // Applied to output
    // ...
}
```

**Insight:** MatMul operations can have element-wise ops fused before/after, avoiding intermediate buffer allocations.

**Applicability:** Our current architecture doesn't support fusion. This could significantly improve performance for patterns like `matmul + bias + activation`.

---

### 9. Growable KV Cache - Ignore, no need for KV cache with RWKV

```rust
pub struct TensorCache<const R: usize, D> {
    all_data: Option<Tensor<R, D>>,
    current_seq_len: usize,
    allocated_seq_len: usize,  // Power-of-2 growth
    concat_dim: usize,
    max_sequence_len: usize,
}

pub fn append(&mut self, v: &Tensor<R, D>) -> Tensor<R, D> {
    // Doubles allocation when needed
    if required_seq_len > self.allocated_seq_len {
        let new_allocated_seq_len = required_seq_len.next_power_of_two();
        // ...
    }
    // Uses slice_assign for in-place update
    *cached = cached.slice_assign(slice, v);
}
```

**Insight:** Power-of-2 growth strategy with `slice_assign` for efficient appending without full reallocation.

**Applicability:** Our state management could adopt this pattern for variable-length generation.

---

### 10. Pipeline Caching to Disk

```rust
let filename = wgpu::util::pipeline_cache_key(&adapter.get_info());
let cache_path = cache_dir.join(&filename);
let cache_data = std::fs::read(&cache_path).ok();
let pipeline_cache = unsafe {
    device.create_pipeline_cache(&PipelineCacheDescriptor {
        data: cache_data.as_deref(),
        label: Some("Fusor ML Pipeline Cache"),
        fallback: true,
    })
};
```

**Insight:** Uses wgpu's pipeline cache feature to persist compiled shaders to disk, reducing startup time on subsequent runs.

**Applicability:** We could add pipeline caching for faster model loading, especially valuable for the Q4K shaders.

---

### 11. Background Device Polling

```rust
#[cfg(not(target_arch = "wasm32"))]
std::thread::spawn({
    let device = device.clone();
    move || loop {
        let Ok(status) = device.wgpu_device().poll(wgpu::PollType::wait_indefinitely())
        else { break; };
        if status == wgpu::PollStatus::QueueEmpty {
            std::thread::sleep(std::time::Duration::from_nanos(10));
        }
    }
});
```

**Insight:** Dedicated polling thread ensures GPU work completes promptly without blocking the main thread.

**Applicability:** We use `pollster::block_on` which is simpler but less efficient for async workloads.

---

### 12. VarBuilder Pattern for Model Loading

```rust
pub struct VarBuilder<'a> {
    reader: &'a mut dyn ReadAndSeek,
    metadata: Arc<GgufMetadata>,
    path: Vec<String>,
}

impl VarBuilder {
    pub fn pp<'b, S: ToString>(&'b mut self, s: S) -> VarBuilder<'b> {
        // Push path segment, returns scoped builder
    }

    pub fn get(&mut self, key: &str, device: &Device) -> Result<QMatrix> {
        let full_path = self.format_path(key);  // e.g., "blk.0.attn.weight"
        // ...
    }
}
```

**Insight:** Hierarchical path building for clean model loading code:

```rust
let mut vb = vb.pp("blk").pp(layer_idx);
let weight = vb.get("attn.weight", device)?;
```

**Applicability:** Our loader uses flat key lookups. This pattern would make layer loading code cleaner.

---

## Summary of Actionable Ideas

| Priority | Idea                                           | Effort | Impact                     |
| -------- | ---------------------------------------------- | ------ | -------------------------- |
| High     | Buffer allocation cache with size+usage keying | Medium | Reduce allocation overhead |
| High     | Pipeline cache persistence to disk             | Low    | Faster startup             |
| Medium   | Operation fusion (matmul + elementwise)        | High   | Better throughput          |
| Medium   | Adaptive GEMV/GEMM selection                   | Low    | Better small-batch perf    |
| Low      | Const-generic rank safety                      | High   | Compile-time safety        |
| Low      | Generic kernel builder                         | High   | Maintainability            |

---

## Dependencies of Interest

From their `Cargo.toml`:

-   `wgpu` (custom fork with yield-now for async)
-   `half` for f16 support
-   `fusor-gguf` for GGUF parsing
-   `petgraph` for compute graph representation
-   `lru` for caching
-   `enumset` for efficient flag sets
-   `rustc-hash` (FxHasher) for fast hashing

---

## Notes

-   The crate is incomplete—compute graph resolution logic wasn't fully accessible
-   Uses Rust 2024 edition features
-   Designed for both native and WASM targets (`WasmNotSend`/`WasmNotSync` traits)
-   Heavy use of `parking_lot::RwLock` over std for performance

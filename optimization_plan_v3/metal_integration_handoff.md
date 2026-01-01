# Metal Integration Handoff

## Status: Infrastructure Complete, Execution Blocked

The Metal backend infrastructure for Q4K quantized matrix multiplication is fully implemented, but efficient execution is blocked by a synchronization architecture issue between Metal and wgpu command queues.

## What's Implemented

### 1. Metal Context (`src/metal/`)

```
src/metal/
├── mod.rs           # Module exports, MetalContext, BufferId, buffer cache
├── kernels.rs       # MSL kernel source (Q4K matmul with simdgroup_matrix)
├── ops.rs           # MetalOps dispatch functions
└── buffer_bridge.rs # Zero-copy wgpu→Metal buffer extraction
```

**Key Components:**

-   `MetalContext` - Holds Metal device, command queue, pipeline state, and buffer cache
-   `BufferId` - Unique identifier for cached weight buffers (derived from tensor ID)
-   `get_metal_buffer_from_wgpu()` - Extracts underlying Metal buffer from wgpu buffer via unsafe wgpu-hal access

### 2. Context Integration (`src/context.rs`)

```rust
pub struct Context {
    // ... existing fields ...
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    metal: Option<Arc<MetalContext>>,
}
```

-   Metal context is created during `Context::new()` when `metal-backend` feature is enabled
-   Accessible via `context.metal()` method

### 3. Weight Buffer Caching (`src/runtime/loader.rs`)

During model loading, Q4K weight tensors are cached in Metal:

```rust
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
if let Some(metal_ctx) = context.metal() {
    let buffer_id = BufferId::from_tensor_id(tensor.id());
    metal_ctx.cache_q4k_weights(buffer_id, &raw_bytes);
}
```

**Result:** 128 Q4K weight buffers (one per layer matrix) are pre-loaded into Metal memory.

### 4. TensorOp Variant (`src/tensor/ops.rs`)

```rust
pub enum TensorOp {
    // ... existing variants ...
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    MetalMatmulQ4K {
        weight_id: u64,
        input: Arc<Buffer>,
        output: Arc<Buffer>,
        k: u32,
        m: u32,
        input_offset: u32,
        output_offset: u32,
    },
}
```

### 5. Metal Kernel Performance

Isolated benchmark results (Metal kernel only, no wgpu interop):

```
Metal Q4K matmul: 207.8 tok/s
WebGPU Q4K shader: 43.5 tok/s
Speedup: 4.78x
```

## The Problem: Synchronization

Metal and wgpu have **separate command queues** that don't automatically synchronize.

### Approach 1: Per-Operation Sync (Tested, Failed)

```rust
// For each Metal op:
queue.submit(wgpu_commands);
device.poll(Wait);  // Wait for wgpu to finish
metal_ctx.execute_matmul();  // Run Metal
```

**Result:** 10.7 tok/s (worse than baseline 15 tok/s)
**Cause:** Sync overhead per operation kills performance

### Approach 2: No Sync (Tested, Failed)

```rust
// Run Metal immediately when op is created
metal_ctx.execute_matmul();  // Metal reads stale data
queue.submit(wgpu_commands);  // wgpu writes input later
```

**Result:** Garbage output
**Cause:** Metal reads input buffer before wgpu writes to it

## Current State

Metal dispatch is **disabled** in `src/tensor/matrix.rs`:

```rust
Matrix::Q4K { w, s } => {
    // Metal integration: Per-op sync is too slow (10 tok/s vs 15 tok/s baseline).
    // Need batched Metal execution or pure Metal path.
    // For now, use WebGPU shader path.
    TensorOp::matmul_vec_q4k(w, s, input, output, act)
}
```

## Recommended Solutions

### Option A: Pure Metal Layer Execution

Run entire layer computation in Metal, sync only at layer boundaries:

```rust
// Per layer (not per matmul):
queue.submit(pre_layer_wgpu_ops);
device.poll(Wait);

// All 4 matmuls in single Metal command buffer
metal_ctx.execute_layer_matmuls(&[att_k, att_v, att_r, att_o, ffn_k, ffn_v]);

// Continue with wgpu for non-matmul ops
```

**Pros:** Minimal sync overhead (1 per layer vs 4+ per layer)
**Cons:** Requires restructuring layer dispatch

### Option B: Metal Shared Events

Use `MTLSharedEvent` for fine-grained synchronization:

```rust
// Create shared event
let event = device.new_shared_event();

// wgpu signals event after writing input
wgpu_encoder.signal_event(event, 1);

// Metal waits for event before reading
metal_encoder.wait_for_event(event, 1);
metal_encoder.encode_matmul(...);
metal_encoder.signal_event(event, 2);

// wgpu waits for Metal before reading output
wgpu_encoder.wait_for_event(event, 2);
```

**Pros:** Fine-grained, no CPU sync
**Cons:** Requires wgpu-hal access to encode events, complex

### Option C: Batched Metal Execution

Collect all Metal ops during encode, execute once after all wgpu ops:

```rust
// During encode: collect Metal ops
let metal_ops = collect_metal_ops(tensor_op);

// After wgpu submit:
queue.submit(wgpu_commands);
device.poll(Wait);

// Execute all Metal ops in single command buffer
metal_ctx.execute_batch(&metal_ops);
```

**Pros:** Simple, single sync point
**Cons:** All Metal ops run after all wgpu ops (may not work for dependent ops)

## Files to Modify for Each Approach

### Option A (Pure Metal Layer):

-   `src/runtime/v7.rs` - Restructure `dispatch_layer` to batch matmuls
-   `src/metal/ops.rs` - Add `execute_layer_matmuls` function
-   `src/tensor/matrix.rs` - Re-enable Metal dispatch

### Option B (Shared Events):

-   `src/metal/mod.rs` - Add shared event management
-   `src/tensor/ops.rs` - Encode event signals in wgpu passes
-   `src/metal/ops.rs` - Encode event waits in Metal command buffers

### Option C (Batched Execution):

-   `src/tensor/ops.rs` - Already has `BatchItem::Metal` infrastructure
-   `src/runtime/v7.rs` - Call `execute_metal_ops_sync` after submit
-   `src/tensor/matrix.rs` - Re-enable Metal dispatch

## Bug Fixes Made

Fixed shader validation error in `unpack4x16half` function (3 files):

-   `src/shaders/matmul_vec_q4k_f16.wgsl`
-   `src/shaders/matmul_mat_q4k_f16.wgsl`
-   `src/shaders/subgroup/matmul_vec_q4k_f16.wgsl`

The issue was `unpack2x16float` returns `vec2<f32>`, not `vec2<f16>`:

```wgsl
// Before (broken):
fn unpack4x16half(x: vec2<u32>) -> vec4<f16> {
    return vec4<f16>(unpack2x16float(x.x), unpack2x16float(x.y));
}

// After (fixed):
fn unpack4x16half(x: vec2<u32>) -> vec4<f16> {
    let lo = unpack2x16float(x.x);
    let hi = unpack2x16float(x.y);
    return vec4<f16>(f16(lo.x), f16(lo.y), f16(hi.x), f16(hi.y));
}
```

## Feature Flag

Enable Metal backend with:

```bash
cargo build --features metal-backend
```

## Test Commands

```bash
# Verify Metal context creation
cargo run --release --features metal-backend --example test_metal_integration

# Benchmark (currently uses WebGPU path)
cargo run --release --features metal-backend --example benchmark -- \
  --model assets/models/2.9b-Q4_K_M.gguf \
  --title "Test" --change "Description"

# Isolated Metal kernel benchmark
cargo run --release --features metal-backend --example bench_metal_vs_wgpu_real
```

## Key Learnings

1. **wgpu-hal buffer access works** - Can extract raw Metal buffer from wgpu buffer
2. **Metal kernel is 4.78x faster** - Significant potential if sync is solved
3. **Per-op sync is too expensive** - Need batched or event-based approach
4. **Buffer sharing works** - Same GPU memory, no copies needed

## Next Steps

1. Choose synchronization approach (recommend Option A or C)
2. Implement chosen approach
3. Re-enable Metal dispatch in `src/tensor/matrix.rs`
4. Benchmark to verify >90 tok/s target

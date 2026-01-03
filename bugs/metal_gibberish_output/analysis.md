# Metal Backend Gibberish Output Bug

## Status: Q8_0 FIX APPLIED

## Symptom

When running models with the Metal backend enabled (`--features metal-acceleration`), the output is gibberish/random text instead of coherent responses. This affects all quantization types (Q4K, Q8_0, F16) when processed through Metal.

**WebGPU works correctly:**

-   Q8_0: 47 tok/s, correct output ✓
-   F16: correct output ✓
-   Q4K: freezes (separate issue)

**Metal produces gibberish for all formats.**

## Q8_0 Fix Applied

**Root Cause Found:** The Metal Int8 kernel used incorrect minmax indexing.

**WebGPU shader** uses flat indexing:

```wgsl
fn unpack_minmax(index: u32) -> vec2<f32> {
    let i = index / INT8_BLOCK_STEP;  // INT8_BLOCK_STEP = 32
    return unpack2x16float(minmax[i]);
}
```

**Metal kernel** (BEFORE - WRONG):

```metal
uint mm_idx = (b * M + row) * num_blocks + block_idx;  // [batch][row][block] layout
```

**Metal kernel** (AFTER - FIXED):

```metal
uint mm_idx = mat_idx / INT8_BLOCK_STEP;  // Flat indexing like WebGPU
```

**Files Modified:**

-   `src/metal/v7_kernels.rs`: Fixed `matmul_int8` and `matmul_int8_squared_relu` kernels

**Test Command:**

```bash
cargo build --release --features metal-acceleration --example chat
./target/release/examples/chat --model assets/models/2.9b-Q8_0.gguf -a
```

## Architecture Overview

The Metal backend (`src/metal/`) provides native Metal execution for RWKV V7 layers:

```
src/metal/
├── mod.rs              # Module exports
├── buffer_bridge.rs    # wgpu → Metal buffer extraction
├── context.rs          # MetalContext with command queue
├── kernels.rs          # Legacy kernels
├── ops.rs              # FFN-only Metal ops
├── v7_kernels.rs       # Complete V7 layer Metal shaders
└── v7_layer.rs         # V7 layer orchestration
```

### Execution Flow

1. `dispatch_layer()` in `v7.rs` calls `try_metal_v7_layer()`
2. If Metal is available, `execute_metal_v7_layer()` runs the entire layer
3. All operations use Metal command buffers with a single sync at the end
4. If Metal fails, falls back to WebGPU

## Potential Causes

### 1. Buffer Synchronization Issues

**Problem:** The Metal backend extracts raw Metal buffers from wgpu buffers using `BufferBridge::extract_from_arc()`. This bypasses wgpu's synchronization mechanisms.

```rust
// buffer_bridge.rs:53
let raw_ptr: *const metal::Buffer = hal_buffer as *const _ as *const metal::Buffer;
let metal_buffer: &metal::Buffer = &*raw_ptr;
```

**Risk:**

-   wgpu may have pending writes that haven't been flushed
-   Metal reads stale data from buffers
-   No memory barrier between wgpu and Metal operations

**Evidence:** The code does `cmd.commit(); cmd.wait_until_completed();` at the end, but there's no sync _before_ Metal starts reading wgpu buffers.

### 2. Shape/Dimension Mismatch

**Problem:** The Metal kernels extract K and M dimensions from the matrix shape:

```rust
// v7_layer.rs:932
let [k, m, _, _] = s.shape().into();
```

This assumes `shape[0] = K` and `shape[1] = M`. But the shape convention in the codebase is:

-   After `from_slice_rev([M, K])`: `Shape::new(K, M, 1, 1)` → `shape[0] = K`, `shape[1] = M`

This appears correct, but needs verification against the actual data layout.

### 3. Data Layout Mismatch

**Problem:** The Metal Q4K kernel assumes row-major layout:

```metal
// v7_kernels.rs:263
uint block_u32_base = (current_row * num_super_blocks_k + sb) * Q4K_BLOCK_U32;
```

This is `block[row][sb]` layout. But the WebGPU loader now uses non-transposed data directly. Need to verify the Metal kernel matches the WebGPU shader's expectations.

### 4. Int8 Matrix Layout

**Problem:** The Metal Int8 kernel assumes a specific layout:

```metal
// v7_kernels.rs:856
uint mat_idx = (b * M + row) * stride + i;
```

This is `matrix[batch][row][k/4]`. The WebGPU Int8 path may use a different layout.

### 5. Missing wgpu Queue Submission

**Problem:** Before Metal can read from wgpu buffers, wgpu must have submitted and completed its work. The current code doesn't ensure this:

```rust
// v7_layer.rs - no wgpu queue sync before Metal execution
let cmd = metal_ctx.ctx.queue().new_command_buffer();
// ... immediately starts reading wgpu buffers
```

### 6. State Buffer Offset Issues

**Problem:** State tensors use views with offsets:

```rust
let metal_att_state = unsafe { BufferBridge::extract_from_arc(&att_state.tensor().buffer)? };
```

The Metal kernels may not account for the view's offset, reading from the wrong location in the state buffer.

## Diagnosis Plan

### Step 1: Add Synchronization Before Metal

Before Metal reads any wgpu buffers, ensure wgpu has flushed:

```rust
// In execute_metal_v7_layer_inner, before extracting buffers:
_context.queue.submit(std::iter::empty());
_context.device.poll(wgpu::Maintain::Wait);
```

### Step 2: Verify Buffer Contents

Add debug code to read back buffer contents before Metal execution and compare with expected values:

```rust
// Read x buffer before Metal
let x_data = x.back().await;
log::debug!("x[0..10] = {:?}", &x_data[0..10]);
```

### Step 3: Test Individual Kernels

Create a test that runs a single Metal kernel (e.g., layer_norm) and compares output with WebGPU:

```rust
// Run layer_norm via Metal
// Run layer_norm via WebGPU
// Compare outputs
```

### Step 4: Check State View Offsets

Verify that Metal kernels correctly handle state tensor view offsets:

```rust
let att_state_offset = att_state.offset();
log::debug!("att_state offset: {:?}", att_state_offset);
```

## Fix Plan

### Phase 1: Synchronization Fix (High Priority)

1. Add wgpu queue submission before Metal execution
2. Add device poll to ensure completion
3. Test if this alone fixes the issue

### Phase 2: Buffer Validation (Medium Priority)

1. Add debug mode that validates buffer contents
2. Compare Metal input buffers with WebGPU expectations
3. Identify which operation first produces incorrect output

### Phase 3: Kernel Audit (Medium Priority)

1. Compare Metal Q4K kernel with WebGPU `matmul_vec_q4k.wgsl`
2. Compare Metal Int8 kernel with WebGPU `matmul_vec_int8.wgsl`
3. Verify data layout assumptions match

### Phase 4: State Handling (Low Priority)

1. Audit state tensor view offset handling
2. Ensure Metal kernels use correct offsets
3. Add offset parameters to kernel structs if needed

## Quick Test

To verify if synchronization is the issue, add this before Metal execution:

```rust
// In src/runtime/v7.rs, before try_metal_v7_layer call:
context.queue.submit(std::iter::empty());
context.device.poll(wgpu::Maintain::Wait);
```

## Files to Modify

-   `src/metal/v7_layer.rs` - Add sync before buffer extraction
-   `src/runtime/v7.rs` - Add wgpu flush before Metal call
-   `src/metal/buffer_bridge.rs` - Consider adding sync helper

## Related Issues

1. **Q4K WebGPU Freeze**: The Q4K model freezes on WebGPU. This is a separate issue, possibly related to the non-transposed shader dispatch calculation or an infinite loop in the shader.

2. **Shape Convention**: The shape convention (`shape[0] = K` vs `shape[0] = M`) has been a source of bugs. A comprehensive audit of all paths is recommended.

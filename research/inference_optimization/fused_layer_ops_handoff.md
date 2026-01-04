# Fused Layer Operations - Implementation Handoff

## Summary

Implemented fused token shift + layer norm shader. Overcame WebGPU binding limit by increasing `max_storage_buffers_per_shader_stage` to 20. Achieved **3.8% speedup** (55.60 vs 53.55 tok/s) but with a **quality bug** that produces different outputs.

## Current Status

**Working but with quality difference:**

-   Fused shader compiles and runs
-   3.8% speedup in generation (55.60 vs 53.55 tok/s)
-   Quality hash differs from baseline (different token outputs)
-   First token matches, subsequent tokens differ

## Root Cause Analysis

The quality difference is likely due to **incorrect tensor indexing**. The original shaders use view-based indexing via `compute_index(view, batch, token, index)` which handles tensor strides and offsets. The fused shader uses a simplified direct calculation `(batch * shape[1] + stack) * stride` which may not match the actual memory layout.

**To fix:** Add proper view metadata (uniform buffers) for each output tensor and use `compute_index()` for all tensor accesses.

## What Was Implemented

1. **Branch**: `feature/fused-layer-operations`
2. **New Files**:

    - `src/shaders/fused_token_shift_ln.wgsl` - Fused shader combining LayerNorm + 6 token shifts
    - `src/runtime/v7_fused.rs` - Copy of v7.rs using the fused operation
    - `examples/benchmark_fused.rs` - Benchmark with `--fused` flag
    - `examples/chat_fused.rs` - Chat example with `--fused` flag

3. **Modified Files**:
    - `src/tensor/ops.rs` - Added `TensorOp::fused_token_shift_ln()`
    - `src/runtime/mod.rs` - Added `v7_fused` module

## The Problem: WebGPU Binding Limit

**Error**: `Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE), limit is 8, count was 17`

The fused shader requires 19 bindings:

-   1 uniform (shape)
-   1 uniform (state view)
-   2 storage read (ln_w, ln_b)
-   1 storage read (cursors)
-   1 storage read (state)
-   1 storage read (input x)
-   6 storage read (mix_r, mix_w, mix_k, mix_v, mix_a, mix_g)
-   6 storage read_write (out_r, out_w, out_k, out_v, out_a, out_g)

WebGPU limits: **8 storage buffers per shader stage** (configurable via `max_storage_buffers_per_shader_stage`)

## Solutions Attempted

1. **Multiple Bind Groups**: WebGPU allows up to 4 bind groups, but the current `Context::checkout_pipeline()` only supports a single bind group layout. Modifying this would require significant infrastructure changes.

2. **Packed Buffers**: Tried packing multiple tensors into single buffers with offsets. This adds complexity and may not improve performance.

## Viable Alternatives

### Option A: Modify Context to Support Multiple Bind Groups

-   Requires changes to `src/context.rs`
-   `checkout_pipeline()` needs to accept multiple `BindGroupLayoutDescriptor`s
-   `BindGroupBuilder` needs group index support
-   Estimated effort: Medium-High

### Option B: Reduce Fusion Scope

Fuse fewer operations to stay within 8 bindings:

-   **LayerNorm + 2 token shifts**: 8 bindings (just fits)
-   Run 3 fused kernels instead of 1 (still 3x fewer than 7 separate ops)

### Option C: Request Higher Limits

```rust
let limits = wgpu::Limits {
    max_storage_buffers_per_shader_stage: 16,
    ..Default::default()
};
```

-   May not be supported on all hardware
-   Apple M-series GPUs support up to 31 storage buffers

### Option D: Alternative Optimizations

Focus on other optimizations from the research doc that don't hit binding limits:

-   **Command Buffer Batching** (02) - Remove Sep markers
-   **Matmul Shader Optimization** (04) - Tiled matmul with shared memory
-   **GPU-side Sampling** (05) - Avoid 260KB readback per token

## Recommendation

**Option C** is the quickest path forward for macOS:

1. Check if the device supports higher limits
2. Request 16 storage buffers in `ContextBuilder`
3. Fall back to separate ops if not supported

**Option D** may provide better ROI:

-   Command buffer batching is low-effort, high-impact
-   GPU-side sampling eliminates a major bottleneck

## Files to Clean Up

If abandoning this approach:

```bash
rm src/shaders/fused_token_shift_ln.wgsl
rm src/runtime/v7_fused.rs
rm examples/benchmark_fused.rs
rm examples/chat_fused.rs
# Revert changes to src/tensor/ops.rs and src/runtime/mod.rs
```

## Test Commands

```bash
# Build
cargo build --release --example benchmark_fused

# Run baseline
./target/release/examples/benchmark_fused --model assets/models/2.9b-Q8_0.gguf --title "Baseline" --change "No fusion"

# Run fused (currently fails due to binding limit)
./target/release/examples/benchmark_fused --model assets/models/2.9b-Q8_0.gguf --title "Fused" --change "Fused LN+TS" --fused
```

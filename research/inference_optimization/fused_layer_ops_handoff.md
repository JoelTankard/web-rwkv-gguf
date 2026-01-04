# Fused Layer Operations - Implementation Handoff

## Summary

Successfully implemented fused token shift + layer norm shader for the attention block. Achieved **2-5% speedup** with correct output. FFN fusion was also implemented but showed no benefit and is disabled.

## Final Results

| Metric     | Baseline         | Fused ATT        | Improvement     |
| ---------- | ---------------- | ---------------- | --------------- |
| Generation | 49-54 tok/s      | 54.7-55.2 tok/s  | **+2-5%**       |
| Variance   | High             | Low              | More consistent |
| Quality    | c55251c506e49c98 | c55251c506e49c98 | ✓ Match         |

## What Works

**Attention LayerNorm + 6 Token Shifts (7 ops → 1):**

-   Fuses layer_norm + 6 token_shift operations into single kernel
-   Only used for single-token inference (generation)
-   Falls back to separate ops for multi-token prefill
-   Requires `max_storage_buffers_per_shader_stage = 20` (set in auto_limits)

## What Didn't Work

**FFN LayerNorm + Token Shift (2 ops → 1):**

-   Implemented but showed no measurable benefit
-   Likely because FFN only has 1 token shift vs 6 in attention
-   The overhead of the fused kernel outweighs the dispatch savings
-   Code is present but disabled in v7_fused.rs

**Fused LoRA Triple (Idea 2):**

-   Skipped - requires custom two-stage matmul kernel
-   The LoRA operations involve actual matrix multiplications with different weight matrices
-   Fusing matmul pairs is complex and would require significant shader work

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

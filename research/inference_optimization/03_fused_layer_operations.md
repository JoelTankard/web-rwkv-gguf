# Fused Layer Operations

## Status: ✅ IMPLEMENTED (Partial)

**Result:** 2-5% speedup with fused attention LayerNorm + Token Shifts

| Metric     | Baseline         | Fused ATT        | Improvement     |
| ---------- | ---------------- | ---------------- | --------------- |
| Generation | 49-54 tok/s      | 54.7-55.2 tok/s  | **+2-5%**       |
| Variance   | High             | Low              | More consistent |
| Quality    | c55251c506e49c98 | c55251c506e49c98 | ✓ Match         |

**Implementation:** `src/runtime/v7_fused.rs`, `src/shaders/fused_token_shift_ln.wgsl`

---

## Current Implementation

Location: `src/runtime/v7.rs:813-1199` (`dispatch_layer`)

Each V7 layer executes ~40+ separate GPU dispatches:

```rust
ops.extend([
    TensorOp::blit(&buffer.x, &buffer.att_x)?,           // 1
    TensorOp::layer_norm(...),                            // 2
    TensorOp::token_shift(...),                           // 3-8 (6 shifts)
    layer.att.w_r.matmul_op(...),                         // 9
    layer.att.w_k.matmul_op(...),                         // 10
    layer.att.w_v.matmul_op(...),                         // 11
    // ... 30+ more operations
]);
```

## Problem

1. **Kernel launch overhead**: Each `TensorOp::Atom` is a separate GPU dispatch
2. **Memory bandwidth**: Intermediate results written to VRAM, then read back
3. **No data locality**: Operations don't share registers/shared memory

## Optimization Ideas

### Idea 1: Fused Token Shift + LayerNorm ✅ IMPLEMENTED

**Status:** Working, 2-5% speedup

Combine the 6 token shifts with layer norm into single kernel:

```wgsl
// fused_token_shift_ln.wgsl
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let token = id.y;
    let emb = id.x * 4;

    // Load x once
    let x = load_x(emb, token);

    // Compute layer norm in registers
    let x_normed = layer_norm(x, ln_w, ln_b);

    // Compute all 6 token shifts from same input
    let x_r = token_shift(x_normed, mix_r, state);
    let x_w = token_shift(x_normed, mix_w, state);
    let x_k = token_shift(x_normed, mix_k, state);
    let x_v = token_shift(x_normed, mix_v, state);
    let x_a = token_shift(x_normed, mix_a, state);
    let x_g = token_shift(x_normed, mix_g, state);

    // Store all outputs
    store_outputs(x_r, x_w, x_k, x_v, x_a, x_g);
}
```

**Benefit:** 7 dispatches → 1 dispatch, 7x less memory traffic for `x`

### Idea 2: Fused Attention LoRA (w1→w2, a1→a2, g1→g2) ❌ SKIPPED

**Status:** Not implemented - requires custom two-stage matmul kernel

The LoRA operations involve actual matrix multiplications with different weight matrices. Fusing matmul pairs would require a custom shader that keeps the intermediate (LoRA rank ~32-128) in shared memory between two matmul stages. This is complex to implement correctly.

The LoRA-style projections are small and can be fused:

```rust
// Current: 6 separate matmuls
layer.att.w1.matmul_op(&buffer.att_wx, &buffer.aux_w, Activation::Tanh)?
layer.att.w2.matmul_op(&buffer.aux_w, &buffer.att_w, Activation::None)?
layer.att.a1.matmul_op(&buffer.att_ax, &buffer.aux_a, Activation::None)?
layer.att.a2.matmul_op(&buffer.aux_a, &buffer.att_a, Activation::None)?
// ...
```

Fused version:

```wgsl
// fused_lora_triple.wgsl
// Computes: w = w2(tanh(w1(x_w))) + w0
//           a = sigmoid(a2(a1(x_a)) + a0)
//           g = g2(sigmoid(g1(x_g)))
// All in one kernel with shared memory for intermediates
```

**Benefit:** 6 matmuls → 1 kernel, intermediate `aux_*` stays in shared memory

### Idea 3: Fused Time Mix Core

The time mix operation has multiple steps that can be fused:

```rust
// Current
TensorOp::pack_kvakk(...)?
TensorOp::time_mix_v7(...)?
TensorOp::group_norm(...)?
TensorOp::time_first_v7(...)?
```

These all operate on the same data and can share registers.

### Idea 4: Fused FFN Block ⚠️ TESTED - NO BENEFIT

**Status:** Implemented FFN LayerNorm + Token Shift fusion, but showed no measurable benefit

FFN only has 1 token shift (vs 6 in attention), so the overhead of the fused kernel outweighs the dispatch savings. The shader exists (`src/shaders/fused_ffn_ln_ts.wgsl`) but is disabled in `v7_fused.rs`.

The FFN is a simple pattern that's highly fusable:

```rust
// Current
layer.ffn.w_k.matmul_op(&buffer.ffn_kx, &buffer.ffn_k, Activation::SquaredRelu)?
layer.ffn.w_v.matmul_op_sparse(&buffer.ffn_k, &buffer.ffn_v, Activation::None)?
TensorOp::channel_mix_v7(...)?
TensorOp::add(&buffer.ffn_x, &buffer.x)?
```

Fused:

```wgsl
// fused_ffn_v7.wgsl
// w_k matmul with squared_relu → w_v matmul (sparse) → channel_mix → residual add
// Intermediate ffn_k stays in shared memory (never hits VRAM)
```

**Benefit:** Eliminates `ffn_k` buffer entirely for single-token inference

### Idea 5: Persistent Kernel Approach

Instead of launching many small kernels, use a persistent kernel that processes the entire layer:

```wgsl
// persistent_v7_layer.wgsl
@compute @workgroup_size(256)
fn main() {
    // Each workgroup processes one "slice" of the layer
    // Synchronize via atomics/barriers instead of kernel launches

    // Phase 1: Token shift + LN
    barrier();

    // Phase 2: QKV projection
    barrier();

    // Phase 3: Time mix
    barrier();

    // Phase 4: Output projection + residual
    barrier();

    // Phase 5: FFN
}
```

**Benefit:** Single kernel launch per layer, maximum data locality

## Estimated Impact

| Fusion            | Dispatches Saved | Memory Bandwidth Saved | Complexity |
| ----------------- | ---------------- | ---------------------- | ---------- |
| Token Shift + LN  | 6                | ~6x for input tensor   | Medium     |
| LoRA Triple       | 4                | ~3x for aux tensors    | Medium     |
| Time Mix Core     | 3                | ~2x                    | High       |
| FFN Block         | 3                | ~2x (eliminates ffn_k) | Medium     |
| Persistent Kernel | ~35              | Maximum                | Very High  |

## Recommended Experiment

Start with **Fused Token Shift + LayerNorm** as it:

-   Has clear boundaries
-   Doesn't change algorithm
-   Provides measurable benefit
-   Is relatively simple to implement

## Implementation Steps

1. Create `src/shaders/fused_token_shift_ln.wgsl`
2. Add `TensorOp::fused_token_shift_ln()` in `ops.rs`
3. Modify `dispatch_layer()` to use fused op when available
4. Benchmark single-token and batch inference

## Files to Modify

-   `src/shaders/fused_token_shift_ln.wgsl` (new)
-   `src/tensor/ops.rs`: New fused operation
-   `src/runtime/v7.rs`: Use fused op in `dispatch_layer`

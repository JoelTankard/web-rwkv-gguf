# Option 5: RWKV-Specific Architecture Optimizations

## RWKV vs Transformer

RWKV has unique characteristics that affect optimization:

| Aspect           | Transformer        | RWKV                      |
| ---------------- | ------------------ | ------------------------- |
| Attention        | O(n²) KV cache     | O(1) state                |
| Memory per token | Grows with context | Constant                  |
| Parallelism      | High (all tokens)  | Medium (sequential state) |
| Bottleneck       | KV cache bandwidth | State update compute      |

## RWKV-Specific Bottlenecks

### 1. State Updates

RWKV maintains per-layer state that must be updated sequentially:

```
state[t] = f(state[t-1], input[t])
```

This creates a **sequential dependency** that limits parallelism.

### 2. Time Mixing (WKV)

The WKV operation is unique to RWKV:

```
wkv = Σ(exp(-(t-i)*w + k[i]) * v[i]) / Σ(exp(-(t-i)*w + k[i]))
```

This is compute-intensive and doesn't benefit directly from weight quantization.

### 3. Channel Mixing (FFN)

Standard FFN with squared ReLU:

```
ffn = W_v @ (square_relu(W_k @ x))
```

This is where weight quantization helps most.

## Optimization Strategies

### Strategy 1: Fused WKV Kernel

Current implementation may have multiple kernel launches for WKV.
Fuse into a single kernel:

```wgsl
@compute @workgroup_size(256, 1, 1)
fn fused_wkv(
    // Inputs
    r: array<f16>,
    k: array<f16>,
    v: array<f16>,
    w: array<f16>,
    state: array<f32>,
    // Output
    output: array<f16>,
    new_state: array<f32>,
) {
    // Compute r*k, update state, compute output in one pass
    // Avoid intermediate buffer writes
}
```

**Expected improvement**: 10-20% for generation

### Strategy 2: State Compression

Current state is F32 for stability. Consider:

-   **F16 state** with periodic renormalization
-   **Quantized state** (INT8 with scale)
-   **Sparse state** (only store significant values)

```rust
enum StateFormat {
    F32,           // Current: 4 bytes/element
    F16,           // 2 bytes/element, some precision loss
    Int8Scaled,    // 1 byte/element + scale
}
```

**Risk**: Quality degradation, especially for long contexts.

### Strategy 3: Chunked State Updates

Instead of updating state token-by-token, batch updates:

```
// Current: Sequential
for token in tokens:
    state = update(state, token)

// Proposed: Chunked (for prefill)
for chunk in tokens.chunks(64):
    state = batch_update(state, chunk)  // Parallel within chunk
```

**Benefit**: Better GPU utilization during prefill.

### Strategy 4: Lazy State Materialization

Don't compute full state until needed:

```rust
enum LazyState {
    Materialized(TensorGpu<f32>),
    Deferred {
        base: Arc<LazyState>,
        delta: TensorGpu<f16>,
    },
}

impl LazyState {
    fn materialize(&self) -> TensorGpu<f32> {
        match self {
            Materialized(s) => s.clone(),
            Deferred { base, delta } => {
                base.materialize() + delta.to_f32()
            }
        }
    }
}
```

**Benefit**: Reduce state read/write bandwidth.

### Strategy 5: Attention-Free Layers

Some RWKV layers contribute less to output quality.
Consider skipping or simplifying:

```rust
fn should_skip_layer(layer_idx: usize, token_idx: usize) -> bool {
    // Skip middle layers for early tokens (less important)
    let is_middle = layer_idx > 8 && layer_idx < 24;
    let is_early_token = token_idx < 10;
    is_middle && is_early_token
}
```

**Risk**: Quality degradation, needs careful tuning.

## RWKV v7 Specific

RWKV v7 introduces new components:

-   **LoRA-style adapters** (w1, w2, a1, a2, g1, g2, v1, v2)
-   **Group normalization** in attention
-   **Additional state dimensions**

These add compute overhead but also optimization opportunities:

-   Adapter matrices are small → keep in F16
-   Group norm can be fused with previous operation

## Performance Expectations

| Optimization    | Speed Impact   | Quality Impact | Complexity |
| --------------- | -------------- | -------------- | ---------- |
| Fused WKV       | +15%           | 0%             | Medium     |
| F16 state       | +10%           | -2%            | Low        |
| Chunked updates | +20% (prefill) | 0%             | Medium     |
| Lazy state      | +5%            | 0%             | High       |
| Layer skipping  | +30%           | -5%            | Medium     |

## Recommendation

**Priority: High for fused kernels, Medium for others**

1. **Fused WKV kernel** - Pure optimization, no quality loss
2. **Chunked state updates** - Significant prefill improvement
3. **F16 state** - Easy win if quality acceptable

## Implementation Priority

```
1. Profile current WKV implementation
2. Implement fused WKV kernel
3. Benchmark chunked state updates
4. Test F16 state quality impact
```

## Implementation Results (2026-01-01)

### Baseline Performance

-   **Model**: 2.9b-Q4_K_M.gguf (RWKV v7)
-   **GPU**: Apple M2 Max
-   **Generation**: 44.53 tok/s
-   **Prefill**: 152.25 tok/s
-   **Quality Hash**: `3895ad4add71cff0`

### Attempted Optimizations

| Optimization         | Generation  | Prefill      | Quality | Result             |
| -------------------- | ----------- | ------------ | ------- | ------------------ |
| Baseline             | 44.53 tok/s | 152.25 tok/s | ✓       | -                  |
| State access pattern | 44.10 tok/s | 151.80 tok/s | ✓       | No improvement     |
| Workgroup size 64    | 43.58 tok/s | 150.93 tok/s | ✓       | Worse (-2%)        |
| Fused state loops    | 44.36 tok/s | 154.70 tok/s | ✗       | Quality regression |

### Key Findings

1. **Current implementation is already well-optimized**: The time_mix_v7 shader uses efficient patterns with shared memory and vectorized operations.

2. **Sequential dependency is fundamental**: The WKV computation requires `sa` to be fully computed before state updates, preventing loop fusion without quality loss.

3. **Workgroup size 32 is optimal**: Increasing to 64 reduced performance, likely due to register pressure on Apple Silicon.

4. **F16 state requires significant refactoring**: Would need changes to:

    - State struct definition in v7.rs
    - `super::model::State` trait (uses `TensorCpu<f32>`)
    - All shader bindings for state
    - Serialization/deserialization logic

5. **Kernel fusion opportunities are limited**: Operations like `pack_kvakk` → `time_mix_v7` could be fused but require significant binding changes for uncertain gains.

### Recommendations

1. **F16 State** (Medium effort, potential +10%): Most promising optimization but requires careful implementation to avoid quality degradation.

2. **Chunked State Updates** (High effort, potential +20% prefill): Would require architectural changes to batch token processing.

3. **Layer Skipping** (Medium effort, potential +30%): Risky for quality but could be configurable for speed-critical applications.

The current implementation appears to be near-optimal for the RWKV v7 architecture on WebGPU. Further gains likely require:

-   Hardware-specific tuning (different workgroup sizes per GPU)
-   Algorithmic changes (approximate attention, pruning)
-   Lower-level optimizations (wgpu-hal, native backends)

# Option 4: Hybrid Precision Strategies

## Current State

All intermediate computations use F16:

-   Activations: F16
-   State: F32 (for stability)
-   Accumulation: F32 in shader, stored as F16

## The Precision Bottleneck

Even with Q4K weights, the **activation pipeline** dominates:

```
Token → Embed(F16) → LayerNorm(F16) → Attention(F16) → FFN(F16) → Output(F16)
                          ↓
                    State(F32)
```

Each layer reads/writes ~5KB of F16 activations per token.

## Strategy 1: Selective F32 Accumulation

Keep critical paths in F32, others in F16:

| Operation           | Current | Proposed      |
| ------------------- | ------- | ------------- |
| MatMul accumulation | F32     | F32 (keep)    |
| LayerNorm           | F16     | F16 (keep)    |
| Attention scores    | F16     | F32 (upgrade) |
| State updates       | F32     | F32 (keep)    |

**Impact**: Better numerical stability, minimal speed change.

## Strategy 2: BF16 for Activations

BF16 (Brain Float 16) has:

-   Same range as F32 (8 exponent bits)
-   Less precision (7 mantissa bits vs 10 for F16)
-   Better for training/inference stability

**Problem**: WebGPU doesn't support BF16 natively. Would need emulation.

## Strategy 3: Mixed Quantization by Layer

Not all layers are equally sensitive. Use higher precision for:

-   First few layers (feature extraction)
-   Last few layers (output generation)
-   Attention layers (precision-sensitive)

Lower precision for:

-   Middle FFN layers (more redundant)

```rust
enum LayerQuant {
    Q4K,  // 4-bit weights
    Q6K,  // 6-bit weights (more accurate)
    F16,  // Full precision
}

fn get_layer_quant(layer_idx: usize, total_layers: usize) -> LayerQuant {
    if layer_idx < 2 || layer_idx >= total_layers - 2 {
        LayerQuant::Q6K  // Higher precision at boundaries
    } else {
        LayerQuant::Q4K  // Lower precision in middle
    }
}
```

## Strategy 4: Dynamic Precision Based on Token Position

During prefill (many tokens): Use lower precision (speed matters)
During generation (one token): Use higher precision (quality matters)

```rust
fn select_precision(num_tokens: usize) -> Precision {
    if num_tokens > 1 {
        Precision::Q4K  // Prefill: speed priority
    } else {
        Precision::Q6K  // Generation: quality priority
    }
}
```

## Strategy 5: Speculative Decoding with Mixed Precision

Use a smaller/faster model for draft tokens, larger model for verification:

```
Draft model (Q4K, fast) → Generate N candidate tokens
Main model (Q6K, accurate) → Verify/reject candidates
```

This can achieve 2-3x speedup if draft acceptance rate is high.

## Performance Expectations

| Strategy             | Speed Impact | Quality Impact | Complexity |
| -------------------- | ------------ | -------------- | ---------- |
| Selective F32        | 0%           | +5%            | Low        |
| BF16 emulation       | -10%         | +10%           | High       |
| Mixed layer quant    | +10%         | -2%            | Medium     |
| Dynamic precision    | +20%         | -1%            | Medium     |
| Speculative decoding | +100-200%    | 0%             | High       |

## Recommendation

**Priority: Medium**

1. **Speculative decoding** offers the best speedup potential but requires significant architecture changes
2. **Mixed layer quantization** is a quick win with minimal quality loss
3. **Dynamic precision** is easy to implement and provides modest gains

## Implementation Notes

For mixed layer quantization, modify the loader:

```rust
fn load_layer_matrix(
    layer_idx: usize,
    total_layers: usize,
    tensor_data: &[u8],
    ggml_type: GgmlType,
) -> Matrix {
    let target_quant = get_layer_quant(layer_idx, total_layers);
    match (ggml_type, target_quant) {
        (GgmlType::Q4K, LayerQuant::Q4K) => load_q4k_native(tensor_data),
        (GgmlType::Q4K, LayerQuant::Q6K) => dequant_and_requant_q6k(tensor_data),
        _ => load_default(tensor_data),
    }
}
```

## References

-   [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
-   [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

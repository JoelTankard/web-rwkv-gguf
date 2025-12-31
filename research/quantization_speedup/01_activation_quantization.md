# Option 1: Activation Quantization (INT8 Activations)

## Concept

Currently only weights are quantized. Activations (the data flowing between layers) remain F16.
Quantizing activations to INT8 would:

-   **Halve activation memory bandwidth**
-   **Enable INT8 matmul** (faster on some hardware)
-   **Reduce memory footprint** during inference

## How llama.cpp Does It

llama.cpp keeps activations in F16/F32 but benefits from:

1. Highly optimized Metal/CUDA kernels
2. Memory-bound workloads where weight quantization alone helps

However, some frameworks (TensorRT-LLM, vLLM) use **W4A8** (4-bit weights, 8-bit activations).

## Implementation Approach

### Per-Token Dynamic Quantization

```
For each layer:
  1. Compute activation in F16
  2. Find absmax of activation tensor
  3. Quantize to INT8: q = round(x * 127 / absmax)
  4. Pass INT8 + scale to next layer
  5. Dequantize at start of next matmul
```

### Challenges

1. **Accuracy degradation**: Activations have outliers that hurt quantization
2. **Quantization overhead**: Finding absmax requires a reduction
3. **RWKV-specific**: State updates may need higher precision

## Potential Speedup

| Component            | Current | With INT8 Act | Improvement |
| -------------------- | ------- | ------------- | ----------- |
| Weight bandwidth     | 1.6 GB  | 1.6 GB        | 0%          |
| Activation bandwidth | 0.5 GB  | 0.25 GB       | 50%         |
| Total                | 2.1 GB  | 1.85 GB       | ~12%        |

**Verdict: Modest improvement, significant complexity. Not the primary bottleneck.**

## Alternative: FP8 Activations

FP8 (E4M3 or E5M2) provides:

-   Same bandwidth as INT8
-   Better dynamic range (no outlier issues)
-   Hardware support on newer GPUs (H100, MI300)

**Problem**: WebGPU doesn't support FP8 natively. Would need emulation.

## Recommendation

**Priority: Medium-Low**

Activation quantization alone won't achieve 2x speedup. The bottleneck is elsewhere.
Consider this only after addressing shader efficiency and memory layout issues.

## References

-   [SmoothQuant](https://arxiv.org/abs/2211.10438) - Activation-aware weight quantization
-   [LLM.int8()](https://arxiv.org/abs/2208.07339) - Mixed-precision decomposition

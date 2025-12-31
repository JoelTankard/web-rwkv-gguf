# Why Q4_K_M Quantization Isn't Providing Expected Speedup

## The Problem

**Current Performance:** 44.63 tok/s with Q4_K_M model  
**Expected Performance:** ~90 tok/s (2x speedup based on kalosm/llama.cpp experience)  
**Memory:** Same as F16 (no reduction observed)

## Root Cause Analysis

### 1. **Activations Are Still F16/F32**

The critical insight: **quantization only affects weight storage, not activations**.

In web-rwkv, the inference pipeline works like this:

```
Input (F16) → MatMul with Q4K weights → Output (F16) → Next layer input (F16)
```

The Q4K weights are dequantized on-the-fly during matmul, but:

-   **Activations between layers remain F16** (2560 dimensions × 2 bytes = 5KB per token per layer)
-   **State tensors remain F32** (for numerical stability)
-   **Intermediate buffers remain F16**

### 2. **Memory Bandwidth Breakdown**

For a 2.9B model with 32 layers, during generation (1 token):

| Component               | F16 Model | Q4K Model | Savings  |
| ----------------------- | --------- | --------- | -------- |
| Weight reads            | ~5.8 GB   | ~1.6 GB   | **3.5x** |
| Activation reads/writes | ~0.5 GB   | ~0.5 GB   | **0x**   |
| State reads/writes      | ~0.2 GB   | ~0.2 GB   | **0x**   |

**Total bandwidth:** F16 = 6.5 GB, Q4K = 2.3 GB → Only **2.8x** reduction, not 3.5x

But wait - the actual speedup should still be significant. So why isn't it?

### 3. **The Real Bottleneck: Compute vs Memory Bound**

llama.cpp on Metal achieves speedup because:

1. **Metal has optimized quantized matmul kernels** using SIMD and matrix hardware
2. **llama.cpp is memory-bandwidth bound** on Apple Silicon during generation

web-rwkv on WebGPU:

1. **WebGPU lacks native quantized matmul support** - we dequantize in shader
2. **Dequantization overhead** adds compute cycles
3. **May be compute-bound** rather than memory-bound due to dequant overhead

### 4. **Shader Efficiency Analysis**

Current Q4K shader (`matmul_vec_q4k_f16.wgsl`) per output element:

-   Read 144 bytes of Q4K data
-   Perform ~256 dequantization operations (bit shifts, masks, multiplies)
-   Perform 256 FMA operations
-   Write 2 bytes output

F16 shader per output element:

-   Read 512 bytes of F16 data
-   Perform 256 FMA operations
-   Write 2 bytes output

**The dequantization overhead may be eating the bandwidth savings.**

## Why llama.cpp/kalosm Works

1. **Metal Performance Shaders (MPS)**: Apple's Metal has highly optimized matrix operations
2. **CPU SIMD**: llama.cpp uses AVX/NEON for efficient quantized ops on CPU
3. **Hybrid execution**: Can offload some work to CPU which has different bottlenecks
4. **Mature optimization**: Years of kernel tuning for specific hardware

## Key Insight

The M2 Max has:

-   **400 GB/s memory bandwidth**
-   **~13.6 TFLOPS FP32 compute**

For generation (batch=1), we need:

-   Read ~1.6 GB weights (Q4K) → 4ms at 400 GB/s
-   Perform ~5.8B FLOPs → 0.4ms at 13.6 TFLOPS

**We should be memory-bound**, meaning Q4K should help. But the dequantization shader may be inefficient enough to make us compute-bound.

## Next Steps

See the other research documents for potential solutions:

1. `01_activation_quantization.md` - Quantize activations too
2. `02_shader_optimization.md` - Optimize dequantization shaders
3. `03_metal_backend.md` - Use Metal directly instead of WebGPU
4. `04_hybrid_precision.md` - Mixed precision strategies
5. `05_architecture_changes.md` - RWKV-specific optimizations
6. `06_memory_layout.md` - Optimize memory access patterns

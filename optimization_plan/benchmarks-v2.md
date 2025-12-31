# Phase 2 Profiling Results

## Date: 2024-12-31

## Hardware

-   GPU: Apple M2 Max (Metal backend)
-   Model: rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf

## Key Finding: Chunk Size Optimization

The token chunk size has a **significant impact** on prefill throughput.

### Chunk Size vs Throughput (256 tokens prefill)

| Chunk Size | Throughput (tok/s) | Improvement vs 64 |
| ---------- | ------------------ | ----------------- |
| 64         | 260                | baseline          |
| 128        | 369                | 1.42x             |
| 256        | 403                | 1.55x             |
| 512        | 408                | 1.57x             |

### Chunk Size vs Throughput (512 tokens prefill)

| Chunk Size | Throughput (tok/s) | Improvement vs 256 |
| ---------- | ------------------ | ------------------ |
| 256        | 412                | baseline           |
| 512        | 466                | 1.13x              |
| 1024       | 466                | 1.13x              |
| 2048       | 465                | 1.13x              |

**Optimal chunk size: 512 tokens** (diminishing returns beyond this)

## Detailed Timing Breakdown

With 128-token prefill, chunk size 128:

| Operation        | Time (ms) | % Total |
| ---------------- | --------- | ------- |
| Inference        | 350.75    | 99.6%   |
| Softmax          | 1.51      | 0.4%    |
| GPU→CPU (to_vec) | 0.01      | 0.0%    |

**Conclusion:** The bottleneck is entirely in the inference forward pass, not in softmax or GPU readback.

## Current Performance vs Target

| Metric             | Current (chunk=512) | Target    | Gap  |
| ------------------ | ------------------- | --------- | ---- |
| Prefill throughput | 466 tok/s           | 700 tok/s | 1.5x |

## Recommendations

1. **Increase default chunk size** from 128 to 512 for prefill operations
2. **Focus optimization on forward pass** - softmax/readback are negligible
3. **Investigate matmul shaders** - the forward pass is dominated by matrix operations

## Phase 2.2: Matrix Multiplication Optimization

### BLOCK_SIZE Testing

Tested different workgroup sizes for `matmul_mat_fp16`:

| BLOCK_SIZE  | Throughput (1024 tokens) | Result    |
| ----------- | ------------------------ | --------- |
| 8 (default) | 488 tok/s                | **Best**  |
| 16          | 447 tok/s                | 8% slower |

**Conclusion:** BLOCK_SIZE=8 is already optimal for Apple M2 Max.

### Forward Pass Analysis

Per-layer matmul operations:

-   **Attention:** w_r, w_k, w_v, w1, w2, a1, a2, g1, g2, v1, v2, w_o = 12 matmuls
-   **FFN:** w_k, w_v = 2 matmuls
-   **Total per layer:** ~14 matmuls
-   **Total per forward pass:** 14 × 32 layers + head = **~450 matmul operations**

### Current Best Performance

| Prefill Length | Chunk Size | Throughput    |
| -------------- | ---------- | ------------- |
| 64             | 64         | 250 tok/s     |
| 128            | 128        | 362 tok/s     |
| 256            | 256        | 408 tok/s     |
| 512            | 512        | 464 tok/s     |
| 1024           | 512        | 471 tok/s     |
| 1024           | 1024       | **488 tok/s** |

### Gap Analysis

| Metric                | Current   | Target    | Gap  |
| --------------------- | --------- | --------- | ---- |
| Prefill (1024 tokens) | 488 tok/s | 700 tok/s | 1.4x |

## Phase 2.3: Token Processing Analysis

### Token Shift Shader

-   Uses `BLOCK_SIZE = 128`
-   Simple linear interpolation between current and previous token
-   Already efficient, no optimization needed

### Time Mix V7 Shader (Core Attention)

-   Uses `BLOCK_SIZE = 32`, `HEAD_SIZE = embed_dim / 4 / num_heads`
-   **Sequential by design** - processes tokens one at a time in a loop
-   State updates create inherent sequential dependency
-   This is fundamental to RWKV architecture, cannot be parallelized

## Summary of Findings

### What's Already Optimized

1. **Matmul BLOCK_SIZE** - 8 is optimal for Apple M2 Max
2. **Command buffer batching** - DEFAULT_SEP=1024 means all layers in one buffer
3. **Native Q4K loading** - inline dequantization avoids repacking overhead

### Key Optimization: Chunk Size

The most impactful optimization is **increasing chunk size**:

| Chunk Size | Throughput | Improvement |
| ---------- | ---------- | ----------- |
| 64         | 250 tok/s  | baseline    |
| 128        | 362 tok/s  | 1.45x       |
| 256        | 408 tok/s  | 1.63x       |
| 512        | 464 tok/s  | 1.86x       |
| 1024       | 488 tok/s  | **1.95x**   |

### Remaining Gap

| Metric                | Current   | Target    | Gap  |
| --------------------- | --------- | --------- | ---- |
| Prefill (1024 tokens) | 488 tok/s | 700 tok/s | 1.4x |

### Recommendations

1. **Increase default chunk size** to 512+ for prefill operations
2. **Consider Metal-specific optimizations** - current shaders are generic WGSL
3. **Profile on other GPUs** - optimal BLOCK_SIZE may vary by hardware

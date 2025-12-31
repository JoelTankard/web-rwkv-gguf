# Recommended Action Plan

## Executive Summary

**Current:** 44.63 tok/s with Q4_K_M  
**Target:** 90+ tok/s (2x speedup)  
**Root cause:** WebGPU lacks native quantized matmul; dequantization overhead negates bandwidth savings

## Why kalosm/llama.cpp is Faster

1. **Native Metal backend** - Uses `simdgroup_matrix` for hardware-accelerated 8x8 matmul
2. **Mature optimization** - Years of kernel tuning
3. **Memory-bound workload** - Weight quantization directly reduces bottleneck
4. **No dequantization overhead** - Hardware handles mixed-precision natively

## Why web-rwkv Q4K Doesn't Help

1. **WebGPU abstraction** - No access to Metal's matrix hardware
2. **Software dequantization** - Adds compute overhead that may exceed bandwidth savings
3. **Suboptimal memory access** - Using <1% of theoretical bandwidth
4. **Compute-bound** - Dequantization makes us compute-limited, not memory-limited

## Prioritized Action Plan

### Phase 1: Profile and Measure (1-2 days)

**Goal:** Confirm bottleneck hypothesis with data

1. Add GPU timestamp profiling to benchmark
2. Run memory vs compute experiments
3. Profile with Metal Instruments
4. Document actual bottleneck

**Success criteria:** Know exactly where time is spent

### Phase 2: Shader Optimization (1 week)

**Goal:** 1.3-1.5x speedup through better shaders

1. Implement subgroup operations for data sharing
2. Optimize dequantization (vectorize, precompute scales)
3. Improve memory coalescing
4. Tune workgroup/tile sizes

**Success criteria:** 60 tok/s

### Phase 3: Memory Layout (1 week)

**Goal:** Additional 1.2-1.4x through better memory access

1. Transpose weight layout for coalesced access
2. Implement double buffering
3. Test tiled memory layout
4. Consider texture memory for weights

**Success criteria:** 70-80 tok/s

### Phase 4: Native Metal Backend (2-4 weeks)

**Goal:** Achieve 2x speedup on macOS

1. Add optional Metal backend using `metal-rs`
2. Implement quantized matmul with `simdgroup_matrix`
3. Keep WebGPU for cross-platform fallback
4. Benchmark against llama.cpp

**Success criteria:** 90+ tok/s on Apple Silicon

## Quick Wins (Can Do Now)

1. **Workgroup size tuning** - Try 64, 128, 256, 512
2. **Enable subgroup-ops feature** - Already partially implemented
3. **Profile current shaders** - Identify hotspots
4. **Compare F16 vs Q4K speed** - Confirm dequant overhead

## Risk Assessment

| Approach             | Risk                           | Mitigation                 |
| -------------------- | ------------------------------ | -------------------------- |
| Shader optimization  | Medium - May hit WebGPU limits | Profile first, know limits |
| Memory layout        | Low - Standard optimization    | Benchmark each change      |
| Native Metal         | High - Platform-specific       | Keep WebGPU fallback       |
| Architecture changes | High - May affect quality      | Test quality metrics       |

## Decision Points

**After Phase 2:**

-   If 60+ tok/s achieved → Continue with Phase 3
-   If <50 tok/s → Skip to Phase 4 (Metal backend)

**After Phase 3:**

-   If 80+ tok/s achieved → May not need Metal backend
-   If <70 tok/s → Proceed with Phase 4

## Resources Needed

-   GPU profiling tools (Xcode Instruments)
-   Metal SDK (for Phase 4)
-   Test hardware (M2 Max available)
-   Benchmark suite (already exists)

## Timeline

| Phase      | Duration  | Target        |
| ---------- | --------- | ------------- |
| 1. Profile | 1-2 days  | Baseline data |
| 2. Shaders | 1 week    | 60 tok/s      |
| 3. Memory  | 1 week    | 80 tok/s      |
| 4. Metal   | 2-4 weeks | 90+ tok/s     |

**Total: 4-6 weeks to 2x speedup**

## Alternative: Use llama.cpp

If native performance is critical and timeline is tight:

-   Use llama.cpp's Metal backend for inference
-   Keep web-rwkv for WebGPU/cross-platform use cases
-   This is what kalosm does (uses llama.cpp under the hood)

## Conclusion

The Q4K quantization is working correctly - weights are stored in 4-bit format. The issue is that WebGPU's software dequantization overhead negates the bandwidth savings. To achieve 2x speedup like llama.cpp:

1. **Short term:** Optimize shaders and memory layout (1.5-2x possible)
2. **Medium term:** Native Metal backend (2x+ achievable)
3. **Long term:** Wait for WebGPU quantized matmul extensions

The most impactful single change would be a native Metal backend for macOS users.

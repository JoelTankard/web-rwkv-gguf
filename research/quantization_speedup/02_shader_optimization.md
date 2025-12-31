# Option 2: Shader Optimization (High Priority)

## Current Problem

The Q4K dequantization shader may be **compute-bound** rather than memory-bound due to:

1. Complex bit manipulation for scale/min extraction
2. Per-element dequantization overhead
3. Suboptimal memory access patterns

## Analysis of Current Shader

From `matmul_vec_q4k_f16.wgsl`:

```wgsl
fn dequant_8_h(qs_packed: u32, d1: f16, m1: f16, d2: f16, m2: f16, ...) -> f16 {
    // 8 bit extractions
    let q0 = f16(qs_packed & 0xFu);
    let q1 = f16((qs_packed >> 8u) & 0xFu);
    // ... 6 more

    // 8 FMA operations
    let w_lo = vec4<f16>(fma(d1, q0, -m1), ...);
    let w_hi = vec4<f16>(fma(d2, q4, -m2), ...);

    // 2 dot products
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}
```

**Per 8 elements**: 8 shifts, 8 masks, 8 int-to-float, 8 FMAs, 2 dot4

## Optimization Strategies

### 1. Vectorized Bit Extraction

Instead of extracting nibbles one by one, use vector operations:

```wgsl
// Current: 8 separate operations
let q0 = f16(qs_packed & 0xFu);
let q1 = f16((qs_packed >> 8u) & 0xFu);

// Better: Extract 4 nibbles at once using unpack
let bytes = unpack4x8unorm(qs_packed);  // Extract as 4 bytes
let lo_nibbles = bytes & vec4(0.0625);  // Mask low nibbles (scaled)
let hi_nibbles = bytes * 16.0 & vec4(0.0625);  // Shift and mask
```

### 2. Precompute Scale Products

Current shader recomputes `d * scale` for each sub-block. Cache these:

```wgsl
// Precompute all 8 scale products at start of super-block
var d_scales: array<f16, 8>;
var m_vals: array<f16, 8>;
for (var i = 0u; i < 8u; i++) {
    let sm = get_scale_min_k4_h(i, scales_u32);
    d_scales[i] = d * sm.x;
    m_vals[i] = dmin * sm.y;
}
```

### 3. Loop Unrolling

WebGPU compilers may not fully unroll loops. Manual unrolling can help:

```wgsl
// Instead of: for (var j64 = 0u; j64 < 4u; j64++)
// Manually unroll the 4 iterations
```

### 4. Reduce Register Pressure

Current shader uses many local variables. Restructure to minimize register usage:

```wgsl
// Accumulate directly instead of storing intermediates
var sum = f16(0.0);
sum = fma(d1, f16(qs & 0xFu), sum - m1 * x.x);
```

### 5. Subgroup Operations

Use subgroup shuffle to share data between threads:

```wgsl
// Share input data across subgroup instead of all threads loading
let shared_input = subgroupBroadcast(my_input, lane_id);
```

### 6. Coalesced Memory Access

Ensure threads in a workgroup access consecutive memory:

```wgsl
// Bad: Each thread reads its own row (strided access)
let data = matrix[row * stride + col];

// Good: Threads read consecutive elements, then shuffle
let data = matrix[workgroup_offset + local_id];
let my_data = subgroupShuffle(data, target_lane);
```

## Benchmark Targets

| Optimization          | Expected Improvement |
| --------------------- | -------------------- |
| Vectorized extraction | 10-20%               |
| Precomputed scales    | 5-10%                |
| Loop unrolling        | 5-15%                |
| Subgroup ops          | 20-40%               |
| Coalesced access      | 10-30%               |

**Combined potential: 50-100% improvement (1.5-2x speedup)**

## Implementation Priority

1. **Subgroup operations** - Biggest potential gain
2. **Coalesced memory access** - Critical for bandwidth
3. **Vectorized extraction** - Reduce ALU pressure
4. **Precomputed scales** - Easy win

## Next Steps

1. Profile current shader with GPU profiler (if available)
2. Implement subgroup-optimized version
3. A/B test each optimization independently
4. Combine winning optimizations

## References

-   [WebGPU Subgroups Proposal](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md)
-   [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
-   [AMD RDNA Optimization Guide](https://gpuopen.com/rdna-performance-guide/)

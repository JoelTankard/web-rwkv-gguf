# Option 6: Memory Layout Optimization

## The Memory Bandwidth Problem

On M2 Max:

-   **Memory bandwidth**: 400 GB/s
-   **Current generation**: 44 tok/s
-   **Model size (Q4K)**: ~1.6 GB weights

Per token, we read: 1.6 GB / 44 = **36 MB**

At 400 GB/s, reading 36 MB takes: 36 / 400000 = **0.09 ms**

But actual token time: 1000 / 44 = **22.7 ms**

**We're only using 0.4% of available bandwidth!**

## Why So Slow?

### 1. Non-Coalesced Memory Access

GPU threads should access consecutive memory addresses.
Current Q4K layout may cause strided access:

```
Thread 0 reads: matrix[row_0 * stride + col]
Thread 1 reads: matrix[row_1 * stride + col]  // Far apart!
```

### 2. Cache Thrashing

Q4K blocks are 144 bytes. If workgroup size doesn't align:

-   Threads may load overlapping cache lines
-   Same data loaded multiple times

### 3. Bank Conflicts in Shared Memory

Shared memory has banks. If multiple threads access same bank:

-   Serialized access
-   Massive slowdown

## Optimization Strategies

### Strategy 1: Transpose Weight Layout

Store weights in column-major order for better access during matmul:

```
// Current: Row-major (good for reading rows)
// matrix[row][col] stored as matrix[row * K + col]

// Proposed: Column-major (good for output-parallel matmul)
// matrix[row][col] stored as matrix[col * M + row]
```

For Q4K, this means reorganizing super-blocks:

```rust
fn transpose_q4k_layout(weights: &[u8], m: usize, k: usize) -> Vec<u8> {
    // Reorganize so consecutive threads read consecutive super-blocks
    let num_sb_k = k / 256;
    let num_sb_m = m;

    let mut transposed = vec![0u8; weights.len()];
    for sb_m in 0..num_sb_m {
        for sb_k in 0..num_sb_k {
            let src_offset = (sb_m * num_sb_k + sb_k) * 144;
            let dst_offset = (sb_k * num_sb_m + sb_m) * 144;
            transposed[dst_offset..dst_offset+144]
                .copy_from_slice(&weights[src_offset..src_offset+144]);
        }
    }
    transposed
}
```

### Strategy 2: Tiled Memory Layout

Organize weights into tiles that fit in shared memory:

```
// Tile size: 32x256 elements = 32 rows × 1 super-block
// Each tile is 32 × 144 bytes = 4.5 KB (fits in shared memory)

Tile layout:
[Tile(0,0)] [Tile(0,1)] [Tile(0,2)] ...
[Tile(1,0)] [Tile(1,1)] [Tile(1,2)] ...
```

### Strategy 3: Interleaved Quantization Data

Current Q4K layout:

```
[d, dmin, scales, qs_0, qs_1, ..., qs_127]
```

Proposed interleaved layout for better vectorization:

```
[d, dmin, scales,
 qs_0_lo, qs_0_hi, qs_1_lo, qs_1_hi, ...]
```

This allows loading low/high nibbles in parallel.

### Strategy 4: Prefetch-Friendly Access

Structure access patterns to enable hardware prefetch:

```wgsl
// Bad: Random access pattern
for (var i = 0u; i < n; i++) {
    let idx = indices[i];  // Unpredictable
    sum += data[idx];
}

// Good: Sequential access with prefetch hint
for (var i = 0u; i < n; i += 4u) {
    // Load 4 consecutive elements
    let d0 = data[base + i];
    let d1 = data[base + i + 1u];
    let d2 = data[base + i + 2u];
    let d3 = data[base + i + 3u];
    sum += d0 + d1 + d2 + d3;
}
```

### Strategy 5: Double Buffering

Overlap memory loads with computation:

```wgsl
var<workgroup> buffer_a: array<f32, 256>;
var<workgroup> buffer_b: array<f32, 256>;

// Load first chunk into buffer_a
load_chunk(buffer_a, 0);
workgroupBarrier();

for (var chunk = 1u; chunk < num_chunks; chunk++) {
    // Load next chunk into buffer_b while computing on buffer_a
    if (chunk % 2u == 1u) {
        load_chunk_async(buffer_b, chunk);
        compute(buffer_a);
    } else {
        load_chunk_async(buffer_a, chunk);
        compute(buffer_b);
    }
    workgroupBarrier();
}
```

### Strategy 6: Texture Memory for Weights

Use texture memory instead of buffer for read-only weights:

-   Better cache behavior
-   Hardware filtering (not needed but free)
-   Different memory path (may reduce contention)

```rust
// Create texture from weight data
let texture = device.create_texture(&TextureDescriptor {
    size: Extent3d { width: k/4, height: m, depth_or_array_layers: 1 },
    format: TextureFormat::Rgba8Uint,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    ..
});
```

## Benchmark Methodology

To identify the actual bottleneck:

```rust
// Test 1: Memory bandwidth (read-only)
fn benchmark_read_bandwidth() {
    // Just read all weights, no compute
}

// Test 2: Compute throughput (no memory)
fn benchmark_compute() {
    // Use data already in registers/shared memory
}

// Test 3: Combined (current)
fn benchmark_combined() {
    // Full matmul
}
```

If Test 1 >> Test 3 time: Memory-bound
If Test 2 >> Test 3 time: Compute-bound
If Test 1 + Test 2 ≈ Test 3: Balanced

## Expected Improvements

| Strategy          | Expected Speedup | Implementation Effort |
| ----------------- | ---------------- | --------------------- |
| Transposed layout | 20-50%           | Medium                |
| Tiled layout      | 30-60%           | High                  |
| Interleaved quant | 10-20%           | Medium                |
| Prefetch-friendly | 10-30%           | Low                   |
| Double buffering  | 20-40%           | Medium                |
| Texture memory    | 10-30%           | Medium                |

## Recommendation

**Priority: High**

Memory layout is likely the primary bottleneck. The fact that we're using <1% of theoretical bandwidth suggests severe inefficiency.

Implementation order:

1. **Profile** to confirm memory-bound
2. **Transposed layout** - Biggest potential gain
3. **Double buffering** - Overlap load/compute
4. **Tiled layout** - If still memory-bound

## Implementation Results (2026-01-01)

### Tested on Apple M2 Max

| Strategy                       | Generation (tok/s) | Change |
| ------------------------------ | ------------------ | ------ |
| Baseline                       | 43.94              | -      |
| Transposed Layout              | 43.83              | -0.25% |
| Double Buffering               | 43.70              | -0.55% |
| Reduced Barriers (2 SB batch)  | 43.87              | -0.16% |
| Final (Transposed + v2 shader) | 43.93              | -0.02% |

### Key Findings

1. **Memory layout optimizations provided no measurable improvement** on Apple M2 Max
2. **Quality verified** - all implementations produce identical output (hash: `3895ad4add71cff0`)
3. **Apple Silicon's unified memory architecture** may already handle memory access patterns efficiently
4. **The bottleneck is likely elsewhere** - possibly compute-bound on dequantization math

### What Was Implemented

1. **Transposed Q4K Layout** (`transpose_q4k_layout` in `loader.rs` and `lazy.rs`)

    - Reorganizes super-blocks from `block[row][sb_k]` to `block[sb_k][row]`
    - Ensures consecutive threads read consecutive memory addresses
    - Implemented in both eager and lazy loading paths

2. **Updated Shaders** (`matmul_vec_q4k_v2.wgsl`, `matmul_mat_q4k_opt.wgsl`)
    - Modified block index calculation for transposed layout
    - `block_u32_base = (sb * m + row) * Q4K_BLOCK_U32`

### Why No Improvement?

The document's analysis assumed we were using <1% of memory bandwidth, but this may not be accurate for Apple Silicon:

1. **Unified Memory** - No discrete GPU memory, so memory access patterns differ from NVIDIA/AMD
2. **Hardware Prefetching** - Apple's GPU may already efficiently prefetch non-coalesced access
3. **Compute-Bound** - The dequantization math (scale extraction, nibble unpacking, FMA) may dominate

### Recommendations

-   **Keep transposed layout** - theoretically correct, no performance regression
-   **Focus optimization efforts elsewhere** - shader F16 arithmetic, subgroup operations
-   **Profile with Metal System Trace** to identify actual bottleneck

## References

-   [NVIDIA CUDA Memory Optimization](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
-   [AMD CDNA Memory Hierarchy](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
-   [Apple Metal Best Practices](https://developer.apple.com/documentation/metal/gpu_programming_guide/about_gpu_memory)

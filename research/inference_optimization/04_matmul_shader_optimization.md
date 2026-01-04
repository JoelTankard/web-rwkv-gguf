# Matrix Multiplication Shader Optimization

## Current Implementation

Location: `src/shaders/matmul_*.wgsl`, `src/tensor/ops.rs:740-1500`

Current matmul variants:

-   `matmul_vec_fp16` / `matmul_mat_fp16` - FP16 weights
-   `matmul_vec_int8` / `matmul_mat_int8` - INT8 quantized
-   `matmul_vec_nf4` / `matmul_mat_nf4` - NF4 quantized
-   `matmul_vec_q8_0` / `matmul_mat_q8_0` - GGML Q8_0 format

Block sizes:

-   Vector variants: `BLOCK_SIZE = 128`
-   Matrix variants: `BLOCK_SIZE = 8`

## Problem

1. **Fixed block sizes**: Not tuned for different GPU architectures
2. **No shared memory tiling**: Matrix variants don't use workgroup shared memory
3. **Suboptimal memory access patterns**: May not be coalesced for all shapes
4. **No vectorized loads**: Could use `vec4` loads more aggressively

## Optimization Ideas

### Idea 1: Tiled Matrix Multiplication with Shared Memory

Current `matmul_mat_fp16` doesn't use shared memory tiling:

```wgsl
// Current approach: each thread computes one output element
// Loads same matrix rows/cols multiple times from global memory

// Optimized: tile-based with shared memory
const TILE_SIZE: u32 = 16;
var<workgroup> tile_a: array<vec4<f16>, TILE_SIZE * TILE_SIZE / 4>;
var<workgroup> tile_b: array<vec4<f16>, TILE_SIZE * TILE_SIZE / 4>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = vec4<f32>(0.0);

    for (var k = 0u; k < K; k += TILE_SIZE) {
        // Cooperative load into shared memory
        tile_a[lid.y * TILE_SIZE/4 + lid.x] = load_matrix_a(k, lid);
        tile_b[lid.x * TILE_SIZE/4 + lid.y] = load_matrix_b(k, lid);

        workgroupBarrier();

        // Compute from shared memory (much faster)
        for (var i = 0u; i < TILE_SIZE; i++) {
            acc += tile_a[lid.y][i] * tile_b[i][lid.x];
        }

        workgroupBarrier();
    }

    store_output(acc);
}
```

**Expected benefit**: 2-4x speedup for batch inference (turbo mode)

### Idea 2: Vectorized Quantized Matmul

Current NF4/INT8 matmuls dequantize one element at a time. Use vectorized dequant:

```wgsl
// Current: scalar dequantization
let q = matrix[idx];
let lo = q & 0xFu;
let hi = q >> 4u;
let v0 = quant[lo] * absmax;
let v1 = quant[hi] * absmax;

// Optimized: vectorized dequantization
fn dequant_nf4_vec8(packed: vec4<u32>, absmax: f16) -> array<vec4<f16>, 2> {
    // Unpack 8 values at once using bit manipulation
    let lo = packed & vec4<u32>(0x0F0F0F0Fu);
    let hi = (packed >> 4u) & vec4<u32>(0x0F0F0F0Fu);

    // Gather from quant table (could use texture for better cache)
    // ...
}
```

### Idea 3: Warp/Subgroup Reduction

Use subgroup operations for the reduction phase:

```wgsl
#ifdef SUBGROUP_SIZE_32_32
@compute @workgroup_size(128)
fn main() {
    var sum = vec4<f32>(0.0);

    // Each thread accumulates partial sum
    for (var i = 0u; i < K; i += 128u) {
        sum += compute_partial(i);
    }

    // Subgroup reduction (much faster than shared memory)
    sum = subgroupAdd(sum);

    // Only first thread in subgroup writes
    if (subgroupElect()) {
        atomicAdd(&output[row], sum);
    }
}
#endif
```

**Note**: Already have `subgroup-ops` feature, but may not be fully utilized.

### Idea 4: Texture-Based Weight Access

For quantized weights, use texture sampling for better cache behavior:

```wgsl
@group(0) @binding(3) var weight_texture: texture_2d<u32>;
@group(0) @binding(4) var weight_sampler: sampler;

fn load_weight(row: u32, col: u32) -> vec4<f16> {
    // Texture cache is optimized for 2D spatial locality
    let texel = textureLoad(weight_texture, vec2<i32>(col/4, row), 0);
    return dequant(texel);
}
```

**Benefit**: Better cache utilization for non-sequential access patterns

### Idea 5: Async Copy / Prefetching

WebGPU doesn't have explicit async copy, but can simulate with double-buffering:

```wgsl
var<workgroup> buffer_a: array<vec4<f16>, 256>;
var<workgroup> buffer_b: array<vec4<f16>, 256>;

fn main() {
    var current = &buffer_a;
    var next = &buffer_b;

    // Prefetch first tile
    load_tile(current, 0);
    workgroupBarrier();

    for (var k = 0u; k < K; k += TILE_SIZE) {
        // Start loading next tile while computing current
        if (k + TILE_SIZE < K) {
            load_tile(next, k + TILE_SIZE);
        }

        // Compute from current tile
        compute_tile(current);

        workgroupBarrier();

        // Swap buffers
        let tmp = current;
        current = next;
        next = tmp;
    }
}
```

### Idea 6: Specialized Single-Token Kernel

For generation (single token), use a different kernel optimized for `N=1`:

```wgsl
// matmul_vec_fp16_single.wgsl
// Optimized for K large, N=1
// Each workgroup computes multiple output rows
// Uses full workgroup for reduction along K dimension

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let row_base = wg.x * 4;  // 4 rows per workgroup
    var acc = array<vec4<f32>, 4>();

    // All 256 threads cooperatively reduce along K
    for (var k = local_id; k < K; k += 256u) {
        let input_val = input[k];
        for (var r = 0u; r < 4u; r++) {
            acc[r] += matrix[row_base + r][k] * input_val;
        }
    }

    // Workgroup reduction
    acc = workgroup_reduce(acc);

    // Single thread writes output
    if (local_id == 0u) {
        for (var r = 0u; r < 4u; r++) {
            output[row_base + r] = acc[r];
        }
    }
}
```

## Estimated Impact

| Optimization        | Single Token | Batch (64+) | Complexity |
| ------------------- | ------------ | ----------- | ---------- |
| Tiled Shared Memory | +5%          | +50-100%    | Medium     |
| Vectorized Dequant  | +10-20%      | +10-20%     | Low        |
| Subgroup Reduction  | +10-15%      | +5%         | Low        |
| Texture Weights     | +5-10%       | +10-20%     | Medium     |
| Double Buffer       | +5%          | +10-15%     | Medium     |
| Single-Token Kernel | +20-30%      | N/A         | Medium     |

## Recommended Experiment

1. **First**: Profile current matmul to identify bottleneck (compute vs memory)
2. **If memory-bound**: Implement tiled shared memory for `matmul_mat_*`
3. **If compute-bound**: Implement vectorized dequantization

## Profiling Command

```bash
# Enable GPU profiling
WGPU_PROFILER=1 cargo run --release --example bench -- --model $MODEL --tokens 100

# Or use Metal GPU profiler on macOS
xcrun metal-profiler capture -- cargo run --release --example bench
```

## Files to Modify

-   `src/shaders/matmul_mat_fp16.wgsl` - Add tiling
-   `src/shaders/matmul_vec_fp16.wgsl` - Single-token optimization
-   `src/shaders/matmul_*_nf4.wgsl` - Vectorized dequant
-   `src/tensor/ops.rs` - Dispatch logic for different kernels

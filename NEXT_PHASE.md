# Next Phase: Q4K Shader Optimization

## Phase 7 Status: ✅ Optimizations Complete

Native quantized matmul shaders are implemented with vectorized dequantization and shared memory tiling.

### Baseline Results (M2 Max, 0.1B model):

| Format      | File Size | Load Time | RAM Δ    | Prefill  | Generation |
| ----------- | --------- | --------- | -------- | -------- | ---------- |
| SafeTensors | 364.5 MB  | 1219 ms   | 889.0 MB | 2723 t/s | 168.7 t/s  |
| GGUF Q4_K   | 126.4 MB  | 776 ms    | 201.4 MB | 2829 t/s | 169.5 t/s  |

**Achieved:**

-   65% smaller file size
-   36% faster load time
-   77% less RAM
-   +4% prefill, +0.5% generation (baseline)

---

## Completed Optimizations

### 1. ✅ Vectorized Dequantization

Implemented in all quantized matmul shaders:

-   Process 8 elements per iteration using `vec4<f32>` operations
-   Use `fma()` for fused multiply-add operations
-   Reduced loop overhead and improved instruction-level parallelism

### 2. ✅ Shared Memory Tiling

Added `var<workgroup> input_cache` to vector shaders:

-   Load input tile into shared memory once per workgroup
-   All threads share the cached data
-   Reduces global memory bandwidth significantly

### 3. ✅ Extended Format Support

Created native matmul shaders for additional quantization formats:

| Format | Block Size | Bytes/Block | Shaders                                        |
| ------ | ---------- | ----------- | ---------------------------------------------- |
| Q4_K   | 256        | 144         | `matmul_vec_q4k.wgsl`, `matmul_mat_q4k.wgsl`   |
| Q5_K   | 256        | 176         | `matmul_vec_q5k.wgsl`, `matmul_mat_q5k.wgsl`   |
| Q6_K   | 256        | 210         | `matmul_vec_q6k.wgsl`, `matmul_mat_q6k.wgsl`   |
| Q8_0   | 32         | 34          | `matmul_vec_q8_0.wgsl`, `matmul_mat_q8_0.wgsl` |

### Components Updated

-   `src/tensor/matrix.rs` - Added `Matrix::Q5K`, `Matrix::Q6K`, `Matrix::Q8_0` variants
-   `src/tensor/ops.rs` - Added `TensorOp::matmul_vec_q5k/q6k/q8_0` and `matmul_mat_q5k/q6k/q8_0`
-   `src/runtime/loader.rs` - Native loading paths for Q5K, Q6K, Q8_0 formats

---

## Next Steps

### 1. Benchmark Optimized Shaders

Run performance tests to measure improvement from vectorization and shared memory:

```bash
cargo run --release --example chat -- --model path/to/model.gguf
```

### 2. Subgroup Operations (Future)

Consider using WebGPU subgroup operations for further optimization:

-   `subgroupAdd()` for reduction operations
-   `subgroupBroadcast()` for sharing data within subgroups
-   Requires `enable subgroups;` directive and feature detection

---

## Reference Implementations

### llama.cpp Vulkan Backend

Key files in `ggml/src/ggml-vulkan/vulkan-shaders/`:

-   `mul_mat_vec_q4_k.comp` - Q4_K matrix-vector multiply
-   `mul_mat_vec_base.glsl` - Shared infrastructure
-   `dequant_funcs.glsl` - Dequantization helpers

### MLC-LLM / WebLLM

Uses TVM to compile optimized WGSL kernels. Achieves ~85% of native Metal performance on M3 Max.

---

## Previous Phases (Completed)

<details>
<summary>Phase 4: K-Quant Dequantization ✅</summary>

---

## Phase 4: K-Quant Dequantization

### Background

K-quants use a more sophisticated "super-block" structure with nested quantization for better quality at the same bit rate. They're the recommended quant types in llama.cpp for most use cases.

### K-Quant Block Structures

#### Q4_K (256 elements per super-block)

```c
// From llama.cpp ggml-common.h
typedef struct {
    half d;                      // super-block scale (2 bytes)
    half dmin;                   // super-block min (2 bytes)
    uint8_t scales[12];          // 4-bit scales for 8 sub-blocks (12 bytes)
    uint8_t qs[128];             // 4-bit quantized values (128 bytes)
} block_q4_K;                    // Total: 144 bytes for 256 elements
```

**Dequantization formula:**

```
For each sub-block i (0..7):
    scale_i = (scales[i] & 0x3F) * d
    min_i = (scales[i] >> 6 | scales[i+8] << 2) * dmin
    For each element j in sub-block:
        value = (qs[...] & 0xF) * scale_i - min_i
```

#### Q5_K (256 elements per super-block)

```c
typedef struct {
    half d;                      // super-block scale
    half dmin;                   // super-block min
    uint8_t scales[12];          // scales and mins
    uint8_t qh[32];              // high bits
    uint8_t qs[128];             // low 4 bits
} block_q5_K;                    // Total: 176 bytes for 256 elements
```

#### Q6_K (256 elements per super-block)

```c
typedef struct {
    uint8_t ql[128];             // low 4 bits
    uint8_t qh[64];              // high 2 bits
    int8_t scales[16];           // scales
    half d;                      // super-block scale
} block_q6_K;                    // Total: 210 bytes for 256 elements
```

### Implementation Steps

1. **Add dequantization functions to `gguf.rs`:**

```rust
fn dequantize_q4_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 144;

    let num_blocks = num_elements / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse header
        let d = f16::from_le_bytes([block[0], block[1]]);
        let dmin = f16::from_le_bytes([block[2], block[3]]);
        let scales = &block[4..16];
        let qs = &block[16..144];

        // Dequantize 8 sub-blocks of 32 elements each
        for sub_block in 0..8 {
            let scale = /* extract from scales */;
            let min = /* extract from scales */;

            for j in 0..32 {
                let q = /* extract 4-bit value from qs */;
                let val = (q as f32) * scale - min;
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
        }
    }

    output
}
```

1. **Update `GgmlType::is_quantized()` to include K-quants** (already done)

2. **Update `tensor()` match arm:**

```rust
GgmlType::Q4K => dequantize_q4_k_to_f16(data, num_elements),
GgmlType::Q5K => dequantize_q5_k_to_f16(data, num_elements),
GgmlType::Q6K => dequantize_q6_k_to_f16(data, num_elements),
```

1. **Add unit tests** with known test vectors

### Reference Implementation

See llama.cpp's `ggml-quants.c` for reference dequantization:

-   `dequantize_row_q4_K()`
-   `dequantize_row_q5_K()`
-   `dequantize_row_q6_K()`

---

## Phase 5: Direct GPU Quantized Loading

### Goal

Bypass F16 staging entirely by loading GGUF quantized data directly into web-rwkv's quantized matrix formats.

### Challenge: Block Size Mismatch

| Format        | Block Size | Scale Storage          |
| ------------- | ---------- | ---------------------- |
| GGUF Q8_0     | 32         | 1 f16 per 32 elements  |
| web-rwkv Int8 | 128        | 1 f16 per 128 elements |
| GGUF Q4_0     | 32         | 1 f16 per 32 elements  |
| web-rwkv Fp4  | 64         | 1 f16 per 64 elements  |

### Option A: CPU Repacking (Recommended First Step)

Repack GGUF blocks into web-rwkv format on CPU. Still avoids F16 staging.

```rust
fn repack_q8_0_to_int8(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<u8>) {
    // GGUF Q8_0: 32 elements per block, scale per block
    // web-rwkv Int8: 128 elements per block, absmax per block

    const GGUF_BLOCK: usize = 32;
    const RWKV_BLOCK: usize = 128;

    let mut weights = Vec::with_capacity(num_elements);
    let mut absmax = Vec::with_capacity(num_elements / RWKV_BLOCK * 2);

    // Process 4 GGUF blocks at a time to make 1 web-rwkv block
    for chunk in data.chunks(GGUF_BLOCK * 4) {
        // Find max scale across 4 GGUF blocks
        let mut max_scale = 0.0f32;
        for i in 0..4 {
            let scale = f16::from_le_bytes([chunk[i*34], chunk[i*34+1]]);
            max_scale = max_scale.max(scale.to_f32().abs());
        }

        // Rescale all 128 values to use the combined scale
        for i in 0..4 {
            let block_scale = f16::from_le_bytes([chunk[i*34], chunk[i*34+1]]).to_f32();
            for j in 0..32 {
                let q = chunk[i*34 + 2 + j] as i8;
                let rescaled = ((q as f32) * block_scale / max_scale) as i8;
                weights.push(rescaled as u8);
            }
        }

        absmax.extend_from_slice(&f16::from_f32(max_scale).to_le_bytes());
    }

    (weights, absmax)
}
```

### Option B: New GPU Shader

Create a new matmul shader that understands GGUF Q8_0 block format directly.

**Pros:**

-   Zero CPU overhead
-   Maximum performance

**Cons:**

-   Significant shader work
-   Need to maintain two code paths

### Option C: Modify web-rwkv Block Sizes

Change `INT8_BLOCK_SIZE` from 128 to 32 to match GGUF.

**Pros:**

-   Direct compatibility

**Cons:**

-   May affect existing quantization quality
-   Breaking change for serialized models

### Implementation Plan

1. **Start with Option A** - CPU repacking is safest and still provides memory benefits
2. **Modify `Loader::load_matrix`** to detect GGUF quantized tensors
3. **Add new loading path** that bypasses `tensor_f16_from_reader`

```rust
// In loader.rs
fn load_matrix_from_gguf_quantized(
    &self,
    context: &Context,
    reader: &impl Reader,
    name: &str,
    quant: Quant,
) -> Result<Matrix, anyhow::Error> {
    // Check if tensor is already quantized in GGUF
    if let Some(gguf_type) = reader.gguf_type(name) {
        match (gguf_type, quant) {
            (GgmlType::Q8_0, Quant::Int8) => {
                // Repack Q8_0 -> Int8 directly
                let (weights, absmax) = repack_q8_0_to_int8(...);
                return Ok(Matrix::Int8 { ... });
            }
            (GgmlType::Q4_0, Quant::NF4) => {
                // Repack Q4_0 -> Fp4 directly
                let (weights, absmax) = repack_q4_0_to_fp4(...);
                return Ok(Matrix::Fp4 { ... });
            }
            _ => {} // Fall through to F16 path
        }
    }

    // Existing F16 path
    self.load_matrix(context, reader, name, quant)
}
```

### Memory Savings Estimate

For a 7B model:

-   Current: Load 14GB F16 → Quantize to 7GB Int8 → Peak: 21GB
-   Direct: Load 7GB Q8_0 → Repack to 7GB Int8 → Peak: 14GB
-   **Savings: ~7GB peak VRAM**

---

## Testing Strategy

### K-Quant Tests

1. Convert test model to Q4_K, Q5_K, Q6_K using llama.cpp's `quantize`
2. Compare output logits against F16 reference
3. Verify perplexity is within expected range

### Direct Loading Tests

1. Compare `Matrix::Int8` created via:
    - F16 → GPU quantize
    - Q8_0 → CPU repack
2. Verify identical inference results
3. Benchmark memory usage with `bench_format`

---

## Files to Modify

| File                                   | Changes                                |
| -------------------------------------- | -------------------------------------- |
| `src/runtime/gguf.rs`                  | Add K-quant dequantization functions   |
| `src/runtime/loader.rs`                | Add direct quantized loading path      |
| `src/tensor/matrix.rs`                 | Possibly add GGUF-native block format  |
| `examples/bench_format.rs`             | Add memory tracking for direct loading |
| `assets/scripts/convert_hf_to_gguf.py` | Add K-quant output support             |

---

## Priority Order

1. **Q4_K dequantization** - Most commonly used K-quant
2. **Q6_K dequantization** - Best quality K-quant
3. **Q5_K dequantization** - Middle ground
4. **Direct Q8_0 → Int8 loading** - Biggest memory win
5. **Direct Q4_0 → Fp4 loading** - Secondary memory win
6. **Q3_K, Q2_K dequantization** - Lower priority, extreme compression

</details>

---

## Phase 7: Native Quantized Matmul Shaders

### Problem Statement

Current benchmark (M2 Max, 0.1B model, Q4_K_M):

| Format      | File Size | RAM Δ    | Generation |
| ----------- | --------- | -------- | ---------- |
| SafeTensors | 364.5 MB  | 912.2 MB | 180.3 t/s  |
| GGUF Q4_K_M | 126.4 MB  | 157.5 MB | 180.8 t/s  |

**Only +0.3% faster** despite 65% smaller files and 83% less RAM. This is because Q4_K is dequantized to F16 at load time - inference runs identical F16 matmul shaders.

### Goal

Implement matrix multiplication shaders that operate directly on Q4_K packed data, dequantizing inline during the multiply-accumulate loop. Expected benefits:

-   **Memory bandwidth reduction**: Read 4 bits instead of 16 bits per weight
-   **Target speedup**: 1.5-2x for memory-bound operations (generation)
-   **Prefill may see less improvement** (already compute-bound)

### Q4_K Block Structure (144 bytes → 256 elements)

```c
struct block_q4_K {
    f16 d;              // super-block scale (2 bytes)
    f16 dmin;           // super-block min (2 bytes)
    u8 scales[12];      // 6-bit scales for 8 sub-blocks (12 bytes)
    u8 qs[128];         // 4-bit quantized values, packed (128 bytes)
};  // 144 bytes total for 256 elements = 4.5 bits/element
```

### WGSL Shader Design

Port from llama.cpp `mul_mat_vec_q4_k.comp`. Key adaptations for WGSL:

```wgsl
// Q4_K dequantization inline in matmul
fn dequant_q4k_block(block_idx: u32, element_idx: u32) -> f32 {
    let block_offset = block_idx * 144u;

    // Read super-block scales
    let d = unpack_f16(q4k_data[block_offset]);
    let dmin = unpack_f16(q4k_data[block_offset + 2u]);

    // Determine sub-block (0-7) and position within
    let sub_block = element_idx / 32u;
    let sub_pos = element_idx % 32u;

    // Extract 6-bit scale and min from packed scales[12]
    let scale = extract_scale(block_offset + 4u, sub_block);
    let min_val = extract_min(block_offset + 4u, sub_block);

    // Extract 4-bit quantized value from qs[128]
    let qs_byte_idx = (element_idx / 2u);
    let qs_byte = q4k_data[block_offset + 16u + qs_byte_idx];
    let q4 = select(qs_byte & 0xFu, qs_byte >> 4u, (element_idx & 1u) == 1u);

    // Dequantize: value = q4 * (d * scale) - (dmin * min_val)
    return f32(q4) * (d * f32(scale)) - (dmin * f32(min_val));
}

@compute @workgroup_size(256)
fn matmul_vec_q4k(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.x;
    var sum = 0.0f;

    // Process 256-element super-blocks
    for (var block = 0u; block < num_blocks; block++) {
        for (var i = 0u; i < 256u; i++) {
            let w = dequant_q4k_block(row * num_blocks + block, i);
            let x = input[block * 256u + i];
            sum += w * x;
        }
    }

    output[row] = sum;
}
```

### Implementation Steps

1. **Create `Matrix::Q4K` variant**

    ```rust
    pub enum Matrix {
        Fp16(TensorGpu<f16, ReadWrite>),
        Int8 { w: TensorGpu<u8, ReadWrite>, m: TensorGpu<f16, ReadWrite> },
        Fp4 { w: TensorGpu<u8, ReadWrite>, q: TensorGpu<f32, Uniform>, m: TensorGpu<f16, ReadWrite> },
        Q4K { data: TensorGpu<u8, ReadWrite>, num_blocks: usize },  // NEW
    }
    ```

2. **Add `matmul_mat_q4k.wgsl` shader**

    - Port llama.cpp GLSL → WGSL
    - Handle WGSL limitations (no `unpack8`, manual bit manipulation)
    - Use workgroup shared memory for input vector caching

3. **Implement `TensorOp::matmul_mat_q4k`**

    ```rust
    pub fn matmul_mat_q4k<F: Float>(
        matrix: &TensorGpu<u8, ReadWrite>,  // Raw Q4_K blocks
        input: impl Into<TensorGpuView<'_, F>>,
        output: impl Into<TensorGpuView<'_, F>>,
        act: Activation,
    ) -> Result<Self, TensorError>
    ```

4. **Update loader to create `Matrix::Q4K` directly**

    ```rust
    (GgmlType::Q4K, Quant::None) => {
        // Load raw Q4_K blocks without dequantization
        let data: TensorGpu<u8, ReadWrite> = ...;
        Ok(Matrix::Q4K { data, num_blocks })
    }
    ```

5. **Benchmark and optimize**
    - Compare against F16 baseline
    - Profile memory bandwidth utilization
    - Tune workgroup sizes

### GLSL → WGSL Porting Notes

| GLSL                   | WGSL Equivalent                                |
| ---------------------- | ---------------------------------------------- |
| `unpack8(u32)`         | Manual: `vec4(x & 0xFF, (x >> 8) & 0xFF, ...)` |
| `fma(a, b, c)`         | `fma(a, b, c)` (native)                        |
| `[[unroll]]`           | `@unroll` or manual unroll                     |
| `shared`               | `var<workgroup>`                               |
| `gl_LocalInvocationID` | `@builtin(local_invocation_id)`                |
| `float16_t`            | `f16` (requires `f16` extension)               |

### Files to Create/Modify

| File                              | Changes                        |
| --------------------------------- | ------------------------------ |
| `src/shaders/matmul_mat_q4k.wgsl` | **NEW** - Q4_K matmul shader   |
| `src/tensor/matrix.rs`            | Add `Matrix::Q4K` variant      |
| `src/tensor/ops.rs`               | Add `matmul_mat_q4k` operation |
| `src/runtime/loader.rs`           | Direct Q4_K loading path       |
| `src/runtime/infer/*.rs`          | Use Q4K matrices in inference  |

### Expected Performance

Based on llama.cpp Vulkan benchmarks and WebLLM results:

-   **Generation (batch=1)**: 1.5-2x speedup (memory-bound)
-   **Prefill (batch=256)**: 1.0-1.3x speedup (compute-bound)
-   **Memory**: Same as current GGUF path (already optimal)

### Risks & Mitigations

1. **WGSL limitations** - No native `unpack8`, may need manual bit ops

    - Mitigation: Pre-compute lookup tables in uniform buffer

2. **Workgroup size constraints** - WebGPU limits vary by device

    - Mitigation: Use conservative 256 threads, tune per-device

3. **f16 extension availability** - Not all WebGPU implementations support
    - Mitigation: Provide f32 fallback path

### Success Criteria

-   [ ] Q4_K matmul shader compiles and runs on Metal/Vulkan
-   [ ] Generation speed ≥1.5x faster than F16 path
-   [ ] Inference quality identical to dequantized path
-   [ ] No regression on prefill performance

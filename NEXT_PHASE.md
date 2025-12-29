# Next Phase: K-Quant Support & Direct GPU Loading

## Overview

This document provides implementation details for the next two phases of GGUF support:

1. **Phase 4**: K-Quant dequantization (Q4_K, Q5_K, Q6_K, etc.)
2. **Phase 5**: Direct GPU quantized loading (bypass F16 staging)

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

2. **Update `GgmlType::is_quantized()` to include K-quants** (already done)

3. **Update `tensor()` match arm:**

```rust
GgmlType::Q4K => dequantize_q4_k_to_f16(data, num_elements),
GgmlType::Q5K => dequantize_q5_k_to_f16(data, num_elements),
GgmlType::Q6K => dequantize_q6_k_to_f16(data, num_elements),
```

4. **Add unit tests** with known test vectors

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

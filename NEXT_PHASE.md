# Next Phase: Ternary Quantization Support (TQ1_0 / TQ2_0)

## Overview

Add support for GGUF ternary quantization formats TQ1_0 and TQ2_0, which offer extreme compression (~1.6 and ~2.1 bits per weight respectively).

### Available Test Models

```
assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-f16.gguf      (5656 MB - baseline)
assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf   (1788 MB - 68% smaller)
assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-TQ2_0.gguf    (1056 MB - 81% smaller)
assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-TQ1_0.gguf    (944 MB  - 83% smaller)
```

---

## Phase 8: TQ1_0 / TQ2_0 Dequantization

### TQ1_0 Block Structure (~1.6 bits per weight)

Ternary quantization with values {-1, 0, +1}. Uses base-3 encoding for maximum compression.

```c
// From llama.cpp ggml-common.h
#define QK_K 256

typedef struct {
    uint8_t qs[(QK_K - 4*QK_K/64) / 5];  // 49 bytes - packed ternary values (base-3)
    uint8_t qh[QK_K/64];                  // 4 bytes - high bits for edge cases
    half d;                               // 2 bytes - scale
} block_tq1_0;  // Total: 55 bytes for 256 elements ≈ 1.72 bits/weight
```

**Encoding:** 5 ternary values packed into 1 byte using base-3 (3^5 = 243 < 256)

**Dequantization:**

```
For each group of 5 values:
    byte_val = qs[i]
    for j in 0..5:
        trit = byte_val % 3  // 0, 1, or 2
        byte_val /= 3
        value = (trit - 1) * d  // Maps to -d, 0, +d
```

### TQ2_0 Block Structure (~2.1 bits per weight)

Ternary quantization with 4 possible values per 2-bit slot, plus scale.

```c
typedef struct {
    uint8_t qs[QK_K/4];  // 64 bytes - 2 bits per weight
    half d;              // 2 bytes - scale
} block_tq2_0;  // Total: 66 bytes for 256 elements ≈ 2.06 bits/weight
```

**Encoding:** 4 weights packed per byte (2 bits each)

**Dequantization:**

```
For each byte:
    for j in 0..4:
        q2 = (byte >> (j*2)) & 0x3  // 0, 1, 2, or 3
        value = (q2 - 1.5) * d      // Maps to -1.5d, -0.5d, +0.5d, +1.5d
```

---

## Implementation Plan

### Step 1: Add GGML Type Definitions

Update `src/runtime/gguf.rs`:

```rust
pub enum GgmlType {
    // ... existing types ...
    TQ1_0 = 16,  // Ternary 1.6 bpw
    TQ2_0 = 17,  // Ternary 2.1 bpw
}

impl GgmlType {
    pub fn block_size(&self) -> usize {
        match self {
            // ... existing ...
            Self::TQ1_0 => 256,
            Self::TQ2_0 => 256,
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            // ... existing ...
            Self::TQ1_0 => 55,   // 49 + 4 + 2
            Self::TQ2_0 => 66,   // 64 + 2
        }
    }
}
```

### Step 2: Implement Dequantization Functions

```rust
fn dequantize_tq1_0_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 55;
    const QS_SIZE: usize = 49;  // (256 - 4*256/64) / 5

    let num_blocks = num_elements / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        let qs = &block[0..QS_SIZE];
        let qh = &block[QS_SIZE..QS_SIZE + 4];
        let d = f16::from_le_bytes([block[53], block[54]]).to_f32();

        // Decode base-3 packed ternary values
        let mut element_idx = 0;
        for &byte in qs.iter() {
            let mut val = byte as u32;
            for _ in 0..5 {
                if element_idx >= BLOCK_SIZE { break; }
                let trit = (val % 3) as i32 - 1;  // -1, 0, +1
                val /= 3;
                let dequant = (trit as f32) * d;
                output.extend_from_slice(&f16::from_f32(dequant).to_le_bytes());
                element_idx += 1;
            }
        }

        // Handle remaining elements using qh (high bits)
        // ... (edge case handling)
    }

    output
}

fn dequantize_tq2_0_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 256;
    const BLOCK_BYTES: usize = 66;

    let num_blocks = num_elements / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        let qs = &block[0..64];
        let d = f16::from_le_bytes([block[64], block[65]]).to_f32();

        // Decode 2-bit packed values
        for &byte in qs.iter() {
            for j in 0..4 {
                let q2 = ((byte >> (j * 2)) & 0x3) as i32;
                let dequant = ((q2 as f32) - 1.5) * d;
                output.extend_from_slice(&f16::from_f32(dequant).to_le_bytes());
            }
        }
    }

    output
}
```

### Step 3: Update Tensor Loading

In `GgufReader::tensor()`:

```rust
GgmlType::TQ1_0 => dequantize_tq1_0_to_f16(data, num_elements),
GgmlType::TQ2_0 => dequantize_tq2_0_to_f16(data, num_elements),
```

---

## Phase 9: Native TQ Matmul Shaders (Optional)

For maximum performance, implement GPU shaders that operate directly on ternary data.

### TQ2_0 Shader (Simpler)

```wgsl
fn dequant_tq2_0(block_idx: u32, element_idx: u32) -> f32 {
    let block_offset = block_idx * 66u;
    let byte_idx = element_idx / 4u;
    let bit_offset = (element_idx % 4u) * 2u;

    let byte = tq2_data[block_offset + byte_idx];
    let q2 = (byte >> bit_offset) & 0x3u;
    let d = unpack_f16(tq2_data[block_offset + 64u]);

    return (f32(q2) - 1.5) * d;
}
```

### TQ1_0 Shader (Complex - Base-3 Decoding)

Base-3 decoding on GPU is expensive. Consider:

1. **Dequant at load time** (current approach) - simpler, good enough for most cases
2. **Lookup table** - precompute all 243 possible 5-trit combinations
3. **Integer division** - expensive but possible with `%` and `/` operators

---

## Testing

### Benchmark Command

```bash
# Test TQ1_0
cargo run --release --example bench_format -- \
  --gguf assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-TQ1_0.gguf

# Test TQ2_0
cargo run --release --example bench_format -- \
  --gguf assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-TQ2_0.gguf

# Compare all formats
cargo run --release --example bench_format -- \
  --gguf assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf
```

### Quality Validation

Ternary quantization has significant quality loss. Test with:

1. Perplexity comparison against F16 baseline
2. Sample generation quality check
3. Specific task benchmarks (if applicable)

---

## Files to Modify

| File                    | Changes                                      |
| ----------------------- | -------------------------------------------- |
| `src/runtime/gguf.rs`   | Add TQ1_0/TQ2_0 type definitions and dequant |
| `src/runtime/loader.rs` | Handle TQ types in tensor loading            |

### Optional (Native Shaders)

| File                                | Changes                     |
| ----------------------------------- | --------------------------- |
| `src/shaders/matmul_vec_tq2_0.wgsl` | TQ2_0 vector matmul shader  |
| `src/shaders/matmul_mat_tq2_0.wgsl` | TQ2_0 matrix matmul shader  |
| `src/tensor/matrix.rs`              | Add `Matrix::TQ2_0` variant |
| `src/tensor/ops.rs`                 | Add TQ2_0 matmul operations |

---

## Expected Results

| Format | File Size | vs F16 | Quality Impact |
| ------ | --------- | ------ | -------------- |
| F16    | 5656 MB   | -      | Baseline       |
| Q4_K_M | 1788 MB   | -68%   | Minimal        |
| TQ2_0  | 1056 MB   | -81%   | Moderate       |
| TQ1_0  | 944 MB    | -83%   | Significant    |

**Note:** Ternary quantization trades quality for extreme compression. Best suited for:

-   Memory-constrained environments
-   Experimentation / prototyping
-   Models where quality loss is acceptable

---

## Reference

-   llama.cpp `ggml-quants.c`: `dequantize_row_tq1_0()`, `dequantize_row_tq2_0()`
-   GGML ternary quantization PR: https://github.com/ggerganov/llama.cpp/pull/5063

# Native Quantized Type Support Plan

> Extend inline GPU dequantization to all GGML quantized types

## Current State

### Already Native (inline GPU dequant) ✅

| Type | Block Size | Bytes/Block | Bits/Element | Status    |
| ---- | ---------- | ----------- | ------------ | --------- |
| Q4K  | 256        | 144         | 4.5          | ✅ Native |
| Q5K  | 256        | 176         | 5.5          | ✅ Native |
| Q6K  | 256        | 210         | 6.5          | ✅ Native |
| Q8_0 | 32         | 34          | 8.5          | ✅ Native |

### Still CPU Dequantizing ❌

| Type | Block Size | Bytes/Block | Bits/Element | Priority      |
| ---- | ---------- | ----------- | ------------ | ------------- |
| Q4_0 | 32         | 18          | 4.5          | High (common) |
| Q2K  | 256        | 84          | 2.625        | Medium        |
| Q3K  | 256        | 110         | 3.4          | Medium        |
| Q8K  | 256        | 292         | 9.125        | Low           |
| Q4_1 | 32         | 20          | 5.0          | Low           |
| Q5_0 | 32         | 22          | 5.5          | Low           |
| Q5_1 | 32         | 24          | 6.0          | Low           |
| Q8_1 | 32         | 36          | 9.0          | Low           |

---

## Architecture: Trait-Based Quantization

### Core Trait Definition

```rust
// src/tensor/quant.rs (new file)

/// Trait for quantized types that can be dequantized inline in shaders
pub trait QuantizedType: Sized {
    /// GGML type identifier
    const GGML_TYPE: GgmlType;

    /// Number of elements per block
    const BLOCK_SIZE: usize;

    /// Bytes per block
    const BLOCK_BYTES: usize;

    /// Bits per element (for documentation)
    const BITS_PER_ELEMENT: f32;

    /// Generate WGSL struct definition for this block type
    fn wgsl_block_struct() -> &'static str;

    /// Generate WGSL function to load a block from buffer
    fn wgsl_load_block() -> &'static str;

    /// Generate WGSL function to dequantize element at index within block
    fn wgsl_dequant_element() -> &'static str;

    /// Generate WGSL for vectorized 4-element dequantization (optional optimization)
    fn wgsl_dequant_vec4() -> Option<&'static str> { None }
}
```

### Example Implementation: Q4_0

```rust
pub struct Q4_0;

impl QuantizedType for Q4_0 {
    const GGML_TYPE: GgmlType = GgmlType::Q4_0;
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18;
    const BITS_PER_ELEMENT: f32 = 4.5;

    fn wgsl_block_struct() -> &'static str {
        r#"
struct BlockQ4_0 {
    d: u32,      // f16 scale (stored as u32 for alignment)
    qs: array<u32, 4>,  // 16 bytes = 32 x 4-bit values
}
"#
    }

    fn wgsl_load_block() -> &'static str {
        r#"
fn load_block_q4_0(data: ptr<storage, array<u32>>, block_idx: u32) -> BlockQ4_0 {
    let base = block_idx * 5u;  // 18 bytes = 5 u32s (rounded)
    var block: BlockQ4_0;
    let d_raw = data[base];
    block.d = d_raw & 0xFFFFu;
    block.qs[0] = (d_raw >> 16u) | (data[base + 1u] << 16u);
    block.qs[1] = (data[base + 1u] >> 16u) | (data[base + 2u] << 16u);
    block.qs[2] = (data[base + 2u] >> 16u) | (data[base + 3u] << 16u);
    block.qs[3] = (data[base + 3u] >> 16u) | (data[base + 4u] << 16u);
    return block;
}
"#
    }

    fn wgsl_dequant_element() -> &'static str {
        r#"
fn dequant_q4_0(block: BlockQ4_0, idx: u32) -> f32 {
    let d = unpack2x16float(block.d).x;
    let qs_idx = idx / 8u;
    let qs_shift = (idx % 8u) * 4u;
    let q = (block.qs[qs_idx] >> qs_shift) & 0xFu;
    return d * (f32(q) - 8.0);
}
"#
    }
}
```

---

## Block Format Reference

### Q4_0 (18 bytes per 32 elements)

```c
struct block_q4_0 {
    f16 d;           // 2 bytes - scale
    u8 qs[16];       // 16 bytes - 32 x 4-bit quantized values
}
// Dequant: x[i] = d * (qs[i] - 8)
```

### Q4_1 (20 bytes per 32 elements)

```c
struct block_q4_1 {
    f16 d;           // 2 bytes - scale
    f16 m;           // 2 bytes - minimum
    u8 qs[16];       // 16 bytes - 32 x 4-bit quantized values
}
// Dequant: x[i] = d * qs[i] + m
```

### Q5_0 (22 bytes per 32 elements)

```c
struct block_q5_0 {
    f16 d;           // 2 bytes - scale
    u8 qh[4];        // 4 bytes - high bits (32 x 1-bit)
    u8 qs[16];       // 16 bytes - low bits (32 x 4-bit)
}
// Dequant: x[i] = d * ((qs[i] | (qh[i] << 4)) - 16)
```

### Q5_1 (24 bytes per 32 elements)

```c
struct block_q5_1 {
    f16 d;           // 2 bytes - scale
    f16 m;           // 2 bytes - minimum
    u8 qh[4];        // 4 bytes - high bits
    u8 qs[16];       // 16 bytes - low bits
}
// Dequant: x[i] = d * (qs[i] | (qh[i] << 4)) + m
```

### Q8_1 (36 bytes per 32 elements)

```c
struct block_q8_1 {
    f16 d;           // 2 bytes - scale
    f16 s;           // 2 bytes - sum (for dot product optimization)
    i8 qs[32];       // 32 bytes - 32 x 8-bit quantized values
}
// Dequant: x[i] = d * qs[i]
```

### Q2K (84 bytes per 256 elements)

```c
struct block_q2_k {
    u8 scales[16];   // 16 bytes - scales and mins for 16 sub-blocks
    u8 qs[64];       // 64 bytes - 256 x 2-bit quantized values
    f16 d;           // 2 bytes - super-block scale
    f16 dmin;        // 2 bytes - super-block min
}
// Complex dequant with sub-block scales
```

### Q3K (110 bytes per 256 elements)

```c
struct block_q3_k {
    u8 hmask[32];    // 32 bytes - high bits mask
    u8 qs[64];       // 64 bytes - low 2 bits
    u8 scales[12];   // 12 bytes - scales (6-bit packed)
    f16 d;           // 2 bytes - super-block scale
}
// Complex dequant with 3-bit values
```

### Q8K (292 bytes per 256 elements)

```c
struct block_q8_k {
    f32 d;           // 4 bytes - scale
    i8 qs[256];      // 256 bytes - 256 x 8-bit quantized values
    i16 bsums[16];   // 32 bytes - block sums for optimization
}
// Dequant: x[i] = d * qs[i]
```

---

## Implementation Plan

### Phase 1: Infrastructure (1-2 days)

1. **Create `src/tensor/quant.rs`**

    - Define `QuantizedType` trait
    - Implement for existing types (Q4K, Q5K, Q6K, Q8_0) to validate design

2. **Create shader template system**
    - `src/shaders/quant/mod.wgsl` - common utilities
    - Template that injects type-specific code

### Phase 2: Simple Types (2-3 days)

Priority order based on model availability:

1. **Q4_0** - Most common legacy format

    - Add `Matrix::Q4_0` variant
    - Create `matmul_vec_q4_0.wgsl` and `matmul_mat_q4_0.wgsl`
    - Add loader path in `loader.rs`

2. **Q8_1** - Simple extension of Q8_0

    - Similar to Q8_0 but with sum field

3. **Q4_1, Q5_0, Q5_1** - 5-bit variants
    - Slightly more complex due to high-bit handling

### Phase 3: K-Quant Extensions (2-3 days)

1. **Q2K** - Aggressive 2-bit quantization

    - Complex sub-block scale handling
    - May need specialized shader for performance

2. **Q3K** - 3-bit quantization

    - Similar complexity to Q2K

3. **Q8K** - High-precision K-quant
    - Simpler than Q2K/Q3K but larger blocks

### Phase 4: Unified Matrix Type (Optional, 1-2 days)

Replace individual `Matrix::Q4K`, `Matrix::Q5K`, etc. with:

```rust
pub enum Matrix {
    Fp16(TensorGpu<f16, ReadWrite>),
    Int8 { w: TensorGpu<u8, ReadWrite>, m: TensorGpu<f16, ReadWrite> },
    Fp4 { ... },
    /// Native quantized matrix - stores raw GGML blocks
    Quantized {
        /// Raw block data
        w: TensorGpu<u8, ReadWrite>,
        /// Shape metadata
        s: TensorGpu<u8, ReadWrite>,
        /// Quantization type
        quant_type: GgmlType,
    },
}
```

This simplifies the enum and makes adding new types trivial.

---

## File Changes Summary

### New Files

```text
src/tensor/quant.rs              # QuantizedType trait + implementations
src/shaders/quant/common.wgsl    # Shared dequant utilities
src/shaders/matmul_vec_q4_0.wgsl # Q4_0 vector matmul
src/shaders/matmul_mat_q4_0.wgsl # Q4_0 matrix matmul
src/shaders/matmul_vec_q2k.wgsl  # Q2K vector matmul
src/shaders/matmul_mat_q2k.wgsl  # Q2K matrix matmul
# ... similar for other types
```

### Modified Files

```text
src/tensor/mod.rs     # Add: pub mod quant;
src/tensor/matrix.rs  # Add new Matrix variants or unified Quantized variant
src/tensor/ops.rs     # Add TensorOp::matmul_vec_q4_0, etc.
src/runtime/loader.rs # Add native loading paths for new types
```

---

## Estimated Impact

| Metric    | Current (CPU dequant) | Native (GPU dequant) | Improvement |
| --------- | --------------------- | -------------------- | ----------- |
| Load Time | ~5s (Q4_0 model)      | ~3s                  | 40% faster  |
| Memory    | 2x (dequant buffer)   | 1x                   | 50% less    |
| Inference | Same                  | Same                 | -           |

The main benefit is **faster loading** and **lower memory** during load, since we skip the CPU dequantization step and intermediate F16 buffer.

---

## Priority Recommendation

1. **Q4_0** - High priority, very common format
2. **Q2K/Q3K** - Medium priority, used in smaller models
3. **Others** - Low priority, less common

Start with Q4_0 as a proof of concept, then use the trait system to quickly add others.

---

## Next Steps

1. Review and approve this plan
2. Implement `QuantizedType` trait with Q4_0 as first target
3. Create Q4_0 shaders
4. Test with a Q4_0 model
5. Iterate for remaining types

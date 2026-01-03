# Code Cleanup Audit

This document identifies dead code and non-essential quantization types that can be removed to focus on **F16**, **Q8_0**, **Q4_0**, **Int8**, and **NF4** support only.

---

## 1. Compiler-Detected Dead Code

From `cargo build` warnings:

| Location                   | Item                   | Type                                            |
| -------------------------- | ---------------------- | ----------------------------------------------- |
| `src/runtime/gguf.rs`      | `resolve_name`         | Method (never used externally, only internally) |
| `src/tensor/lazy.rs:11`    | `Q4K_BLOCK_BYTES`      | Constant                                        |
| `src/tensor/lazy.rs:14`    | `transpose_q4k_layout` | Function                                        |
| `src/runtime/loader.rs:27` | `Q4K_BLOCK_BYTES`      | Constant                                        |
| `src/runtime/loader.rs:35` | `transpose_q4k_layout` | Function                                        |

---

## 2. Quantization Types to Remove

**Target types to KEEP:** F16, Q8_0, Q4_0, Int8, NF4 (Fp4)

### 2.1 Matrix Variants to Remove (`src/tensor/matrix.rs`)

| Variant       | Lines   | Description        |
| ------------- | ------- | ------------------ |
| `Matrix::Q4K` | 94-103  | Q4_K native blocks |
| `Matrix::Q5K` | 104-112 | Q5_K native blocks |
| `Matrix::Q6K` | 113-121 | Q6_K native blocks |

**Keep:** `Matrix::Fp16`, `Matrix::Int8`, `Matrix::Fp4` (NF4), `Matrix::Q8_0`

### 2.2 TensorOp Functions to Remove (`src/tensor/ops.rs`)

| Function             | Lines     | Description                  |
| -------------------- | --------- | ---------------------------- |
| `matmul_vec_q4k`     | 1355-1467 | Q4_K vector matmul           |
| `matmul_mat_q4k`     | 1469-1554 | Q4_K matrix matmul           |
| `matmul_vec_q4k_opt` | 1556-1628 | Q4_K optimized vector matmul |
| `matmul_mat_q4k_opt` | 1630-1697 | Q4_K optimized matrix matmul |
| `matmul_vec_q5k`     | 1699-1766 | Q5_K vector matmul           |
| `matmul_mat_q5k`     | 1768-1834 | Q5_K matrix matmul           |
| `matmul_vec_q6k`     | 1836-1903 | Q6_K vector matmul           |
| `matmul_mat_q6k`     | 1905-1971 | Q6_K matrix matmul           |
| `quantize_mat_q4k`   | (search)  | Q4_K quantization shader op  |
| `quantize_mat_q5k`   | (search)  | Q5_K quantization shader op  |
| `quantize_mat_q6k`   | (search)  | Q6_K quantization shader op  |

**Keep:** `matmul_vec_fp16`, `matmul_mat_fp16`, `matmul_vec_int8`, `matmul_mat_int8`, `matmul_vec_nf4`, `matmul_mat_nf4`, `matmul_vec_q8_0`, `matmul_mat_q8_0`, `quantize_mat_int8`, `quantize_mat_nf4`

### 2.3 Shaders to Remove (`src/shaders/`)

| Shader                             | Description                  |
| ---------------------------------- | ---------------------------- |
| `matmul_vec_q4k.wgsl`              | Q4_K vector matmul           |
| `matmul_vec_q4k_f16.wgsl`          | Q4_K vector matmul (f16)     |
| `matmul_vec_q4k_v2.wgsl`           | Q4_K optimized vector matmul |
| `matmul_mat_q4k.wgsl`              | Q4_K matrix matmul           |
| `matmul_mat_q4k_f16.wgsl`          | Q4_K matrix matmul (f16)     |
| `matmul_mat_q4k_opt.wgsl`          | Q4_K optimized matrix matmul |
| `matmul_vec_q5k.wgsl`              | Q5_K vector matmul           |
| `matmul_mat_q5k.wgsl`              | Q5_K matrix matmul           |
| `matmul_vec_q6k.wgsl`              | Q6_K vector matmul           |
| `matmul_mat_q6k.wgsl`              | Q6_K matrix matmul           |
| `subgroup/matmul_vec_q4k.wgsl`     | Q4_K subgroup variant        |
| `subgroup/matmul_vec_q4k_f16.wgsl` | Q4_K subgroup variant (f16)  |

**Keep:** `matmul_vec_fp16.wgsl`, `matmul_mat_fp16.wgsl`, `matmul_vec_int8.wgsl`, `matmul_mat_int8.wgsl`, `matmul_vec_nf4.wgsl`, `matmul_mat_nf4.wgsl`, `quant_mat_int8.wgsl`, `quant_mat_nf4.wgsl`, `matmul_vec_q8_0.wgsl`, `matmul_mat_q8_0.wgsl`, `subgroup/matmul_vec_fp16.wgsl`, `subgroup/matmul_vec_int8.wgsl`, `subgroup/matmul_vec_nf4.wgsl`

### 2.4 GGUF Dequantization Functions to Remove (`src/runtime/gguf.rs`)

| Function                 | Lines   | Description            |
| ------------------------ | ------- | ---------------------- |
| `dequantize_q4_k_to_f16` | 91-201  | Q4_K dequantization    |
| `dequantize_q5_k_to_f16` | 203-262 | Q5_K dequantization    |
| `dequantize_q6_k_to_f16` | 264-416 | Q6_K dequantization    |
| `dequantize_q3_k_to_f16` | 418-508 | Q3_K dequantization    |
| `dequantize_q2_k_to_f16` | 510-565 | Q2_K dequantization    |
| `repack_q4_k_to_int8`    | 771-838 | Q4_K to Int8 repacking |
| `repack_q5_k_to_int8`    | 840-918 | Q5_K to Int8 repacking |
| `repack_q6_k_to_int8`    | 920-997 | Q6_K to Int8 repacking |
| `get_scale_min_k4`       | 77-89   | Helper for Q4_K/Q5_K   |

**Keep:** `dequantize_q8_0_to_f16`, `dequantize_q4_0_to_f16`, `repack_q8_0_to_int8`, `repack_q4_0_to_nf4`

### 2.5 GgmlType Enum Cleanup (`src/runtime/gguf.rs:1030-1065`)

Remove handling for these types (keep enum variants for parsing, but remove dequant paths):

-   `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`
-   `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `Q8K`
-   All `IQ*` types

### 2.6 Lazy Matrix Variants to Remove (`src/tensor/lazy.rs`)

| Variant                  | Lines | Description       |
| ------------------------ | ----- | ----------------- |
| `LazyMatrixVariant::Q4K` | 53-56 | Q4_K lazy loading |
| `LazyMatrixVariant::Q5K` | 57-60 | Q5_K lazy loading |
| `LazyMatrixVariant::Q6K` | 61-64 | Q6_K lazy loading |

**Keep:** `LazyMatrixVariant::Fp16`, `LazyMatrixVariant::Int8`, `LazyMatrixVariant::Q8_0`

### 2.7 Loader Direct Loading Paths (`src/runtime/loader.rs`)

| Function/Path                          | Lines   | Description                  |
| -------------------------------------- | ------- | ---------------------------- |
| `try_load_matrix_direct` Q4K path      | 842-846 | Disabled Q4_K native loading |
| `try_load_matrix_direct` Q5K path      | 847-865 | Q5_K native loading          |
| `try_load_matrix_direct` Q6K path      | 866-884 | Q6_K native loading          |
| `try_load_matrix_direct` Q4_0→NF4 path | 885-901 | Q4_0 to NF4 conversion       |

**Keep:** Q8_0→Int8 path (829-841)

### 2.8 Quant Enum Simplification (`src/runtime/model.rs:122-135`)

Remove:

-   `Quant::SF4`

**Keep:** `Quant::None` (F16 default), `Quant::Int8`, `Quant::NF4`

---

## 3. Graph Operations to Remove (`src/tensor/graph.rs`)

The lazy graph system references Matrix types. After removing non-target matrix types, update:

-   `MatrixRef` handling in `LazyOp::MatMul` and `LazyOp::MatMulBias`
-   Metal Q4K dispatch code (lines 762-778) - already gated by feature flag

---

## 5. Macros Helper Functions (`src/tensor/ops.rs`)

**Keep all:** `Macros::nf4`, `Macros::int8`, `Macros::shader_f16`, `Macros::tensor`, etc.

---

## 6. Constants to Remove

None - keep `NF4_BLOCK_SIZE` and `INT8_BLOCK_SIZE`.

---

## 7. Float4Quant Class (`src/tensor/matrix.rs:20-78`)

**Keep** - used for NF4 quantization.

---

## Summary

### Files to Modify

1. `src/tensor/matrix.rs` - Remove Q4K, Q5K, Q6K variants
2. `src/tensor/ops.rs` - Remove non-target matmul ops and quantization ops
3. `src/tensor/lazy.rs` - Remove non-target lazy variants
4. `src/tensor/graph.rs` - Update after matrix changes
5. `src/runtime/gguf.rs` - Remove non-target dequantization functions
6. `src/runtime/loader.rs` - Remove non-target loading paths
7. `src/runtime/model.rs` - Simplify Quant enum

### Files to Delete

-   ~12 shader files (Q4K, Q5K, Q6K variants - see section 2.3)

### Estimated Impact

-   ~1500+ lines of Rust code removal
-   ~12 shader files deletion
-   Simpler Matrix enum with 4 variants (Fp16, Int8, Fp4/NF4, Q8_0) + Q4_0 via dequant

---

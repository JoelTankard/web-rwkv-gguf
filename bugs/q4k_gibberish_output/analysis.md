# Q4_K Model Gibberish Output Bug

## Symptom

When running the chat example with a Q4_K_M quantized model, the output is gibberish (random words, mixed languages) instead of coherent English text.

**Example:**

```text
User: what is an apple?
A: zwкипедиДДДкипедиМкипедиПрzwкипедиДкипедиДкипедиСаλкипедиЗаMOSzw...
```

Or random English/French word salad after attempted fixes.

**Expected:** Coherent English response about apples.

## Environment

-   Model: `assets/models/2.9b-Q4_K_M.gguf`
-   Reference models that work correctly:
    -   WebGPU `2.9b-f16.gguf` - ~40 tok/s
    -   Metal `2.9b-Q8_0.gguf` - ~50 tok/s expected

## Investigation Progress

### Session 1: Initial Analysis

**Verified Components (Working Correctly):**

1. **Dequantization Logic**: CPU and shader dequantization match exactly (verified with `test_q4k_dequant.rs`)
2. **Shader Selection**: `Matrix::Q4K` correctly uses `_opt` shaders with transposed layout
3. **Transpose Function**: `transpose_q4k_layout` correctly transposes from row-major to column-major
4. **Block Layout**: Q4_K block structure (144 bytes) is correctly parsed

### Session 2: Shape Convention Investigation

**Attempted Fix (INCOMPLETE):**
Investigated shape convention mismatch between loader and shader.

**Shape Convention Analysis:**

-   GGUF stores weight matrices as `[out_features, in_features]` = `[M, K]`
-   After `shape.reverse()` in gguf.rs: `[K, M]`
-   After `from_slice_rev([K, M])`: `Shape::new(M, K, 1, 1)` → `shape[0]=M, shape[1]=K`

**Changes Made (may need revision):**

1. `loader.rs:864-865`: Swapped K/M extraction to match actual shape convention
2. `loader.rs:876`: Created `shader_shape = Shape::new(k, m, 1, 1)` for shader metadata
3. `lazy.rs:155-165`: Same fix for lazy loading path

**Result:** Still producing gibberish output. The shape fix alone is not sufficient.

### Current Hypotheses

1. **Shape Convention Mismatch**: The shader expects `va.shape = [K, M, B]` but may be receiving wrong values
2. **Transpose Direction Wrong**: The transpose may be in the wrong direction for the shader's indexing
3. **Input Cache Indexing**: The `input_cache` array indexing in `dot_q4k_cached` may not match element ordering
4. **Dispatch Calculation**: The dispatch grid calculation may not match shader expectations

### Key Insight: F16 Matrix Shape Convention

F16 matmul uses `matrix.check_shape([k, m, b, 1])` expecting `shape[0]=K, shape[1]=M`.
But `from_slice_rev([K, M])` produces `shape[0]=M, shape[1]=K`.

**This suggests F16 matrices have a different shape convention OR there's something else going on.**

Need to verify:

1. What shape does F16 matrix actually have after loading?
2. Does the F16 shader use the same View struct convention?

## Shader Layout Analysis

| Shader File               | Layout Used                             | Status                 |
| ------------------------- | --------------------------------------- | ---------------------- |
| `matmul_vec_q4k_v2.wgsl`  | Transposed: `(sb * m + row)`            | ✅ Used by Matrix::Q4K |
| `matmul_mat_q4k_opt.wgsl` | Transposed: `(sb_idx * num_rows + row)` | ✅ Used by Matrix::Q4K |
| `matmul_vec_q4k.wgsl`     | Row-major: `(row * num_sb_k + sb)`      | ⚠️ Not used            |

## Q4_K Block Structure (144 bytes per 256 elements)

```text
Offset  Size  Field
0       2     d (f16) - super-block scale
2       2     dmin (f16) - super-block min
4       12    scales (12 bytes) - 8 sub-block scales/mins, 6-bit each
16      128   qs (128 bytes) - 256 4-bit quantized values, 2 per byte
```

## SOLUTION FOUND

**Root Cause:** The transposed shader path (`matmul_vec_q4k_opt`, `matmul_mat_q4k_opt`) had issues with the transpose logic or indexing.

**Fix Applied:** Switch to non-transposed shader path:

1. `loader.rs`: Use raw Q4K data without transpose, create `shader_shape = Shape::new(k, m, 1, 1)`
2. `lazy.rs`: Same change for lazy loading
3. `matrix.rs`: Use `matmul_vec_q4k` and `matmul_mat_q4k` instead of `_opt` variants

**Additional Fix:** Fixed `unpack4x16half` shader validation error in 3 files:

-   `matmul_vec_q4k_f16.wgsl`
-   `subgroup/matmul_vec_q4k_f16.wgsl`
-   `matmul_mat_q4k_f16.wgsl`

The bug was `vec4<f16>(unpack2x16float(...))` - `unpack2x16float` returns `vec2<f32>`, not `vec2<f16>`.
Fixed to: explicit conversion `f16(lo.x), f16(lo.y), f16(hi.x), f16(hi.y)`

**Result:** Model now produces coherent output.

**Future Work:** The transposed shader path needs debugging - the transpose logic or indexing has a bug that causes incorrect matmul results

## Test Commands

```bash
# Test F16 model (should work)
cargo run --release --example chat -- --model assets/models/2.9b-f16.gguf -a

# Test Q4_K model (currently broken)
cargo run --release --example chat -- --model assets/models/2.9b-Q4_K_M.gguf -a

# Run dequantization unit test
cargo run --release --example test_q4k_dequant
```

## Files to Examine

-   `src/runtime/loader.rs:35-46` - transpose_q4k_layout function
-   `src/runtime/loader.rs:850-879` - Q4_K loading path (modified)
-   `src/tensor/lazy.rs:152-166` - Q4K lazy loading (modified)
-   `src/tensor/ops.rs:1564-1627` - matmul_vec_q4k_opt
-   `src/tensor/ops.rs:1636-1696` - matmul_mat_q4k_opt
-   `src/shaders/matmul_vec_q4k_v2.wgsl` - vector shader (transposed)
-   `src/shaders/matmul_mat_q4k_opt.wgsl` - matrix shader (transposed)

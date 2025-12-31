# Shader Rule Violations Report

Based on my analysis of all 43 shaders against the rules in `@/Users/joel/Dev/Experimental/web-rwkv-gguf/optimization_plan/shaders_rules.md`, here are the violations found:

---

## Rule 3: Kill Branches (Divergent Control Flow)

### **High Priority - Dynamic Branches**

| File                                                                                | Line                                         | Issue                                          |
| ----------------------------------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_mat_q4k.wgsl:70-79` | `get_scale_min_k4()`                         | Divergent `if j < 4u` branch per-invocation    |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_q4k.wgsl:62-72` | `get_scale_min_k4()`                         | Same divergent branch pattern                  |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/token_shift.wgsl:101-116`  | `if token == 0u`                             | Divergent branch for first token vs others     |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/channel_mix.wgsl:104`      | `1.0 / (1.0 + exp(-load_r(bti)))`            | Sigmoid computed inline - could use branchless |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v4.wgsl:87,93`    | `1.0 / (1.0 + exp(-...))`                    | Sigmoid computed inline                        |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/quant_mat_nf4.wgsl:70-76`  | Inner loop with `if abs(qq - xx) <= min_err` | Highly divergent branch in hot loop            |

---

## Rule 9: Expensive Built-ins

### **`exp` usage (expensive)**

| File                                                                                     | Lines | Count                                           | Context |
| ---------------------------------------------------------------------------------------- | ----- | ----------------------------------------------- | ------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/softmax.wgsl:89,109`            | 2     | `exp(value - maximum)` called twice per element |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v7.wgsl:69`            | 1     | `exp(-0.606531 * sigmoid(x))` in `act_w()`      |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v4.wgsl:98-99,109-110` | 4     | Multiple `exp()` calls per iteration            |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/channel_mix.wgsl:104`           | 1     | `exp(-load_r(bti))`                             |

### **Repeated `inverseSqrt` (uses sqrt internally)**

| File                                                                             | Line                                    | Context |
| -------------------------------------------------------------------------------- | --------------------------------------- | ------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/layer_norm.wgsl:105`    | `inverseSqrt(_var)`                     |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/normalize.wgsl:108,144` | `inverseSqrt()` in rms_norm and l2_norm |

---

## Rule 2: Memory Bandwidth (f32 vs f16)

### **Workgroup memory using f32 where f16 might suffice**

| File                                                                              | Lines                                                       | Issue                                                 |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/softmax.wgsl:9`          | `var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>`       | Could use f16 for intermediate sums                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/layer_norm.wgsl:14-16`   | `mu`, `m2`, `count` arrays all f32                          | Welford algorithm may need f32 precision              |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/normalize.wgsl:11`       | `var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>`       | Could use f16                                         |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_fp16.wgsl:23` | `var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>`       | Accumulator in f32 (may be intentional for precision) |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_mat_q4k.wgsl:45`  | `var<workgroup> sa: array<array<vec4<f32>, 64>, TILE_SIZE>` | Large f32 workgroup array                             |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v7.wgsl:36-41`  | 6 shared arrays all `vec4<f32>`                             | Could potentially use f16                             |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v6.wgsl:38-41`  | 4 shared arrays all `vec4<f32>`                             | Same issue                                            |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v5.wgsl:38-41`  | 4 shared arrays all `vec4<f32>`                             | Same issue                                            |

### **State buffers always f32**

| File                                                                          | Line                      | Issue          |
| ----------------------------------------------------------------------------- | ------------------------- | -------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v7.wgsl:23` | `state: array<vec4<f32>>` | No f16 variant |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v6.wgsl:24` | `state: array<vec4<f32>>` | No f16 variant |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v5.wgsl:24` | `state: array<vec4<f32>>` | No f16 variant |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v4.wgsl:20` | `state: array<vec4<f32>>` | No f16 variant |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/channel_mix.wgsl:16` | `state: array<vec4<f32>>` | No f16 variant |

---

## Rule 6: Workgroup Size Tuning

### **Hardcoded reduction steps assume BLOCK_SIZE=128**

| File                                                                                          | Lines                               | Issue                             |
| --------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/softmax.wgsl:68-74,94-100`           | `reduce_*(index, 64u)` down to `1u` | Assumes 128 threads, no 128u step |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/layer_norm.wgsl:89-95`               | Same pattern                        |                                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/normalize.wgsl:63-69,99-105,135-141` | Same pattern                        |                                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_fp16.wgsl:90-96`          | Same pattern                        |                                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_int8.wgsl:98-104`         | Same pattern                        |                                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_nf4.wgsl:157-163`         | Same pattern                        |                                   |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_q4k.wgsl:168-174`         | Same pattern                        |                                   |

**Note:** If `BLOCK_SIZE` is changed to 64 (recommended for Apple Silicon per Rule 6), these reductions would be incorrect. The reduction should be parameterized.

---

## Rule 5: Workgroup Memory Opportunities

### **Missing workgroup caching**

| File                                                                                | Issue                                     |
| ----------------------------------------------------------------------------------- | ----------------------------------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_q4k.wgsl:29-30` | `input_cache` declared but **never used** |

```wgsl
@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_q4k.wgsl:29-30
// Shared memory cache for input vector tile (64 vec4 = 256 elements)
var<workgroup> input_cache: array<vec4<f32>, 64>;
```

This is dead code - the cache is declared but input is read directly from global memory in the loop.

---

## Rule 8: Dead Code

| File                                                                                | Lines                    | Issue                     |
| ----------------------------------------------------------------------------------- | ------------------------ | ------------------------- |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/matmul_vec_q4k.wgsl:29-30` | `input_cache`            | Unused workgroup variable |
| `@/Users/joel/Dev/Experimental/web-rwkv-gguf/src/shaders/time_mix_v7.wgsl:248-255`  | Commented-out code block | Should be removed         |

---

## Summary by Priority

### **High Impact (Rule 2, 3, 9)**

1. **Q4K shaders**: Divergent `if j < 4u` branch in `get_scale_min_k4()` - replace with branchless math
2. **softmax.wgsl**: Double `exp()` call - cache the result
3. **time_mix_v4.wgsl**: 4x `exp()` calls per loop iteration
4. **quant_mat_nf4.wgsl**: Divergent inner loop for quantization

### **Medium Impact (Rule 2, 6)**

5. **State buffers**: All time_mix/channel_mix use f32 state with no f16 option
6. **Workgroup arrays**: Many use f32 where f16 might work
7. **Reduction hardcoding**: Assumes BLOCK_SIZE=128, not flexible for Apple Silicon tuning

### **Low Impact (Rule 8)**

8. **Dead code**: Unused `input_cache` in matmul_vec_q4k.wgsl
9. **Commented code**: time_mix_v7.wgsl lines 248-255

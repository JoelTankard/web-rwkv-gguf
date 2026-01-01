# Metal Kernel Rewrite Plan

## Goal

Achieve 200+ tok/s on Apple Silicon by fixing the Metal kernel implementations to match the correct RWKV algorithms.

## Current State

-   **Isolated Metal Q4K matmul**: 207 tok/s ✅
-   **Full Metal layer execution**: 12 tok/s ❌
-   **Encoder overhead**: Fixed (0.4ms encode time)
-   **GPU execution**: 248ms per token (should be ~5ms)

## Root Cause

The Metal kernels are **algorithmically incorrect**, not slow. The `time_mix_v7_fp16` kernel is a simplified element-wise version that doesn't implement the full RWKV recurrence.

### Current Metal Kernel (Wrong)

```metal
// src/metal/kernels.rs:682-742
// Element-wise: one thread per element, no state matrix operations
uint state_idx = batch * state_stride + elem * (head_size + 1) + head;
float s = state[state_idx];
float new_s = ww * s + kk * vv + bonus * aa;
float y = rr * new_s;
```

### Correct Algorithm (from WebGPU)

```wgsl
// src/shaders/time_mix_v7.wgsl
// Full algorithm: HEAD_SIZE × HEAD_SIZE state matrix per head
for (var j = 0u; j < HEAD_SIZE; j += 1u) {
    // Load 4 state vectors (4 × HEAD_SIZE matrix slice)
    ss[0] = state[bji]; ss[1] = state[bji + stride]; ...

    // Update state: s = w * s + k * v + bonus * a
    ss[0] = ss[0] * ww[0] + kk[0] * vv + sa * bb[0];

    // Accumulate output: y += r * s
    y += rr[0] * ss[0] + rr[1] * ss[1] + ...

    // Store updated state
    state[bji] = ss[0]; ...
}
```

## Architecture

The pipeline should be **pure Rust + Metal** with no WebGPU in the inference path:

```
┌─────────────────────────────────────────────────────────────┐
│                     Inference Flow                          │
├─────────────────────────────────────────────────────────────┤
│  1. Embedding lookup (Metal)                                │
│  2. For each layer:                                         │
│     └─ Execute ALL ops in single Metal command buffer       │
│  3. Head projection (Metal)                                 │
│  4. Single sync at end                                      │
└─────────────────────────────────────────────────────────────┘
```

## Kernels to Fix

### Priority 1: time_mix_v7 (Critical)

The main recurrence kernel. Current implementation is fundamentally wrong.

**State Layout**: `[batch, head_size + 1, num_emb]` stored as f32

-   Row 0: Previous token's hidden state (for token shift)
-   Rows 1..head_size+1: HEAD_SIZE × HEAD_SIZE state matrix per head

**Algorithm**:

1. Load shared data into threadgroup memory (r, w, k, a, kk)
2. Compute bonus term: `sa = sum_j(-kk[j] * state[batch, j, i])`
3. For each j in HEAD_SIZE:
    - Load state matrix slice: `ss[0..3] = state[batch, j*4+1..j*4+4, i]`
    - Update: `ss = w * ss + k * v + sa * b`
    - Accumulate: `y += r * ss`
    - Store updated state
4. Write output

**Metal Implementation Strategy**:

-   Use `threadgroup` memory for shared r, w, k, a, b values
-   Process vec4 (4 elements) at a time for memory coalescing
-   One threadgroup per (token, head) pair
-   HEAD_SIZE threads per threadgroup

### Priority 2: time_first (Important)

Computes the "bonus" term for attention.

**Algorithm**:

```
y = sum_j(u[j] * k[j] * r[j]) * v
x = x + y
```

**Metal Implementation**:

-   Parallel reduction within threadgroup
-   Use simd_sum for efficient reduction

### Priority 3: token_shift (Verify)

Current implementation looks correct but verify state indexing.

**Algorithm**:

```
output = lerp(prev, current, mix_factor)
prev = state[batch, 0, elem] for first token, else input[token-1]
```

### Priority 4: channel_mix_v7 (Verify)

FFN state update. Simpler than attention.

**Algorithm**:

```
state[batch, head_size+1, elem] = v[elem]
x = x + v
```

## Implementation Plan

### Phase 1: Understand State Layout (Day 1)

1. **Analyze WebGPU state indexing**

    - File: `src/shaders/time_mix_v7.wgsl`
    - Function: `compute_index(batch, token, index)`
    - Understand view.stride, view.offset

2. **Map to Metal buffer layout**

    - State buffer: extracted from wgpu via `get_metal_buffer_from_wgpu`
    - Same underlying memory, need correct indexing

3. **Create test harness**
    - Compare Metal kernel output vs WebGPU output
    - Use small model (0.1b) for fast iteration

### Phase 2: Rewrite time_mix_v7 (Day 1-2)

1. **Port algorithm from WGSL to Metal**

    ```metal
    kernel void time_mix_v7_fp16(
        device const uint* cursors [[buffer(0)]],
        device float* state [[buffer(1)]],
        device const half* r [[buffer(2)]],
        device const half* w [[buffer(3)]],
        device const half* n [[buffer(4)]],  // packed k,v,a,kk
        device half* x [[buffer(5)]],
        constant uint& head_size [[buffer(6)]],
        constant uint& num_heads [[buffer(7)]],
        constant uint& num_tokens [[buffer(8)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]
    ) {
        // Threadgroup memory for shared data
        threadgroup float4 shared_r[HEAD_SIZE];
        threadgroup float4 shared_w[HEAD_SIZE];
        threadgroup float4 shared_k[HEAD_SIZE];
        threadgroup float4 shared_a[HEAD_SIZE];
        threadgroup float4 shared_b[HEAD_SIZE];

        uint token = tgid.x;
        uint head = tgid.y;
        uint index = tid;  // Element within head

        // Load shared data
        if (index < HEAD_SIZE) {
            uint ti = token * stride + head * HEAD_SIZE + index;
            shared_r[index] = load_r(ti);
            shared_w[index] = act_w(load_w(ti));
            shared_k[index] = load_k(ti);
            float4 a = load_a(ti);
            float4 kk = load_kk(ti);
            shared_a[index] = -kk;
            shared_b[index] = kk * a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute bonus term
        float4 sa = float4(0.0);
        for (uint j = 0; j < HEAD_SIZE; j++) {
            float4 aa = shared_a[j];
            // Load state matrix slice
            uint bji = compute_index(batch, j * 4 + 1, index);
            float4 ss0 = state[bji];
            float4 ss1 = state[bji + stride];
            float4 ss2 = state[bji + stride * 2];
            float4 ss3 = state[bji + stride * 3];
            sa += mat4(ss0, ss1, ss2, ss3) * aa;
        }

        // Main recurrence
        float4 vv = load_v(ti);
        float4 y = float4(0.0);
        for (uint j = 0; j < HEAD_SIZE; j++) {
            float4 rr = shared_r[j];
            float4 ww = shared_w[j];
            float4 kk = shared_k[j];
            float4 bb = shared_b[j];

            uint bji = compute_index(batch, j * 4 + 1, index);
            float4 ss0 = state[bji];
            float4 ss1 = state[bji + stride];
            float4 ss2 = state[bji + stride * 2];
            float4 ss3 = state[bji + stride * 3];

            ss0 = ss0 * ww[0] + kk[0] * vv + sa * bb[0];
            ss1 = ss1 * ww[1] + kk[1] * vv + sa * bb[1];
            ss2 = ss2 * ww[2] + kk[2] * vv + sa * bb[2];
            ss3 = ss3 * ww[3] + kk[3] * vv + sa * bb[3];

            y += rr[0] * ss0 + rr[1] * ss1 + rr[2] * ss2 + rr[3] * ss3;

            state[bji] = ss0;
            state[bji + stride] = ss1;
            state[bji + stride * 2] = ss2;
            state[bji + stride * 3] = ss3;
        }

        store_x(ti, y);
    }
    ```

2. **Handle vec4 packing**

    - Input tensors are packed as `vec2<u32>` (4 × f16)
    - Need `unpack4x16float` equivalent in Metal

3. **Test against WebGPU output**
    - Run same input through both paths
    - Compare outputs element-by-element

### Phase 3: Rewrite time_first (Day 2)

1. **Port reduction algorithm**

    ```metal
    // Parallel reduction: y = sum_j(u[j] * k[j] * r[j])
    threadgroup float4 shared_x[HEAD_SIZE];
    shared_x[tid] = u * k * r;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = HEAD_SIZE >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            shared_x[tid] += shared_x[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float xx = dot(shared_x[0], float4(1.0));
    x = x + xx * v;
    ```

### Phase 4: Verify Other Kernels (Day 2-3)

1. **token_shift**: Verify state indexing matches WebGPU
2. **channel_mix_v7**: Verify FFN state update
3. **layer_norm, group_norm**: Should be correct, verify
4. **matmul kernels**: Already working (207 tok/s isolated)

### Phase 5: Integration Testing (Day 3)

1. **End-to-end test**

    - Run full model inference
    - Compare output quality with WebGPU path
    - Measure tok/s

2. **Correctness validation**

    - Generate text with both paths
    - Verify outputs match (within fp16 tolerance)

3. **Performance profiling**
    - Use Metal System Trace
    - Identify any remaining bottlenecks

## Success Criteria

-   [ ] Metal time_mix_v7 produces correct output (matches WebGPU within tolerance)
-   [ ] Metal time_first produces correct output
-   [ ] Full model inference works end-to-end
-   [ ] Generation quality matches WebGPU path
-   [ ] Performance: 150+ tok/s (target: 200+ tok/s)

## Files to Modify

1. `src/metal/kernels.rs` - Rewrite time_mix_v7, time_first kernels
2. `src/metal/ops.rs` - Update dispatch parameters if needed
3. `src/metal/layer_dispatcher.rs` - Verify buffer passing
4. `src/runtime/v7.rs` - Ensure pure Metal path (no wgpu in inference)

## Testing Commands

```bash
# Build with Metal backend
cargo build --release --features metal-backend

# Run isolated kernel test
cargo run --release --features metal-backend --example bench_metal_vs_wgpu_real

# Run full benchmark
cargo run --release --features metal-backend --example benchmark -- \
    --model assets/models/2.9b-Q4_K_M.gguf \
    --title "Metal Kernel Fix" \
    --change "Correct time_mix_v7 implementation"
```

## References

-   WebGPU time_mix_v7: `src/shaders/time_mix_v7.wgsl`
-   WebGPU time_first: `src/shaders/time_mix_v7.wgsl` (same file, different entry point)
-   RWKV-7 paper: <https://arxiv.org/abs/2404.05892>
-   Metal Best Practices: <https://developer.apple.com/documentation/metal/gpu_programming_guide>

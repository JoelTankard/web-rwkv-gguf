# Pure Metal Execution: Path to 200 tok/s

## Goal

**Target:** 200+ tok/s on Apple Silicon  
**Approach:** Execute entire RWKV layer computation in Metal, eliminating all wgpu/Metal synchronization overhead  
**Proven Potential:** Metal Q4K kernel achieves 207.8 tok/s in isolation (4.78x faster than WebGPU)

## Why Pure Metal is the Only Path

The synchronization problem between wgpu and Metal command queues is **unsolvable without significant overhead**:

| Approach          | Sync Points | Result                           |
| ----------------- | ----------- | -------------------------------- |
| Per-op sync       | 192/pass    | 10.7 tok/s (worse than baseline) |
| Per-layer sync    | 64/pass     | ~60-90 tok/s (still too slow)    |
| Batched execution | N/A         | Broken (data dependencies)       |
| **Pure Metal**    | **2/pass**  | **200+ tok/s**                   |

**The only way to achieve 200 tok/s is to eliminate inter-queue synchronization entirely.**

---

## Architecture

### Target Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PURE METAL FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CPU: Load embeddings into GPU buffer                           │
│       ↓                                                         │
│  SYNC (1)                                                       │
│       ↓                                                         │
│  Metal: embed_layer_norm                                        │
│       ↓                                                         │
│  Metal: layer[0] → layer[1] → ... → layer[31]                  │
│       ↓                                                         │
│  Metal: head_layer_norm → head_matmul                          │
│       ↓                                                         │
│  SYNC (2)                                                       │
│       ↓                                                         │
│  CPU: Read output logits                                        │
│                                                                 │
│  Total: 2 sync points per forward pass                          │
└─────────────────────────────────────────────────────────────────┘
```

### Metal Kernels Required

| Kernel                 | Complexity | Lines of MSL | Status       |
| ---------------------- | ---------- | ------------ | ------------ |
| `matmul_vec_q4k`       | High       | ~150         | ✅ Done      |
| `matmul_vec_fp16`      | Medium     | ~80          | To implement |
| `layer_norm`           | Low        | ~40          | To implement |
| `group_norm`           | Medium     | ~60          | To implement |
| `token_shift`          | Low        | ~30          | To implement |
| `add` / `mul` / `blit` | Low        | ~20 each     | To implement |
| `time_mix_v7`          | High       | ~200         | To implement |
| `channel_mix_v7`       | Medium     | ~80          | To implement |
| `pack_kvakk`           | Low        | ~40          | To implement |
| `time_first_v7`        | Medium     | ~100         | To implement |
| `control_k_v7`         | Low        | ~30          | To implement |
| `l2_norm`              | Low        | ~30          | To implement |
| `add_activate`         | Low        | ~30          | To implement |
| `lerp`                 | Low        | ~20          | To implement |
| `affine`               | Low        | ~20          | To implement |

**Total: ~15 kernels, ~1000 lines of MSL**

---

## Implementation Phases

### Phase 1: Simple Ops (Foundation)

Port the simple element-wise and reduction operations first. These are building blocks.

#### 1.1 Element-wise Ops

```metal
// add.metal - Add two tensors
kernel void add(
    device const half* a [[buffer(0)]],
    device half* b [[buffer(1)]],  // in-place: b = a + b
    uint tid [[thread_position_in_grid]]
) {
    b[tid] = a[tid] + b[tid];
}

// mul.metal - Multiply two tensors
kernel void mul(
    device const half* a [[buffer(0)]],
    device half* b [[buffer(1)]],  // in-place: b = a * b
    uint tid [[thread_position_in_grid]]
) {
    b[tid] = a[tid] * b[tid];
}

// blit.metal - Copy tensor
kernel void blit(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    dst[tid] = src[tid];
}

// affine.metal - Scale and shift: y = a*x + b
kernel void affine(
    device half* x [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant float& shift [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    x[tid] = half(float(x[tid]) * scale + shift);
}
```

#### 1.2 Layer Norm

```metal
kernel void layer_norm(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup processes one token (n elements)
    uint base = tgid * n;

    // Compute mean using simd reduction
    float sum = 0.0f;
    for (uint i = lane; i < n; i += 32) {
        sum += float(input[base + i]);
    }
    sum = simd_sum(sum);
    float mean = sum / float(n);

    // Compute variance
    float var_sum = 0.0f;
    for (uint i = lane; i < n; i += 32) {
        float diff = float(input[base + i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = simd_sum(var_sum);
    float inv_std = rsqrt(var_sum / float(n) + eps);

    // Normalize and apply affine transform
    for (uint i = lane; i < n; i += 32) {
        float x = float(input[base + i]);
        float y = (x - mean) * inv_std;
        output[base + i] = half(y * float(weight[i]) + float(bias[i]));
    }
}
```

#### 1.3 Token Shift

```metal
kernel void token_shift(
    device const uint* cursors [[buffer(0)]],
    device const half* mix [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const half* input [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& num_emb [[buffer(5)]],
    constant uint& num_token [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;

    if (elem >= num_emb || token >= num_token) return;

    uint idx = elem + token * num_emb;

    // Determine if new sequence
    uint cursor = token > 0 ? cursors[token - 1] : 0;
    bool new_seq = (token == 0) || (cursor != cursors[token]);

    float prev_x;
    if (new_seq) {
        prev_x = state[elem];
    } else {
        prev_x = float(input[elem + (token - 1) * num_emb]);
    }

    // Update state for last token
    if (token == num_token - 1) {
        state[elem] = float(input[idx]);
    }

    // Channel mix: x = prev_x + v
    output[idx] = half(prev_x + float(mix[idx]));
}
```

### Phase 2: FP16 Matmul

The LoRA matrices (w1, w2, a1, a2, g1, g2, v1, v2) are FP16, not Q4K. Need a fast FP16 matmul.

```metal
kernel void matmul_vec_fp16(
    device const half* weights [[buffer(0)]],  // [K, M]
    device const half* input [[buffer(1)]],    // [K]
    device half* output [[buffer(2)]],         // [M]
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup handles 4 output rows
    uint row = tgid * 4 + simd_group;
    if (row >= M) return;

    float sum = 0.0f;

    // Each thread accumulates K/32 elements
    for (uint k = simd_lane; k < K; k += 32) {
        sum += float(weights[row * K + k]) * float(input[k]);
    }

    // Reduce across simdgroup
    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[row] = half(sum);
    }
}
```

### Phase 3: Time Mix V7 (The Complex One)

This is the most complex kernel. It implements the RWKV-7 time mixing with:

-   State management across tokens
-   Exponential decay computation
-   Multi-head attention-like operations

```metal
// time_mix_v7.metal
// This kernel processes the recurrent state update for RWKV-7

kernel void time_mix_v7(
    device const uint* cursors [[buffer(0)]],
    device float* state [[buffer(1)]],           // [num_emb, head_size+1, batch]
    device const half* r [[buffer(2)]],          // receptance [head_size, num_head, num_token]
    device const half* w [[buffer(3)]],          // time decay [head_size, num_head, num_token]
    device const half* n [[buffer(4)]],          // packed k,v,a,kk [head_size, num_head, num_token, 4]
    device half* x [[buffer(5)]],                // output [head_size, num_head, num_token]
    constant uint& head_size [[buffer(6)]],
    constant uint& num_head [[buffer(7)]],
    constant uint& num_token [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint h = tid.x;  // head index
    uint i = tid.y;  // position in head
    uint t = tid.z;  // token index

    if (h >= num_head || i >= head_size || t >= num_token) return;

    uint idx = i + h * head_size + t * head_size * num_head;

    // Load current inputs
    float r_val = float(r[idx]);
    float w_val = float(w[idx]);
    float k_val = float(n[idx]);
    float v_val = float(n[idx + head_size * num_head * num_token]);
    float a_val = float(n[idx + 2 * head_size * num_head * num_token]);
    float kk_val = float(n[idx + 3 * head_size * num_head * num_token]);

    // Get state indices
    uint state_idx = i + h * (head_size + 1);

    // Determine if this is a new sequence
    uint cursor = t > 0 ? cursors[t - 1] : 0;
    bool new_seq = (t == 0) || (cursor != cursors[t]);

    // Load or initialize state
    float s;
    if (new_seq) {
        s = state[state_idx];
    } else {
        // State was updated by previous token (need threadgroup sync for this)
        s = state[state_idx];
    }

    // Time decay
    float decay = exp(-exp(w_val));

    // Update state: s = decay * s + k * v
    float new_s = decay * s + k_val * v_val;

    // Output: x = r * s
    float out = r_val * new_s;

    // Store updated state
    state[state_idx] = new_s;

    // Store output
    x[idx] = half(out);
}
```

**Note:** The actual time_mix_v7 is more complex with additional terms. This is a simplified version. The full implementation needs to match `src/shaders/time_mix_v7.wgsl` exactly.

### Phase 4: Channel Mix V7

```metal
kernel void channel_mix_v7(
    device const uint* cursors [[buffer(0)]],
    device float* state [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* x [[buffer(3)]],
    constant uint& num_emb [[buffer(4)]],
    constant uint& num_token [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;

    if (elem >= num_emb || token >= num_token) return;

    uint idx = elem + token * num_emb;

    // Determine if new sequence
    uint cursor = token > 0 ? cursors[token - 1] : 0;
    bool new_seq = (token == 0) || (cursor != cursors[token]);

    float prev_x;
    if (new_seq) {
        prev_x = state[elem];
    } else {
        prev_x = float(x[elem + (token - 1) * num_emb]);
    }

    // Update state for last token
    if (token == num_token - 1) {
        state[elem] = float(x[idx]);
    }

    // Channel mix: x = prev_x + v
    x[idx] = half(prev_x + float(v[idx]));
}
```

### Phase 5: Integration - MetalLayerExecutor

Create a unified layer executor that runs all operations in sequence:

```rust
// src/metal/layer.rs

pub struct MetalLayerExecutor {
    ctx: Arc<MetalContext>,
    pipelines: MetalLayerPipelines,
}

pub struct MetalLayerPipelines {
    layer_norm: ComputePipelineState,
    token_shift: ComputePipelineState,
    matmul_vec_q4k: ComputePipelineState,
    matmul_vec_fp16: ComputePipelineState,
    time_mix_v7: ComputePipelineState,
    group_norm: ComputePipelineState,
    channel_mix_v7: ComputePipelineState,
    add: ComputePipelineState,
    mul: ComputePipelineState,
    blit: ComputePipelineState,
    // ... other pipelines
}

impl MetalLayerExecutor {
    /// Execute a complete forward pass through all layers
    pub fn forward(
        &self,
        input: &Buffer,      // Embedded input [num_emb, num_token]
        state: &Buffer,      // RNN state
        weights: &LayerWeights,
        output: &Buffer,     // Output logits
    ) {
        let cmd = self.ctx.queue().new_command_buffer();

        // Embed layer norm
        self.encode_layer_norm(cmd, input, &weights.embed_ln);

        // Process each layer
        for layer_idx in 0..weights.num_layers {
            self.encode_layer(cmd, layer_idx, input, state, &weights.layers[layer_idx]);
        }

        // Head layer norm + matmul
        self.encode_layer_norm(cmd, input, &weights.head_ln);
        self.encode_matmul_fp16(cmd, input, &weights.head_w, output);

        cmd.commit();
        cmd.wait_until_completed();
    }

    fn encode_layer(
        &self,
        cmd: &CommandBuffer,
        layer_idx: usize,
        x: &Buffer,
        state: &Buffer,
        weights: &LayerWeightRefs,
    ) {
        // ATTENTION BLOCK
        self.encode_blit(cmd, x, &self.att_x);
        self.encode_layer_norm(cmd, &self.att_x, &weights.att_ln);

        // Token shifts (6 of them)
        for (mix, output) in &weights.att_token_shifts {
            self.encode_token_shift(cmd, mix, state, &self.att_x, output);
        }

        // Q4K matmuls: w_r, w_k, w_v
        self.encode_matmul_q4k(cmd, &weights.w_r, &self.att_rx, &self.att_r);
        self.encode_matmul_q4k(cmd, &weights.w_k, &self.att_kx, &self.att_k);
        self.encode_matmul_q4k(cmd, &weights.w_v, &self.att_vx, &self.att_v);

        // LoRA matmuls (FP16)
        self.encode_matmul_fp16_tanh(cmd, &weights.w1, &self.att_wx, &self.aux_w);
        self.encode_matmul_fp16(cmd, &weights.w2, &self.aux_w, &self.att_w);
        self.encode_add(cmd, &weights.w0, &self.att_w);

        // ... continue with remaining ops

        // Time mix
        self.encode_time_mix_v7(cmd, state, &self.att_r, &self.att_w, &self.att_n, &self.att_x);

        // Group norm
        self.encode_group_norm(cmd, &self.att_x, &weights.gn);

        // Output matmul
        self.encode_matmul_q4k(cmd, &weights.w_o, &self.att_x, &self.att_o);
        self.encode_add(cmd, &self.att_o, x);

        // FFN BLOCK
        self.encode_blit(cmd, x, &self.ffn_x);
        self.encode_layer_norm(cmd, &self.ffn_x, &weights.ffn_ln);
        self.encode_token_shift(cmd, &weights.ffn_x_k, state, &self.ffn_x, &self.ffn_kx);
        self.encode_matmul_q4k_squared_relu(cmd, &weights.ffn_k, &self.ffn_kx, &self.ffn_k);
        self.encode_matmul_q4k(cmd, &weights.ffn_v, &self.ffn_k, &self.ffn_v);
        self.encode_channel_mix_v7(cmd, state, &self.ffn_v, &self.ffn_x);
        self.encode_add(cmd, &self.ffn_x, x);

        // Rescale if needed
        if (layer_idx + 1) % rescale == 0 {
            self.encode_affine(cmd, x, 0.5, 0.0);
        }
    }
}
```

### Phase 6: Runtime Integration

Modify `src/runtime/v7.rs` to use Metal path:

```rust
impl<F: Float> Dispatcher<RnnJob> for Bundle<F> {
    fn dispatch(&self, seed: Self::Info) -> Result<RnnJob, RuntimeError> {
        let context = &self.model.context;

        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        if let Some(metal_executor) = context.metal_layer_executor() {
            return self.dispatch_metal(seed, metal_executor);
        }

        // Fall back to wgpu path
        self.dispatch_wgpu(seed)
    }

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    fn dispatch_metal(
        &self,
        seed: Self::Info,
        executor: &MetalLayerExecutor,
    ) -> Result<RnnJob, RuntimeError> {
        // Load embeddings (wgpu)
        let embed_cmds = self.encode_embed(&seed);
        context.queue.submit(embed_cmds);
        context.device.poll(Wait);  // SYNC 1

        // Execute all layers in Metal
        executor.forward(
            &self.buffer.input,
            &self.state,
            &self.model.weights,
            &self.header.head_o,
        );  // SYNC 2 (inside forward)

        Ok(RnnJob {
            commands: vec![],  // Already submitted
            output: self.header.head_o.clone(),
            ...
        })
    }
}
```

---

## File Structure

```
src/metal/
├── mod.rs              # Module exports
├── context.rs          # MetalContext (existing)
├── kernels.rs          # MSL kernel source strings
├── ops.rs              # Individual kernel dispatch (existing)
├── layer.rs            # MetalLayerExecutor (new)
├── pipelines.rs        # Pipeline state management (new)
└── buffer_bridge.rs    # wgpu↔Metal buffer sharing (existing)
```

### New Files to Create

1. **`src/metal/layer.rs`** - MetalLayerExecutor implementation
2. **`src/metal/pipelines.rs`** - Pipeline state caching
3. **`src/metal/kernels.rs`** - Extend with all new kernels

### Files to Modify

1. **`src/metal/mod.rs`** - Export new modules
2. **`src/context.rs`** - Add `metal_layer_executor()` method
3. **`src/runtime/v7.rs`** - Add Metal dispatch path

---

## Testing Strategy

### Unit Tests

Each kernel should have a unit test comparing Metal output to wgpu output:

```rust
#[test]
fn test_metal_layer_norm() {
    let context = create_test_context();
    let input = random_tensor([2560, 1]);
    let weight = random_tensor([2560]);
    let bias = random_tensor([2560]);

    // wgpu reference
    let wgpu_output = wgpu_layer_norm(&input, &weight, &bias);

    // Metal implementation
    let metal_output = metal_layer_norm(&input, &weight, &bias);

    assert_tensors_close(&wgpu_output, &metal_output, 1e-3);
}
```

### Integration Tests

Full layer comparison:

```rust
#[test]
fn test_metal_layer_matches_wgpu() {
    let model = load_test_model();
    let input = random_input();

    let wgpu_output = model.forward_wgpu(&input);
    let metal_output = model.forward_metal(&input);

    assert_tensors_close(&wgpu_output, &metal_output, 1e-2);
}
```

### Benchmark Validation

```bash
# Compare outputs
cargo run --release --features metal-backend --example validate_metal

# Benchmark
./benchmark.sh -t "Pure Metal" -c "Full Metal layer execution"
```

---

## Timeline

| Week | Milestone   | Deliverable                                      |
| ---- | ----------- | ------------------------------------------------ |
| 1    | Simple ops  | add, mul, blit, affine, layer_norm, token_shift  |
| 2    | FP16 matmul | matmul_vec_fp16, group_norm, l2_norm             |
| 3    | Time mix    | time_mix_v7, time_first_v7, pack_kvakk           |
| 4    | Channel mix | channel_mix_v7, control_k_v7, add_activate, lerp |
| 5    | Integration | MetalLayerExecutor, runtime integration          |
| 6    | Testing     | Unit tests, integration tests, benchmarks        |

---

## Success Metrics

| Metric           | Target                | Validation             |
| ---------------- | --------------------- | ---------------------- |
| Generation speed | >200 tok/s            | benchmark.sh           |
| Output accuracy  | <1e-2 max diff        | validate_metal example |
| Memory usage     | ≤ wgpu baseline       | Activity Monitor       |
| Stability        | No crashes in 1hr run | stress test            |

---

## Risks and Mitigations

| Risk                      | Impact | Mitigation                                     |
| ------------------------- | ------ | ---------------------------------------------- |
| Numerical precision drift | Medium | Use f32 for accumulation, validate each kernel |
| State management bugs     | High   | Extensive unit tests, compare with wgpu        |
| Memory leaks              | Medium | Use Instruments, careful buffer lifecycle      |
| Performance regression    | Low    | Benchmark each phase, profile hotspots         |

---

## Appendix: WGSL to MSL Translation Guide

| WGSL                              | MSL                                                       |
| --------------------------------- | --------------------------------------------------------- |
| `@compute @workgroup_size(x,y,z)` | `[[kernel]]` with `uint3 tid [[thread_position_in_grid]]` |
| `var<workgroup>`                  | `threadgroup`                                             |
| `workgroupBarrier()`              | `threadgroup_barrier(mem_flags::mem_threadgroup)`         |
| `subgroupAdd(x)`                  | `simd_sum(x)`                                             |
| `vec4<f16>`                       | `half4`                                                   |
| `unpack2x16float(x)`              | `as_type<half2>(x)`                                       |

---

## References

-   [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
-   [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
-   [RWKV-7 Architecture](https://github.com/BlinkDL/RWKV-LM)
-   Existing wgpu shaders in `src/shaders/`

# Pure Metal Layer Execution Plan

## Goal

Achieve 200+ tok/s on Apple Silicon by executing entire RWKV layers in pure Metal, eliminating wgpu/Metal synchronization overhead.

## Current State

-   **Baseline**: ~55 tok/s (WebGPU)
-   **Metal kernel isolated**: 207 tok/s (4.78x faster)
-   **Metal with per-op sync**: 10 tok/s (broken - sync overhead dominates)
-   **Metal kernels implemented**: Phases 1-4 complete (add, mul, blit, affine, layer_norm, group_norm, l2_norm, token_shift, matmul_vec_q4k, matmul_mat_q4k, time_mix_v7, time_first_v7, pack_kvakk, control_k_v7, channel_mix_v7, softmax)
-   **MetalLayerExecutor**: Created but not integrated

## Problem Analysis

The issue is **synchronization overhead**. Each Metal operation requires:

1. wgpu command buffer submission
2. CPU wait for wgpu completion
3. Metal command buffer creation
4. Metal execution
5. CPU wait for Metal completion
6. Resume wgpu

With ~50 ops per layer × 32 layers = 1600 sync points per token. At ~1ms per sync, this adds 1.6 seconds per token!

## Solution: Pure Metal Layer Execution

Execute an **entire layer** in a single Metal command buffer with **zero intermediate syncs**.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Inference Flow                          │
├─────────────────────────────────────────────────────────────┤
│  1. Embedding (wgpu) - one-time at start                    │
│  2. For each layer:                                         │
│     ├─ Sync wgpu → Metal (once)                             │
│     ├─ Execute ALL layer ops in Metal (no sync)             │
│     └─ Sync Metal → wgpu (once)                             │
│  3. Header (wgpu) - one-time at end                         │
└─────────────────────────────────────────────────────────────┘
```

### Sync Points

-   **Before**: 1600 syncs per token
-   **After**: 64 syncs per token (2 per layer × 32 layers)
-   **Reduction**: 25x fewer syncs

---

## Implementation Phases

### Phase 1: Metal Buffer Management (Day 1)

**Goal**: Create a unified buffer pool that both wgpu and Metal can access.

**Tasks**:

1. Create `MetalBufferPool` in `src/metal/buffer_pool.rs`

    - Pre-allocate all intermediate buffers as Metal buffers
    - Use `MTLResourceOptions::StorageModeShared` for CPU/GPU access
    - Map buffer IDs to Metal buffers

2. Modify `TensorGpu` to optionally use Metal-backed storage

    - Add `metal_buffer: Option<metal::Buffer>` field
    - Create `TensorGpu::new_metal()` constructor
    - Implement `as_metal_buffer()` accessor

3. Create buffer bridge for wgpu ↔ Metal
    - Extract underlying Metal buffer from wgpu buffer (already done in `buffer_bridge.rs`)
    - Ensure zero-copy sharing works

**Files to modify**:

-   `src/metal/buffer_pool.rs` (new)
-   `src/tensor/mod.rs`
-   `src/metal/buffer_bridge.rs`

### Phase 2: Layer Tensor Allocation (Day 1-2)

**Goal**: Allocate all layer intermediate tensors as Metal buffers.

**Tasks**:

1. Identify all intermediate buffers used in `dispatch_layer`:

    ```rust
    // Attention intermediates
    buffer.att_x, buffer.att_rx, buffer.att_wx, buffer.att_kx, buffer.att_vx
    buffer.att_ax, buffer.att_gx, buffer.att_r, buffer.att_k, buffer.att_v
    buffer.att_w, buffer.att_a, buffer.att_g, buffer.att_kk, buffer.att_vv
    buffer.att_n, buffer.att_o, buffer.att_v0
    buffer.aux_w, buffer.aux_a, buffer.aux_g, buffer.aux_v

    // FFN intermediates
    buffer.ffn_x, buffer.ffn_kx, buffer.ffn_k, buffer.ffn_v

    // Main state
    buffer.x
    ```

2. Create `MetalLayerBuffers` struct:

    ```rust
    pub struct MetalLayerBuffers {
        // All intermediate buffers as Metal buffers
        pub att_x: metal::Buffer,
        pub att_rx: metal::Buffer,
        // ... etc
    }
    ```

3. Allocate at model load time, not per-inference

**Files to modify**:

-   `src/metal/layer_buffers.rs` (new)
-   `src/runtime/v7.rs` (Buffer struct)

### Phase 3: Metal Layer Dispatcher (Day 2-3)

**Goal**: Create a Metal-native layer dispatcher that mirrors `dispatch_layer`.

**Tasks**:

1. Create `MetalLayerDispatcher` in `src/metal/layer_dispatcher.rs`:

    ```rust
    pub struct MetalLayerDispatcher {
        ctx: Arc<MetalContext>,
        executor: MetalLayerExecutor,
    }

    impl MetalLayerDispatcher {
        pub fn dispatch_layer(
            &self,
            cmd: &CommandBufferRef,
            layer: &MetalLayerWeights,
            buffers: &MetalLayerBuffers,
            state: &MetalLayerState,
            config: &LayerConfig,
        ) {
            // Encode ALL layer ops into single command buffer
            self.encode_attention(cmd, layer, buffers, state, config);
            self.encode_ffn(cmd, layer, buffers, state, config);
        }
    }
    ```

2. Implement `encode_attention`:

    ```rust
    fn encode_attention(&self, cmd: &CommandBufferRef, ...) {
        // 1. Layer norm
        self.executor.encode_layer_norm(cmd, &layer.att_ln_w, &layer.att_ln_b,
                                         &buffers.att_x, config.num_emb, 1);

        // 2. Token shifts (6x)
        self.executor.encode_token_shift(cmd, &layer.att_x_r, &state.att,
                                          &buffers.att_x, &buffers.att_rx, ...);
        // ... etc for x_w, x_k, x_v, x_a, x_g

        // 3. Linear projections (matmul)
        self.executor.encode_matmul_q4k(cmd, &layer.att_w_r, &buffers.att_rx,
                                         &buffers.att_r, k, m);
        // ... etc for w_k, w_v

        // 4. Adapt layers (w1/w2, a1/a2, g1/g2)
        // 5. Control operations
        // 6. Time mix
        // 7. Group norm
        // 8. Time first
        // 9. Gate multiply
        // 10. Output projection
        // 11. Residual add
    }
    ```

3. Implement `encode_ffn`:
    ```rust
    fn encode_ffn(&self, cmd: &CommandBufferRef, ...) {
        // 1. Layer norm
        // 2. Token shift
        // 3. Key projection with SquaredRelu
        // 4. Value projection (sparse)
        // 5. Channel mix
        // 6. Residual add
    }
    ```

**Files to modify**:

-   `src/metal/layer_dispatcher.rs` (new)
-   `src/metal/mod.rs`

### Phase 4: Weight Loading for Metal (Day 3)

**Goal**: Load model weights directly into Metal buffers.

**Tasks**:

1. Create `MetalLayerWeights` struct:

    ```rust
    pub struct MetalLayerWeights {
        // Attention weights
        pub att_ln_w: metal::Buffer,
        pub att_ln_b: metal::Buffer,
        pub att_x_r: metal::Buffer,  // token shift mix
        // ... all Q4K weight matrices
        pub att_w_r: metal::Buffer,  // Q4K blocks
        pub att_w_k: metal::Buffer,
        // ... etc
    }
    ```

2. Modify loader to create Metal buffers for Q4K weights:

    - Already partially done in `runtime/loader.rs:865-870`
    - Extend to ALL layer weights, not just Q4K

3. Create `MetalModel` struct that holds all Metal weights:
    ```rust
    pub struct MetalModel {
        pub layers: Vec<MetalLayerWeights>,
        pub embed: MetalEmbedWeights,
        pub head: MetalHeadWeights,
    }
    ```

**Files to modify**:

-   `src/metal/weights.rs` (new)
-   `src/runtime/loader.rs`

### Phase 5: Runtime Integration (Day 4)

**Goal**: Integrate Metal layer execution into the runtime.

**Tasks**:

1. Add Metal execution path to `Bundle::dispatch`:

    ```rust
    impl<F: Float> Dispatcher<RnnJob> for Bundle<F> {
        fn dispatch(&self, seed: Self::Info) -> Result<RnnJob, RuntimeError> {
            // Check if Metal is available and should be used
            #[cfg(all(target_os = "macos", feature = "metal-backend"))]
            if self.use_metal_layers {
                return self.dispatch_metal(seed);
            }

            // Fallback to wgpu path
            self.dispatch_wgpu(seed)
        }
    }
    ```

2. Implement `dispatch_metal`:

    ```rust
    fn dispatch_metal(&self, seed: Self::Info) -> Result<RnnJob, RuntimeError> {
        // 1. Embedding (wgpu)
        let embed_ops = self.encode_embedding();
        context.encode(&embed_ops);
        context.queue.submit(...);
        context.device.poll(Wait);  // Sync point 1

        // 2. Layers (pure Metal)
        let metal_dispatcher = &self.metal_dispatcher;
        for (index, layer) in self.metal_model.layers.iter().enumerate() {
            let cmd = metal_dispatcher.begin_command_buffer();
            metal_dispatcher.dispatch_layer(&cmd, layer, &buffers, &state, &config);
            cmd.commit();
            cmd.wait_until_completed();  // Sync point 2 (per layer)
        }

        // 3. Header (wgpu)
        let header_ops = self.encode_header();
        context.encode(&header_ops);
        // ... submit and return
    }
    ```

3. Handle state synchronization:
    - State tensors need to be accessible by both wgpu and Metal
    - Use shared storage mode for state buffers

**Files to modify**:

-   `src/runtime/v7.rs`
-   `src/runtime/mod.rs`

### Phase 6: Testing & Optimization (Day 5)

**Goal**: Verify correctness and optimize performance.

**Tasks**:

1. Create Metal-specific test:

    ```rust
    #[test]
    fn test_metal_layer_output() {
        // Run same input through wgpu and Metal paths
        // Compare outputs (should be identical within FP16 tolerance)
    }
    ```

2. Profile with Instruments:

    - Metal System Trace
    - Identify any remaining bottlenecks

3. Optimize thread group sizes:

    - Tune for M1/M2/M3 architectures
    - Use `recommendedMaxWorkingSetSize`

4. Add async execution:
    - Use `addCompletedHandler` instead of `waitUntilCompleted`
    - Pipeline layer execution

---

## Missing Metal Kernels

Review of `dispatch_layer` shows we need these additional kernels:

### Already Implemented ✅

-   `layer_norm` ✅
-   `token_shift` ✅
-   `matmul_vec_q4k` / `matmul_mat_q4k` ✅
-   `add` ✅
-   `mul` ✅
-   `blit` ✅
-   `affine` ✅
-   `group_norm` ✅
-   `l2_norm` ✅
-   `time_mix_v7` ✅
-   `time_first_v7` ✅
-   `pack_kvakk` ✅
-   `control_k_v7` ✅
-   `channel_mix_v7` ✅

### Need to Implement ❌

-   `add_activate` (add with activation) ❌
-   `lerp` (linear interpolation) ❌
-   `matmul with activation` (Tanh, Sigmoid, SquaredRelu) ❌
    -   Current matmul kernels don't support fused activation
    -   Need: `matmul_q4k_tanh`, `matmul_q4k_sigmoid`, `matmul_q4k_squared_relu`

### Implementation Priority

1. **High**: `add_activate`, `lerp` - simple kernels
2. **Medium**: Fused matmul+activation - performance optimization
3. **Low**: Can use separate activation pass as fallback

---

## Risk Mitigation

### Risk 1: Numerical Differences

-   **Mitigation**: Compare wgpu vs Metal outputs layer-by-layer
-   **Tolerance**: FP16 allows ~0.1% relative error

### Risk 2: State Corruption

-   **Mitigation**: Use explicit memory barriers between layers
-   **Test**: Run multi-turn conversations, check for drift

### Risk 3: Memory Pressure

-   **Mitigation**: Reuse buffers across layers (already done)
-   **Monitor**: Track `currentAllocatedSize` vs `recommendedMaxWorkingSetSize`

### Risk 4: Fallback Path

-   **Mitigation**: Keep wgpu path working, use feature flag
-   **Default**: Metal on macOS, wgpu elsewhere

---

## Success Metrics

| Metric              | Current | Target | Stretch |
| ------------------- | ------- | ------ | ------- |
| Generation tok/s    | 55      | 150    | 200+    |
| Prefill tok/s       | 55      | 200    | 400+    |
| Memory usage        | 2.5GB   | 2.5GB  | 2.0GB   |
| First token latency | 50ms    | 20ms   | 10ms    |

---

## Timeline

| Day | Phase                  | Deliverable                     |
| --- | ---------------------- | ------------------------------- |
| 1   | Buffer Management      | Metal buffer pool, wgpu bridge  |
| 1-2 | Layer Buffers          | MetalLayerBuffers allocation    |
| 2-3 | Layer Dispatcher       | Metal dispatch_layer equivalent |
| 3   | Weight Loading         | MetalLayerWeights, MetalModel   |
| 4   | Runtime Integration    | dispatch_metal in Bundle        |
| 5   | Testing & Optimization | Correctness tests, profiling    |

---

## Quick Start Implementation Order

For fastest path to working prototype:

1. **Start with single-token generation** (matmul_vec only)
2. **Implement missing kernels**: `add_activate`, `lerp`
3. **Create minimal MetalLayerDispatcher** with hardcoded buffer offsets
4. **Test one layer** in isolation
5. **Scale to all layers**
6. **Add prefill support** (matmul_mat)

This avoids the complexity of buffer management initially and proves the concept works.

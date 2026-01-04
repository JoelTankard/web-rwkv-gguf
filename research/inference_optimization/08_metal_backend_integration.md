# Metal Backend Integration

## Current State

Based on previous work (see memory), Metal backend for Q4K matmul is implemented but disabled:

-   MetalContext exists in `src/metal/`
-   Zero-copy wgpu→Metal buffer extraction works
-   Metal kernel is 4.78x faster than WebGPU in isolation (207 vs 44 tok/s)
-   **Problem**: Per-op sync between Metal/wgpu is too slow (10 tok/s vs 15 baseline)

Current state: Metal dispatch disabled in `src/tensor/matrix.rs:144-148`

## Problem

The synchronization overhead between WebGPU and Metal APIs negates the kernel speedup:

-   Each Metal dispatch requires waiting for wgpu to finish
-   Each wgpu dispatch after Metal requires waiting for Metal to finish
-   Result: 10 tok/s with Metal vs 15 tok/s pure WebGPU

## Optimization Ideas

### Idea 1: Pure Metal Layer Execution

Execute entire layer in Metal, sync only once per layer:

```rust
fn dispatch_layer_metal(
    layer: &Layer,
    buffers: &Runtime,
    state: &State,
) -> Result<(), MetalError> {
    // Single sync point: wait for wgpu to finish previous layer
    sync_wgpu_to_metal();

    // Execute ALL layer operations in Metal
    metal_context.encode(|encoder| {
        // Attention block
        encoder.dispatch_layer_norm(&layer.att_ln, &buffers.att_x);
        encoder.dispatch_token_shift_6(...);
        encoder.dispatch_matmul_q4k(&layer.att.w_r, ...);
        encoder.dispatch_matmul_q4k(&layer.att.w_k, ...);
        encoder.dispatch_matmul_q4k(&layer.att.w_v, ...);
        // ... all attention ops

        // FFN block
        encoder.dispatch_layer_norm(&layer.ffn_ln, &buffers.ffn_x);
        encoder.dispatch_matmul_q4k(&layer.ffn.w_k, ...);
        encoder.dispatch_matmul_q4k(&layer.ffn.w_v, ...);
        // ... all FFN ops
    });

    // Single sync point: wait for Metal to finish this layer
    sync_metal_to_wgpu();

    Ok(())
}
```

**Expected benefit**: Amortize sync cost over ~40 operations instead of per-op

### Idea 2: Metal Shared Events

Use MTLSharedEvent for fine-grained synchronization without CPU roundtrip:

```rust
struct MetalWgpuSync {
    shared_event: metal::SharedEvent,
    signal_value: AtomicU64,
}

impl MetalWgpuSync {
    fn signal_from_metal(&self, encoder: &metal::CommandEncoder) {
        let value = self.signal_value.fetch_add(1, Ordering::SeqCst);
        encoder.encode_signal_event(&self.shared_event, value);
    }

    fn wait_in_wgpu(&self, encoder: &wgpu::CommandEncoder) {
        let value = self.signal_value.load(Ordering::SeqCst);
        // wgpu-hal can wait on MTLSharedEvent
        encoder.wait_for_metal_event(&self.shared_event, value);
    }
}
```

**Benefit**: GPU-to-GPU sync without CPU involvement

### Idea 3: Batched Metal Execution

Collect all Metal ops, execute at end of forward pass:

```rust
struct DeferredMetalOps {
    ops: Vec<MetalOp>,
}

impl DeferredMetalOps {
    fn push(&mut self, op: MetalOp) {
        self.ops.push(op);
    }

    fn execute_all(&mut self, context: &MetalContext) {
        if self.ops.is_empty() {
            return;
        }

        // Single command buffer for all deferred ops
        let command_buffer = context.queue.command_buffer();
        let encoder = command_buffer.compute_command_encoder();

        for op in self.ops.drain(..) {
            op.encode(encoder);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}
```

**Problem**: This breaks data dependencies - Metal ops need results from wgpu ops.

### Idea 4: Full Metal Forward Pass

Implement entire V7 forward pass in Metal, only use wgpu for non-matmul ops:

```rust
#[cfg(target_os = "macos")]
fn dispatch_v7_metal(
    model: &Model,
    input: &TensorGpu<f16>,
    state: &State,
) -> Result<TensorGpu<f32>, MetalError> {
    let metal = MetalContext::get();

    // Transfer input to Metal once
    let metal_input = metal.import_wgpu_buffer(&input.buffer);
    let metal_state = metal.import_wgpu_buffer(&state.data.buffer);

    // Execute entire forward pass in Metal
    let output = metal.execute_v7_forward(
        &model.metal_weights,
        metal_input,
        metal_state,
    )?;

    // Transfer output back to wgpu once
    let wgpu_output = metal.export_to_wgpu(&output);

    Ok(wgpu_output)
}
```

**Benefit**: Zero sync overhead during forward pass
**Cost**: Duplicate implementation of all ops in Metal

### Idea 5: Hybrid Execution with Op Fusion

Use Metal only for fused operations that benefit most:

```rust
enum ExecutionBackend {
    WebGPU,
    Metal,
    Fused(FusedOp),
}

struct FusedOp {
    // Metal kernel that fuses multiple operations
    kernel: metal::ComputePipelineState,
    // Operations that are fused
    ops: Vec<OpType>,
}

// Only use Metal for operations where fusion provides >2x speedup
fn choose_backend(ops: &[TensorOp]) -> ExecutionBackend {
    if can_fuse_to_metal(ops) && estimated_speedup(ops) > 2.0 {
        ExecutionBackend::Fused(create_fused_metal_kernel(ops))
    } else {
        ExecutionBackend::WebGPU
    }
}
```

### Idea 6: Async Metal Execution

Run Metal operations asynchronously, overlap with wgpu:

```rust
struct AsyncMetalExecutor {
    command_queue: metal::CommandQueue,
    in_flight: VecDeque<metal::CommandBuffer>,
    max_in_flight: usize,
}

impl AsyncMetalExecutor {
    fn submit(&mut self, ops: Vec<MetalOp>) {
        // Wait if too many in flight
        while self.in_flight.len() >= self.max_in_flight {
            self.in_flight.pop_front().unwrap().wait_until_completed();
        }

        let cmd = self.encode_ops(ops);
        cmd.commit();  // Don't wait
        self.in_flight.push_back(cmd);
    }

    fn sync(&mut self) {
        for cmd in self.in_flight.drain(..) {
            cmd.wait_until_completed();
        }
    }
}
```

## Estimated Impact

| Approach           | Speedup vs WebGPU | Sync Overhead | Complexity |
| ------------------ | ----------------- | ------------- | ---------- |
| Pure Metal Layer   | 2-3x              | Low           | High       |
| Shared Events      | 1.5-2x            | Very Low      | Medium     |
| Batched Execution  | N/A (breaks deps) | N/A           | N/A        |
| Full Metal Forward | 3-4x              | Minimal       | Very High  |
| Hybrid Fusion      | 1.5-2x            | Medium        | High       |
| Async Execution    | 1.3-1.5x          | Medium        | Medium     |

## Recommended Experiment

1. **First**: Pure Metal Layer - test if sync-per-layer is acceptable
2. **Second**: Shared Events - if per-layer sync still too slow
3. **Long-term**: Full Metal Forward for maximum performance

## Implementation Notes

The existing Metal infrastructure includes:

-   `src/metal/context.rs`: MetalContext with buffer cache
-   `src/metal/v7_layer.rs`: Partial V7 layer implementation
-   Zero-copy buffer sharing via wgpu-hal

Key challenge: Maintaining correctness while minimizing sync points.

## Files to Modify

-   `src/metal/v7_layer.rs`: Complete layer implementation
-   `src/metal/sync.rs`: Shared event synchronization (new)
-   `src/runtime/v7.rs`: Metal dispatch path
-   `src/tensor/ops.rs`: `TensorOp::MetalV7Layer` variant

# State Management Optimization

## Current Implementation

Location: `src/runtime/v7.rs:145-279` (`State`)

The RWKV state is stored as:

```rust
pub struct State {
    pub context: Context,
    pub info: ModelInfo,
    pub data: Vec<TensorGpu<f32, ReadWrite>>,  // One tensor per layer
}
```

Each layer's state shape: `[num_emb, head_size + 2, num_batch, 1]`

State operations:

-   `att(layer)` - View into attention state
-   `ffn(layer)` - View into FFN state
-   `load(tensor, batch)` - Load state for a batch
-   `back(batch)` - Read state back to CPU

## Problem

1. **Fragmented storage**: One tensor per layer = many small buffers
2. **Per-layer state access**: Each `token_shift` and `time_mix` accesses state separately
3. **F32 storage**: State stored as F32, but could use F16 for most operations
4. **Batch dimension overhead**: State shape includes batch even for single-sequence inference

## Optimization Ideas

### Idea 1: Contiguous State Tensor

Store all layer states in a single contiguous buffer:

```rust
pub struct ContiguousState {
    pub context: Context,
    pub info: ModelInfo,
    // Single tensor: [num_emb, head_size + 2, num_layer, num_batch]
    pub data: TensorGpu<f32, ReadWrite>,
}

impl ContiguousState {
    fn att(&self, layer: usize) -> TensorGpuView<f32> {
        let head_size = self.info.num_emb / self.info.num_head;
        self.data.view(.., 0..head_size + 1, layer, ..)
    }

    fn ffn(&self, layer: usize) -> TensorGpuView<f32> {
        let head_size = self.info.num_emb / self.info.num_head;
        self.data.view(.., head_size + 1, layer, ..)
    }
}
```

**Benefits:**

-   Single buffer allocation
-   Better memory locality
-   Simpler GPU→CPU transfer for full state

### Idea 2: F16 State with F32 Accumulation

Store state in F16, convert to F32 only when needed:

```rust
pub struct MixedPrecisionState {
    // Main state in F16 (half the memory)
    pub data: TensorGpu<f16, ReadWrite>,
    // Temporary F32 buffer for accumulation
    pub accumulator: TensorGpu<f32, ReadWrite>,
}

impl MixedPrecisionState {
    fn time_mix_update(&self, layer: usize, ...) -> TensorOp {
        // Load F16 state → F32 accumulator
        // Perform time mix in F32
        // Store back to F16 state
        TensorOp::List(vec![
            TensorOp::blit_f16_to_f32(&self.data, &self.accumulator)?,
            TensorOp::time_mix_v7(..., &self.accumulator, ...)?,
            TensorOp::blit_f32_to_f16(&self.accumulator, &self.data)?,
        ])
    }
}
```

**Benefits:**

-   50% memory reduction for state
-   May improve cache utilization
-   Minimal precision loss for RNN state

### Idea 3: Lazy State Materialization

Only materialize state when needed (for branching/saving):

```rust
pub struct LazyState {
    // GPU state (always up to date)
    gpu: TensorGpu<f32, ReadWrite>,
    // CPU cache (may be stale)
    cpu_cache: Option<TensorCpu<f32>>,
    // Dirty flag
    gpu_dirty: bool,
}

impl LazyState {
    fn mark_dirty(&mut self) {
        self.gpu_dirty = true;
        self.cpu_cache = None;
    }

    async fn get_cpu(&mut self) -> &TensorCpu<f32> {
        if self.cpu_cache.is_none() || self.gpu_dirty {
            self.cpu_cache = Some(self.gpu.back().await);
            self.gpu_dirty = false;
        }
        self.cpu_cache.as_ref().unwrap()
    }
}
```

**Benefits:**

-   Avoids unnecessary GPU→CPU transfers
-   State stays on GPU during generation

### Idea 4: State Compression

Compress state for long-context scenarios:

```rust
pub struct CompressedState {
    // Recent state (full precision)
    recent: TensorGpu<f32, ReadWrite>,
    // Historical state (compressed)
    historical: TensorGpu<u8, ReadWrite>,  // Quantized
    // Compression metadata
    scales: TensorGpu<f16, ReadWrite>,
}

impl CompressedState {
    fn compress_old_state(&mut self, threshold: usize) {
        // Quantize state older than threshold
        // Keep recent state at full precision
    }

    fn decompress_for_inference(&self) -> TensorGpu<f32, ReadWrite> {
        // Decompress on-the-fly during inference
    }
}
```

**Benefits:**

-   Enables longer context with same memory
-   Useful for very long conversations

### Idea 5: Batched State Updates

Batch all state updates into single operation:

```rust
// Current: separate state access per token_shift
TensorOp::token_shift(&cursors, &mix_r, state.att(layer)?, &x, &rx, true)?
TensorOp::token_shift(&cursors, &mix_w, state.att(layer)?, &x, &wx, true)?
// ... 6 more

// Optimized: single fused state update
TensorOp::fused_token_shift_6(
    &cursors,
    &[mix_r, mix_w, mix_k, mix_v, mix_a, mix_g],
    state.att(layer)?,
    &x,
    &[rx, wx, kx, vx, ax, gx],
)?
```

**Benefits:**

-   Single state read instead of 6
-   Better memory bandwidth utilization

### Idea 6: Ring Buffer State for Streaming

For streaming inference, use ring buffer to avoid copying:

```rust
pub struct RingBufferState {
    data: TensorGpu<f32, ReadWrite>,
    // Current position in ring buffer
    head: usize,
    // Capacity
    capacity: usize,
}

impl RingBufferState {
    fn push(&mut self, new_state: &TensorGpu<f32, ReadWrite>) {
        // Write to current head position
        let offset = self.head * self.state_size;
        // ... copy to offset
        self.head = (self.head + 1) % self.capacity;
    }

    fn get_recent(&self, n: usize) -> TensorGpuView<f32> {
        // Return view of last n states
    }
}
```

**Benefits:**

-   No memory movement for state history
-   Efficient for sliding window attention

## Estimated Impact

| Optimization         | Memory Savings | Speed Improvement | Complexity |
| -------------------- | -------------- | ----------------- | ---------- |
| Contiguous State     | 0%             | 5-10%             | Low        |
| F16 State            | 50%            | 0-5%              | Medium     |
| Lazy Materialization | 0%             | 10-20%            | Low        |
| State Compression    | 75%+           | -5% (overhead)    | High       |
| Batched Updates      | 0%             | 10-15%            | Medium     |
| Ring Buffer          | 0%             | 5%                | Medium     |

## Recommended Experiment

1. **First**: Contiguous state tensor - simple change, immediate benefit
2. **Second**: Batched state updates (combine with fused token shift)
3. **Third**: F16 state if memory is a concern

## Files to Modify

-   `src/runtime/v7.rs`: `State` struct and methods
-   `src/tensor/ops.rs`: Fused state operations
-   `src/shaders/token_shift.wgsl`: Batched version

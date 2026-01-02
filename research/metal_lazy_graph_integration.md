# Metal Integration with Lazy Tensor Graph

## Overview

This document outlines the plan for integrating native Metal execution with the lazy tensor graph system. The goal is to achieve 150-200+ tok/s on Apple Silicon by leveraging Metal's native Int8 SIMD operations while avoiding per-operation synchronization overhead.

## Current State

-   **WebGPU Int8**: ~10-15 tok/s (4x slower than F16 due to lack of native int8 ops)
-   **WebGPU F16**: ~40-55 tok/s
-   **Metal Int8 (isolated)**: ~207 tok/s (4.78x faster than WebGPU)
-   **Metal with per-op sync**: ~10 tok/s (sync overhead dominates)

## The Opportunity

The lazy tensor graph already batches operations before execution. This is the perfect integration point for Metal:

1. Build the compute graph lazily (already done)
2. At `resolve()` time, detect if we can use Metal
3. Compile the entire subgraph into a single Metal command buffer
4. Execute with minimal sync points

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Lazy Tensor Graph                          │
├─────────────────────────────────────────────────────────────┤
│  LazyOp::MatMul → LazyOp::Add → LazyOp::Activation          │
│       ↓              ↓              ↓                       │
│  [Fusion Pass: Detect patterns, create MatMulBias]          │
│       ↓                                                     │
│  [Metal Compilation Pass: Convert to Metal command buffer]  │
│       ↓                                                     │
│  [Single Metal Dispatch with one sync point]                │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase A1: Metal Context & Buffer Bridge (Current Focus)

**Goal**: Establish Metal context and zero-copy buffer sharing with wgpu.

**Files to create**:

-   `src/metal/mod.rs` - Module root
-   `src/metal/context.rs` - MetalContext with device, queue, pipeline cache
-   `src/metal/buffer_bridge.rs` - Extract Metal buffer from wgpu buffer

**Key code**:

```rust
pub struct MetalContext {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<String, metal::ComputePipelineState>,
}

impl MetalContext {
    /// Extract underlying Metal buffer from wgpu buffer (zero-copy)
    pub unsafe fn extract_metal_buffer(wgpu_buffer: &wgpu::Buffer) -> metal::Buffer {
        // Use wgpu-hal to get the underlying Metal buffer
        wgpu_buffer.as_hal::<wgpu::hal::api::Metal, _, _>(|hal_buffer| {
            hal_buffer.unwrap().raw_buffer().clone()
        })
    }
}
```

### Phase A2: Metal Int8 Matmul+Bias Kernel

**Goal**: Create optimized Metal kernel for fused Int8 matmul with bias.

**File**: `src/metal/shaders/matmul_int8_bias.metal`

```metal
#include <metal_stdlib>
using namespace metal;

kernel void matmul_int8_bias(
    device const int8_t* matrix [[buffer(0)]],      // Quantized weights
    device const half* scales [[buffer(1)]],        // Per-block scales
    device const half* input [[buffer(2)]],         // Input tensor
    device const half* bias [[buffer(3)]],          // Bias vector
    device half* output [[buffer(4)]],              // Output tensor
    constant uint4& shape [[buffer(5)]],            // [K, M, T, B]
    uint3 gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Use simdgroup_matrix for hardware-accelerated 8x8 matmul
    simdgroup_matrix<half, 8, 8> acc = simdgroup_matrix<half, 8, 8>(0.0h);

    // Dequantize int8 weights on-the-fly
    // Accumulate using SIMD matrix ops
    // Add bias
    // Apply activation (fused)

    // Write output
}
```

**Key optimizations**:

-   `simdgroup_matrix` for 8x8 hardware matmul (4x faster than manual)
-   Fused dequantization + matmul + bias + activation
-   Coalesced memory access patterns
-   Threadgroup memory for weight tiles

### Phase A3: Lazy Graph Metal Dispatch

**Goal**: Add Metal execution path to `ComputeGraph::resolve()`.

**Modify**: `src/tensor/graph.rs`

```rust
impl ComputeGraph {
    pub fn resolve(
        &mut self,
        target: NodeIndex,
        context: &Context,
    ) -> Result<Arc<Buffer>, TensorError> {
        // ... existing cache check ...

        // Apply graph optimizations
        self.optimize();

        // Check if Metal dispatch is beneficial
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.should_use_metal(target) {
            return self.resolve_metal(target, context);
        }

        // Fall back to WebGPU path
        self.resolve_wgpu(target, context)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn should_use_metal(&self, target: NodeIndex) -> bool {
        // Use Metal if:
        // 1. Graph contains Int8 matmul operations
        // 2. Graph is large enough to amortize sync overhead
        // 3. All operations have Metal implementations
        self.contains_int8_matmul(target) && self.node_count(target) >= 3
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn resolve_metal(
        &mut self,
        target: NodeIndex,
        context: &Context,
    ) -> Result<Arc<Buffer>, TensorError> {
        let metal_ctx = context.metal_context()?;
        let execution_order = self.topological_sort(target);

        // Create single Metal command buffer for entire graph
        let cmd = metal_ctx.queue.new_command_buffer();

        for node_idx in execution_order {
            self.encode_metal_op(cmd, node_idx, metal_ctx)?;
        }

        cmd.commit();
        cmd.wait_until_completed();

        // Return result buffer
        self.get(target)
            .and_then(|n| n.buffer.clone())
            .ok_or_else(|| TensorErrorKind::Deduce.into())
    }
}
```

### Phase A4: Matrix Type Support

**Goal**: Support Int8 matrices in the fused matmul+bias operation.

**Modify**: `src/tensor/ops.rs`

```rust
impl TensorOp {
    pub fn matmul_int8_bias<'a, 'b, 'c, F0: Float, F1: Float>(
        matrix: &TensorGpu<u8, ReadWrite>,
        scales: &TensorGpu<f16, ReadWrite>,
        input: impl Into<TensorGpuView<'a, F0>>,
        bias: impl Into<TensorGpuView<'b, f16>>,
        output: impl Into<TensorGpuView<'c, F1>>,
        act: Activation,
    ) -> Result<Self, TensorError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            // Use Metal kernel for Int8
            return Self::matmul_int8_bias_metal(matrix, scales, input, bias, output, act);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            // Fall back to WebGPU (slower)
            return Self::matmul_int8_bias_wgpu(matrix, scales, input, bias, output, act);
        }
    }
}
```

---

## Phase B: Pure Metal Layer Execution (Future)

This is the full solution for maximum performance. See `pure_metal_layer.md` for details.

### B1: Metal Buffer Pool

-   Pre-allocate all intermediate buffers as Metal buffers
-   Use `MTLResourceOptions::StorageModeShared` for CPU/GPU access
-   Zero-copy sharing with wgpu

### B2: Metal Layer Weights

-   Load Q4K/Int8 weights directly into Metal buffers
-   Create `MetalLayerWeights` struct for each layer

### B3: Metal Layer Dispatcher

-   Mirror `dispatch_layer` in pure Metal
-   Encode ALL layer ops into single command buffer
-   No intermediate syncs

### B4: Runtime Integration

-   Add `dispatch_metal` path to `Bundle::dispatch`
-   Sync only at layer boundaries (64 syncs vs 1600 per token)

### Expected Results

| Implementation             | Sync Points/Token | Expected tok/s |
| -------------------------- | ----------------- | -------------- |
| WebGPU (current)           | N/A               | 40-55          |
| Metal per-op               | ~1600             | 10             |
| Metal lazy graph (Phase A) | ~50-100           | 80-120         |
| Metal pure layer (Phase B) | ~64               | 150-200+       |

---

## Dependencies

```toml
[target.'cfg(target_os = "macos")'.dependencies]
metal = "0.27"
objc = "0.2"
```

## Feature Flag

```toml
[features]
metal = ["metal", "objc"]
```

## Testing Strategy

1. **Correctness**: Compare Metal vs WebGPU outputs (FP16 tolerance ~0.1%)
2. **Performance**: Benchmark tok/s with `examples/bench_metal_lazy.rs`
3. **Memory**: Monitor `currentAllocatedSize` vs `recommendedMaxWorkingSetSize`

## References

-   [metal-rs crate](https://github.com/gfx-rs/metal-rs)
-   [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
-   [llama.cpp Metal implementation](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-metal.m)
-   [wgpu HAL Metal access](https://docs.rs/wgpu/latest/wgpu/struct.Buffer.html#method.as_hal)

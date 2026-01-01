# Metal Backend Integration Guide

## Current State

The Metal backend is already implemented in `src/metal/`:

-   `context.rs` - Device, queue, and pipeline management
-   `kernels.rs` - MSL kernels for Q4K matmul with simdgroup parallelism
-   `ops.rs` - Rust API for dispatching Metal operations

**Benchmark results:**

-   Batched: 24.7 µs/matmul, 531 GFLOPS, 211 tok/s
-   WebGPU baseline: 44.2 tok/s
-   **Speedup: 4.78x**

## Integration Steps

### Step 1: Add Metal Context to Runtime

Modify `src/context.rs` to optionally hold a Metal context:

```rust
pub struct Context {
    // Existing wgpu fields
    adapter: Arc<Adapter>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    // New: Optional Metal context
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    metal: Option<Arc<crate::metal::MetalContext>>,
}

impl Context {
    pub fn new(adapter: Adapter) -> Self {
        let metal_ctx = {
            #[cfg(all(target_os = "macos", feature = "metal-backend"))]
            { crate::metal::MetalContext::new().map(Arc::new) }
            #[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
            { None }
        };

        Self {
            // ... existing fields
            #[cfg(all(target_os = "macos", feature = "metal-backend"))]
            metal: metal_ctx,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    pub fn metal(&self) -> Option<&Arc<crate::metal::MetalContext>> {
        self.metal.as_ref()
    }
}
```

### Step 2: Create Metal Buffers for Q4K Weights

When loading Q4K weights, create corresponding Metal buffers:

```rust
// In src/tensor/matrix.rs or src/runtime/loader.rs

pub struct Q4KMatrix {
    // Existing wgpu buffer
    pub wgpu_buffer: TensorGpu<u8, ReadWrite>,

    // Metal buffer (shares same data on macOS)
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    pub metal_buffer: Option<metal::Buffer>,
}

impl Q4KMatrix {
    pub fn new(context: &Context, data: &[u8], shape: Shape) -> Self {
        let wgpu_buffer = context.tensor_from_data(shape, data.to_vec());

        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        let metal_buffer = context.metal().map(|ctx| {
            ctx.create_buffer(data)
        });

        Self {
            wgpu_buffer,
            #[cfg(all(target_os = "macos", feature = "metal-backend"))]
            metal_buffer,
        }
    }
}
```

### Step 3: Modify Matrix::Q4K Dispatch

Update `src/tensor/matrix.rs` to use Metal when available:

```rust
impl Matrix {
    pub fn matmul_vec_op<'a>(&self, /* params */) -> TensorOp {
        match self {
            Matrix::Q4K { w, s } => {
                #[cfg(all(target_os = "macos", feature = "metal-backend"))]
                if let Some(metal_ctx) = context.metal() {
                    if let Some(metal_weights) = w.metal_buffer.as_ref() {
                        return TensorOp::MetalMatmulQ4K {
                            ctx: metal_ctx.clone(),
                            weights: metal_weights.clone(),
                            input: input.clone(),
                            output: output.clone(),
                            k, m,
                        };
                    }
                }

                // Fallback to WebGPU
                TensorOp::matmul_vec_q4k(/* existing params */)
            }
            // ... other variants
        }
    }
}
```

### Step 4: Add Metal TensorOp Variant

Add a new variant to `TensorOp` for Metal operations:

```rust
pub enum TensorOp {
    // Existing variants...

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    MetalMatmulQ4K {
        ctx: Arc<MetalContext>,
        weights: metal::Buffer,
        input: TensorGpu<f32, ReadWrite>,
        output: TensorGpu<f32, ReadWrite>,
        k: u32,
        m: u32,
    },
}
```

### Step 5: Implement Command Batching

The key to performance is batching. Modify the inference loop:

```rust
// In src/runtime/infer/mod.rs or similar

pub struct InferenceContext {
    context: Arc<Context>,

    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    metal_batch: Option<MetalCommandBatch>,
}

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
struct MetalCommandBatch {
    command_buffer: metal::CommandBuffer,
}

impl InferenceContext {
    pub fn begin_token(&mut self) {
        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        if let Some(metal) = self.context.metal() {
            self.metal_batch = Some(MetalCommandBatch {
                command_buffer: metal.queue().new_command_buffer().to_owned(),
            });
        }
    }

    pub fn end_token(&mut self) {
        #[cfg(all(target_os = "macos", feature = "metal-backend"))]
        if let Some(batch) = self.metal_batch.take() {
            batch.command_buffer.commit();
            batch.command_buffer.wait_until_completed();
        }
    }
}
```

## Testing the Integration

### Unit Test

```rust
#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[test]
fn test_metal_matmul_correctness() {
    // Create test data
    let k = 256;
    let m = 64;

    // Run with Metal
    let metal_result = run_metal_matmul(weights, input, k, m);

    // Run with WebGPU
    let wgpu_result = run_wgpu_matmul(weights, input, k, m);

    // Compare results (allow small floating point differences)
    assert_results_close(&metal_result, &wgpu_result, 1e-4);
}
```

### Benchmark Validation

```bash
# Before integration
./benchmark.sh -t "Pre-Metal" -c "WebGPU only" -f metal_integration

# After integration
./benchmark.sh -t "Post-Metal" -c "Metal backend enabled" -f metal_integration
```

## Troubleshooting

### Metal Context Fails to Initialize

Check that:

1. Running on macOS
2. `metal-backend` feature is enabled
3. Metal is available (not in VM without GPU passthrough)

### Performance Not Improved

Check that:

1. Commands are being batched (not sync per op)
2. Metal buffers are being used (not falling back to WebGPU)
3. Add logging to verify Metal path is taken

### Numerical Differences

Q4K dequantization should produce identical results. If not:

1. Check scale/min extraction logic matches WGSL shader
2. Verify byte ordering (endianness)
3. Compare intermediate values step by step

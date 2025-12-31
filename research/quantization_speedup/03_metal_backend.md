# Option 3: Native Metal Backend (Bypass WebGPU)

## The WebGPU Overhead Problem

WebGPU is a **portable abstraction** over native APIs:

-   On macOS: WebGPU → wgpu → Metal
-   On Windows: WebGPU → wgpu → Vulkan/DX12
-   On Linux: WebGPU → wgpu → Vulkan

Each layer adds overhead and limits access to hardware-specific features.

## Why llama.cpp is Faster

llama.cpp uses **native Metal directly**:

-   Direct access to Metal Performance Shaders (MPS)
-   Metal-specific optimizations (simdgroup operations)
-   No abstraction overhead
-   Access to Apple's optimized matrix multiply

### Metal-Specific Features Not Available in WebGPU

1. **Simdgroup Matrix Operations** (`simdgroup_matrix`)

    - Hardware-accelerated 8x8 matrix multiply
    - ~4x faster than manual implementation

2. **Threadgroup Memory Barriers** (fine-grained)

    - More efficient synchronization

3. **Texture Memory** for weights

    - Better cache utilization for read-only data

4. **Async Copy** (`threadgroup_async_copy`)
    - Overlap compute with memory transfer

## Implementation Approaches

### Approach A: Conditional Metal Backend

Add a compile-time feature for native Metal:

```rust
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal_backend {
    use metal::*;
    // Native Metal implementation
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
mod metal_backend {
    // Fallback to WebGPU
}
```

**Pros**: Best performance on Mac
**Cons**: Platform-specific code, maintenance burden

### Approach B: Use metal-rs Crate

The `metal` crate provides Rust bindings to Metal:

```rust
use metal::*;

let device = Device::system_default().unwrap();
let command_queue = device.new_command_queue();

// Create compute pipeline with optimized kernel
let kernel = include_str!("shaders/matmul_q4k.metal");
let library = device.new_library_with_source(kernel, &CompileOptions::new()).unwrap();
```

### Approach C: Hybrid wgpu + Metal

Use wgpu for most operations, but call Metal directly for critical paths:

```rust
// Get underlying Metal device from wgpu
let metal_device = unsafe { wgpu_device.as_hal::<wgpu::hal::api::Metal>() };

// Use Metal for quantized matmul
metal_matmul_q4k(metal_device, weights, input, output);
```

## Performance Expectations

Based on llama.cpp benchmarks on M2 Max:

| Implementation           | Generation Speed |
| ------------------------ | ---------------- |
| WebGPU (current)         | 44 tok/s         |
| Native Metal (estimated) | 80-100 tok/s     |

**Potential improvement: 1.8-2.3x**

## Metal Shader Example

```metal
#include <metal_stdlib>
using namespace metal;

kernel void matmul_q4k(
    device const uint8_t* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Use simdgroup_matrix for 8x8 matrix multiply
    simdgroup_matrix<half, 8, 8> acc;
    simdgroup_matrix<half, 8, 8> a_mat;
    simdgroup_matrix<half, 8, 8> b_mat;

    // Dequantize Q4K weights into a_mat
    // ...

    // Hardware-accelerated matrix multiply
    simdgroup_multiply_accumulate(acc, a_mat, b_mat, acc);
}
```

## Challenges

1. **Code duplication**: Need separate Metal and WebGPU shaders
2. **Testing complexity**: Different code paths per platform
3. **wgpu HAL access**: Requires unsafe code and unstable APIs
4. **Build complexity**: Metal SDK only on macOS

## Recommendation

**Priority: High for macOS-only deployment**

If the primary target is macOS/iOS, native Metal is the fastest path to 2x speedup.
However, this sacrifices cross-platform compatibility.

Consider this if:

-   Primary users are on Apple Silicon
-   Performance is critical
-   Cross-platform is secondary concern

## References

-   [metal-rs crate](https://github.com/gfx-rs/metal-rs)
-   [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
-   [llama.cpp Metal implementation](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-metal.m)

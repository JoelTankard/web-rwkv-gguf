# Option 7: WebGPU Limitations and Workarounds

## WebGPU vs Native API Comparison

| Feature                      | Metal | CUDA              | WebGPU            |
| ---------------------------- | ----- | ----------------- | ----------------- |
| Simdgroup matrix ops         | ✅    | ✅ (Tensor Cores) | ❌                |
| Subgroup operations          | ✅    | ✅                | ⚠️ (experimental) |
| Async memory copy            | ✅    | ✅                | ❌                |
| Texture sampling for compute | ✅    | ✅                | ✅                |
| Shared memory atomics        | ✅    | ✅                | ✅                |
| FP16 compute                 | ✅    | ✅                | ⚠️ (SHADER_F16)   |
| INT8 dot product             | ✅    | ✅                | ❌                |

## Key Missing Features

### 1. No Native Quantized MatMul

Metal has `simdgroup_matrix` for hardware-accelerated 8x8 matrix multiply.
CUDA has Tensor Cores for mixed-precision matmul.
WebGPU has... nothing. We must emulate with scalar/vector ops.

**Impact**: 2-4x slower than native implementations.

### 2. Limited Subgroup Support

WebGPU subgroups are experimental and limited:

-   Not all operations available
-   No guarantee of subgroup size
-   May not work on all backends

**Workaround**: Use workgroup shared memory instead (slower).

### 3. No Async Memory Operations

Metal/CUDA can overlap memory transfers with compute.
WebGPU compute shaders are synchronous within a dispatch.

**Workaround**: Split work into multiple dispatches with barriers.

### 4. No INT8 Dot Product

Modern GPUs have INT8 dot product instructions (4x throughput vs FP32).
WebGPU doesn't expose these.

**Workaround**: Use FP16 or emulate INT8 (loses the benefit).

## Potential Solutions

### Solution A: wgpu HAL Escape Hatch

wgpu allows accessing the underlying backend:

```rust
#[cfg(target_os = "macos")]
unsafe {
    device.as_hal::<wgpu::hal::api::Metal, _, _>(|metal_device| {
        // Direct Metal calls here
    });
}
```

**Pros**: Access to all Metal features
**Cons**: Unsafe, platform-specific, may break

### Solution B: Custom wgpu Backend

Fork wgpu and add quantized matmul as a custom operation:

```rust
// In wgpu fork
impl Device {
    pub fn create_quantized_matmul_pipeline(&self, config: QuantConfig) -> Pipeline {
        // Use Metal simdgroup_matrix internally
    }
}
```

**Pros**: Clean API, full optimization
**Cons**: Maintenance burden, diverges from upstream

### Solution C: Hybrid Execution

Use WebGPU for most ops, shell out to native for critical paths:

```rust
fn matmul_q4k(weights: &[u8], input: &[f16], output: &mut [f16]) {
    #[cfg(target_os = "macos")]
    {
        metal_matmul_q4k(weights, input, output);
        return;
    }

    // Fallback to WebGPU
    webgpu_matmul_q4k(weights, input, output);
}
```

**Pros**: Best of both worlds
**Cons**: Complex, two code paths

### Solution D: WebGPU Extensions

Wait for/propose WebGPU extensions:

-   `chromium_experimental_subgroup_matrix` (Intel)
-   Potential future quantized matmul extension

**Pros**: Standards-based, portable
**Cons**: Slow process, uncertain timeline

## Comparison with kalosm/Fusor

Fusor (kalosm's new backend) uses WebGPU but with:

-   Kernel fusion compiler (reduces dispatch overhead)
-   Custom optimized kernels
-   Still limited by WebGPU capabilities

kalosm's current speed advantage likely comes from:

1. Using llama.cpp's Metal backend (not WebGPU)
2. More mature optimization
3. Different model architecture (Transformer vs RWKV)

## Realistic Expectations

Given WebGPU limitations, achievable speedup vs current:

| Optimization   | Speedup      | Notes               |
| -------------- | ------------ | ------------------- |
| Better shaders | 1.3-1.5x     | Shader optimization |
| Memory layout  | 1.2-1.4x     | Coalesced access    |
| Subgroups      | 1.1-1.3x     | If available        |
| **Combined**   | **1.5-2.0x** | Best case           |

To exceed 2x, likely need native Metal backend.

## Recommendation

**Short term (1-2 weeks):**

-   Optimize shaders (02_shader_optimization.md)
-   Fix memory layout (06_memory_layout.md)
-   Target: 60-70 tok/s

**Medium term (1-2 months):**

-   Implement native Metal backend for macOS
-   Keep WebGPU for cross-platform
-   Target: 80-100 tok/s on Metal

**Long term:**

-   Contribute to WebGPU quantized matmul proposals
-   Monitor Fusor development
-   Consider WASM SIMD for CPU fallback

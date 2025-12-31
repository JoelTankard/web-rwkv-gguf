# GPU Abstraction Alternatives Comparison

This document compares all viable alternatives for GPU compute in Rust, focusing on performance and cross-platform compatibility.

## Overview of Options

### 1. wgpu (Current)

**What it is:** Safe, cross-platform graphics API based on WebGPU standard.

**Architecture:**

```
wgpu (safe Rust API)
    └── wgpu-core (validation, resource tracking)
            └── wgpu-hal (unsafe HAL)
                    └── Native APIs (Vulkan/Metal/DX12/GL)
```

**Pros:**

-   Safe API with comprehensive validation
-   Cross-platform including web (WebGPU/WebGL2)
-   Actively maintained by Mozilla/gfx-rs team
-   Large community and ecosystem
-   Used in Firefox, Deno, and many production apps

**Cons:**

-   Validation overhead (typically 5-15% CPU-side)
-   Some WebGPU limitations vs native APIs
-   Shader language limited to WGSL (or SPIR-V with naga)

**Best for:** Most applications, especially those needing web support.

---

### 2. wgpu-hal

**What it is:** The unsafe hardware abstraction layer that powers wgpu.

**Architecture:**

```
Your Code (unsafe)
    └── wgpu-hal (minimal validation)
            └── Native APIs (Vulkan/Metal/DX12/GL)
```

**Pros:**

-   Same codebase as wgpu (proven, maintained)
-   Minimal overhead (no validation layer)
-   Cross-platform (Vulkan/Metal/DX12/GL)
-   WebGPU-aligned API design
-   Can mix with wgpu in same application

**Cons:**

-   Unsafe API - you handle all validation
-   No web support (wgpu-hal doesn't compile to WASM)
-   Less documentation than wgpu
-   Must manage resource lifetimes manually
-   Explicit barriers required

**Best for:** Native-only applications needing maximum performance with cross-platform support.

---

### 3. ash (Vulkan Bindings)

**What it is:** Low-level, zero-overhead Vulkan bindings for Rust.

**Architecture:**

```
Your Code (unsafe)
    └── ash (thin Vulkan wrapper)
            └── Vulkan Driver
```

**Pros:**

-   Zero abstraction overhead
-   Full Vulkan feature access
-   Excellent for Vulkan-specific optimizations
-   Used by wgpu-hal internally

**Cons:**

-   Vulkan-only (no Metal, no DX12)
-   Extremely verbose API
-   Manual memory management
-   No cross-platform without MoltenVK/DXVK
-   Steep learning curve

**Best for:** Vulkan-specific applications, game engines with dedicated graphics teams.

---

### 4. vulkano

**What it is:** Safe Vulkan wrapper with Rust idioms.

**Architecture:**

```
Your Code (safe Rust)
    └── vulkano (validation + Rust safety)
            └── Vulkan Driver
```

**Pros:**

-   Safer than raw Vulkan
-   Rust-idiomatic API
-   Compile-time shader validation
-   Good documentation

**Cons:**

-   Vulkan-only
-   Some overhead for safety
-   Smaller community than wgpu
-   No web support
-   Requires shaderc for shader compilation

**Best for:** Vulkan-focused projects wanting more safety than ash.

---

### 5. metal-rs

**What it is:** Rust bindings for Apple's Metal API.

**Architecture:**

```
Your Code (unsafe)
    └── metal-rs
            └── Metal Framework
```

**Pros:**

-   Direct Metal access
-   Zero abstraction overhead
-   Full Metal feature set

**Cons:**

-   Apple platforms only
-   Unsafe API
-   No cross-platform

**Best for:** Apple-exclusive applications.

---

## Performance Comparison

### Theoretical Overhead

| Layer       | CPU Overhead | GPU Overhead |
| ----------- | ------------ | ------------ |
| wgpu        | 5-15%        | 0%           |
| wgpu-hal    | 1-3%         | 0%           |
| ash/vulkano | 0-1%         | 0%           |
| metal-rs    | 0-1%         | 0%           |

**Key insight:** GPU shader execution time is identical across all options. The overhead is purely CPU-side API call processing.

### For RWKV Inference Specifically

The web-rwkv project is **GPU compute bound**, not CPU API bound:

```
Typical inference breakdown:
├── GPU Compute (shader execution): 85-95%
├── Memory transfers: 3-10%
└── CPU API overhead: 2-5%
```

Reducing the 2-5% CPU overhead by half would yield **1-2.5% total improvement** - not worth the migration cost and loss of web support.

---

## Cross-Platform Support Matrix

| Option   | Windows | macOS | Linux | iOS | Android | Web |
| -------- | ------- | ----- | ----- | --- | ------- | --- |
| wgpu     | ✅      | ✅    | ✅    | ✅  | ✅      | ✅  |
| wgpu-hal | ✅      | ✅    | ✅    | ✅  | ✅      | ❌  |
| ash      | ✅      | ⚠️¹   | ✅    | ❌  | ✅      | ❌  |
| vulkano  | ✅      | ⚠️¹   | ✅    | ❌  | ✅      | ❌  |
| metal-rs | ❌      | ✅    | ❌    | ✅  | ❌      | ❌  |

¹ Requires MoltenVK translation layer

---

## Migration Effort Estimate

| From wgpu To  | Effort                 | Risk      | Benefit              |
| ------------- | ---------------------- | --------- | -------------------- |
| wgpu-hal      | High (2-4 weeks)       | Medium    | 5-10% CPU reduction  |
| ash           | Very High (1-2 months) | High      | 10-15% CPU reduction |
| vulkano       | High (2-4 weeks)       | Medium    | 10-15% CPU reduction |
| Multi-backend | Extreme (2-3 months)   | Very High | Platform-optimal     |

---

## Recommendation Summary

For web-rwkv-gguf:

1. **Keep wgpu** - Best balance of features, safety, and cross-platform support
2. **Consider wgpu-hal** only if:
    - Web support is dropped
    - Profiling shows significant CPU API overhead
    - Team has capacity for unsafe code maintenance
3. **Avoid ash/vulkano** - Loss of cross-platform not justified by minimal gains

# Executive Summary: gfx-hal Replacement Research

## TL;DR

**gfx-hal is deprecated and should NOT be used as a replacement for wgpu.**

The gfx-hal crate has been in maintenance mode since v0.9 (2021) and is no longer actively developed. The gfx-rs team has moved all development to `wgpu-hal`, which is the spiritual successor and the HAL layer that powers wgpu itself.

## Key Findings

### 1. gfx-hal Status: DEPRECATED

-   **Maintenance mode since v0.9** (announced in [gfx-rs/gfx#3768](https://github.com/gfx-rs/gfx/discussions/3768))
-   No new features or significant updates
-   Development effort moved to wgpu-hal
-   Not recommended for new projects

### 2. Recommended Alternatives (Ranked by Practicality)

| Option             | Cross-Platform | Web Support      | Performance | Migration Effort | Recommendation           |
| ------------------ | -------------- | ---------------- | ----------- | ---------------- | ------------------------ |
| **wgpu (current)** | ✅ Full        | ✅ WebGPU/WebGL2 | Good        | None             | Keep for now             |
| **wgpu-hal**       | ✅ Full        | ❌ No            | Better      | High             | Consider for native-only |
| **ash (Vulkan)**   | ⚠️ Vulkan only | ❌ No            | Best        | Very High        | Not recommended          |
| **vulkano**        | ⚠️ Vulkan only | ❌ No            | Best        | Very High        | Not recommended          |
| **gfx-hal**        | ✅ Full        | ⚠️ WebGL only    | N/A         | N/A              | **DEPRECATED**           |

### 3. Performance Reality Check

The performance overhead of wgpu over raw Vulkan/Metal is typically **5-15%** for CPU-side API calls, but this is rarely the bottleneck in GPU compute workloads like RWKV inference. The actual compute shader execution time is identical across all abstractions.

**For this project specifically:**

-   Bottleneck is GPU compute (shader execution), not API overhead
-   wgpu's validation overhead is negligible for compute-heavy workloads
-   Cross-platform support (including web) is a key feature to preserve

## Recommendation

**Stay with wgpu** for the following reasons:

1. **gfx-hal is dead** - No active development, no future
2. **wgpu-hal loses web support** - Critical for this project's WebGPU target
3. **Performance gains are minimal** - GPU compute time dominates, not API overhead
4. **Migration cost is high** - Unsafe APIs require extensive validation code
5. **wgpu is actively maintained** - Regular updates, bug fixes, new features

If native-only performance becomes critical in the future, consider:

1. **wgpu-hal** for a middle-ground approach (same ecosystem, less overhead)
2. **Platform-specific backends** for maximum performance (ash/metal crates directly)

See the detailed documents in this folder for full analysis.

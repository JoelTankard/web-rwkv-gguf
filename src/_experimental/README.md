# Experimental Code

This directory contains experimental implementations that are **not actively maintained**.

## Metal Backend (`metal/`)

Native Metal backend for accelerated quantized operations on macOS.

**Status:** Disabled by default. The WebGPU native quantized shaders (Q4K, Q5K, Q6K, Q8_0) now achieve comparable or better performance without the synchronization overhead.

**Why it exists:** Metal's `simdgroup_matrix` operations can achieve ~200 tok/s for Int8/Q4K matmul in isolation. However, per-operation synchronization between Metal and wgpu caused a net slowdown (~10 tok/s vs 15 tok/s baseline).

**Potential future work:**

-   Pure Metal layer execution (sync once per layer instead of per-op)
-   Metal shared events for fine-grained sync
-   Batched Metal execution after all wgpu ops

**To enable:** Build with `--features metal-acceleration`

**Files:**

-   `context.rs` - MetalContext with buffer cache
-   `buffer_bridge.rs` - Zero-copy wgpu→Metal buffer extraction
-   `kernels.rs` - Metal shader sources for Int8/Q4K matmul
-   `ops.rs` - FFN execution via Metal
-   `v7_layer.rs` - Full V7 layer execution via Metal
-   `v7_kernels.rs` - V7-specific Metal kernels

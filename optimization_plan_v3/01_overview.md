# Hybrid GPU Backend Optimization Plan

## Goal

Maximize inference speed while maintaining cross-platform compatibility and browser support.

**Priority order:**

1. Inference speed (most important)
2. Cross-platform support (native: macOS, Windows, Linux)
3. Browser support (WebGPU in Chrome, Firefox, Safari)

## Current State

-   **WebGPU baseline**: 44.2 tok/s on Apple M2 Max
-   **Metal kernel (batched)**: 211 tok/s (4.78x faster)
-   **Target**: 90+ tok/s on Apple Silicon

## Strategy: Hybrid Backend with Native Acceleration

```
┌─────────────────────────────────────────────────────────────┐
│                    web-rwkv Runtime                         │
├─────────────────────────────────────────────────────────────┤
│                   GPU Backend Router                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Metal     │  │   CUDA      │  │      WebGPU         │ │
│  │  (macOS)    │  │  (NVIDIA)   │  │  (Cross-platform)   │ │
│  │  4.78x      │  │  Future     │  │     Fallback        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Principles

1. **Detect GPU at runtime** - Query adapter info to determine optimal backend
2. **Native kernels for hot paths** - Use Metal/CUDA for Q4K matmul (80%+ of compute)
3. **WebGPU for everything else** - Keep non-critical ops in WebGPU for simplicity
4. **Graceful fallback** - If native backend unavailable, use WebGPU
5. **Same API surface** - User code doesn't change based on backend

## Documents in This Plan

1. `01_overview.md` - This document
2. `02_architecture.md` - Detailed architecture and GPU routing
3. `03_metal_integration.md` - Metal backend integration guide
4. `04_cuda_integration.md` - Future CUDA backend (placeholder)
5. `05_webgpu_baseline.md` - WebGPU optimizations and fallback
6. `06_benchmarking.md` - How to benchmark and validate changes
7. `07_implementation_phases.md` - Step-by-step implementation plan

## Quick Start: Benchmarking

Always benchmark before and after changes:

```bash
# Run benchmark with interactive prompts
./benchmark.sh

# Run with specific options
./benchmark.sh -t "Baseline" -c "Before Metal integration" -f metal_comparison

# Full options
./benchmark.sh --help
```

Results are saved to `benchmarks/<filename>.md` with:

-   Load time
-   Prefill speed (tok/s)
-   Generation speed (tok/s)
-   Quality hash (for regression detection)

# Implementation Phases

## Phase 1: Metal Backend Integration (1-2 weeks)

**Goal:** Integrate existing Metal kernel into inference pipeline

### Tasks

1. **Add Metal context to Context struct**

    - File: `src/context.rs`
    - Create Metal context alongside wgpu context
    - Expose via `context.metal()` method

2. **Create Metal buffers during model loading**

    - File: `src/runtime/loader.rs`
    - When loading Q4K weights, also create Metal buffer
    - Store both buffers in `Matrix::Q4K`

3. **Modify Matrix::Q4K dispatch**

    - File: `src/tensor/matrix.rs`
    - Check for Metal availability
    - Route to Metal kernel when available

4. **Implement command batching**
    - File: `src/runtime/infer/mod.rs`
    - Create command buffer at start of token
    - Encode all matmuls into single buffer
    - Commit and wait at end of token

### Validation

```bash
# Before
./benchmark.sh -t "Pre-Metal" -c "WebGPU only" -f phase1

# After
./benchmark.sh -t "Post-Metal" -c "Metal integrated" -f phase1
```

**Success criteria:** Generation speed > 90 tok/s on Apple Silicon

---

## Phase 2: Buffer Sharing Optimization (1 week)

**Goal:** Eliminate buffer copies between wgpu and Metal

### Tasks

1. **Investigate wgpu-hal Metal access**

    - Can we get raw Metal buffer from wgpu buffer?
    - If yes, avoid duplicate buffer creation

2. **Implement shared buffer abstraction**

    - Single buffer accessible by both backends
    - No copy overhead

3. **Profile memory usage**
    - Ensure we're not doubling memory consumption

### Validation

```bash
./benchmark.sh -t "Shared Buffers" -c "Zero-copy Metal access" -f phase2
```

**Success criteria:** No memory regression, same or better speed

---

## Phase 3: Attention Kernel Optimization (1-2 weeks)

**Goal:** Accelerate attention computation with Metal

### Tasks

1. **Profile attention bottleneck**

    - Identify which attention ops are slowest
    - Measure potential improvement

2. **Implement Metal attention kernel**

    - File: `src/metal/kernels.rs`
    - Add attention kernel with simdgroup ops

3. **Integrate with time_mix dispatch**
    - File: `src/runtime/infer/v7.rs`
    - Route attention to Metal when available

### Validation

```bash
./benchmark.sh -t "Metal Attention" -c "Attention accelerated" -f phase3
```

**Success criteria:** Additional 10-20% speedup

---

## Phase 4: CUDA Backend (Future, 2-3 weeks)

**Goal:** Add NVIDIA GPU acceleration

### Tasks

1. **Add CUDA dependencies**

    - `cuda-rs` or `cudarc` crate
    - Feature flag: `cuda-backend`

2. **Port Q4K kernel to CUDA**

    - Translate MSL kernel to CUDA
    - Use warp-level primitives

3. **GPU detection for NVIDIA**
    - Detect NVIDIA GPUs via adapter info
    - Route to CUDA backend

### Validation

Test on Windows/Linux with NVIDIA GPU

**Success criteria:** 2x+ speedup over WebGPU on NVIDIA

---

## Phase 5: WebGPU Optimizations (Ongoing)

**Goal:** Improve fallback performance

### Tasks

1. **Subgroup operations**

    - Use when available
    - Fallback shader for unsupported GPUs

2. **F16 shaders**

    - Reduce memory bandwidth
    - Use when SHADER_F16 available

3. **Workgroup size tuning**
    - Auto-tune based on GPU

### Validation

Test on various GPUs and browsers

---

## Timeline Summary

| Phase                   | Duration  | Target Speedup        |
| ----------------------- | --------- | --------------------- |
| 1. Metal Integration    | 1-2 weeks | 4-5x on Apple Silicon |
| 2. Buffer Sharing       | 1 week    | Memory optimization   |
| 3. Metal Attention      | 1-2 weeks | +10-20%               |
| 4. CUDA Backend         | 2-3 weeks | 2x+ on NVIDIA         |
| 5. WebGPU Optimizations | Ongoing   | Incremental           |

## Quick Reference

### Run Benchmark

```bash
./benchmark.sh -t "Title" -c "Description" -f output_file
```

### Test Metal Kernel

```bash
cargo run --release --features metal-backend --example bench_metal_vs_wgpu_real
```

### Build Without Metal

```bash
cargo build --release  # WebGPU only
```

### Build With Metal

```bash
cargo build --release --features metal-backend
```

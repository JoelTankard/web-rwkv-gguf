# Model Loading Optimization

## Current State

| Backend           | Load Time | Technique                     |
| ----------------- | --------- | ----------------------------- |
| WebGPU (web-rwkv) | **~4.6s** | Rayon parallel dequantization |
| llama.cpp         | ~250ms    | Memory-mapped files (mmap)    |

### Optimizations Applied

1. **Parallel dequantization with rayon** - Q4_K and Q6_K dequantization uses multiple CPU cores

### Attempted but Reverted

1. **Native Q4_K/Q5_K/Q6_K loading** - Caused inference quality issues (NaN in output)
2. **Head weight native Q6_K** - Also caused inference issues

### Current Performance

| Metric       | Value              |
| ------------ | ------------------ |
| Load Time    | ~4.6s              |
| Prefill      | ~147 tok/s         |
| Generation   | ~40.5 tok/s        |
| Quality Hash | 3895ad4add71cff0 ✓ |

### Remaining Bottleneck

The embed tensor (65536 × 2560 = 167M elements) must be dequantized from Q4_K to F16 for CPU-side token lookup during inference. This is the primary bottleneck.

## Detailed Profiling Results

**Model:** rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf (32 layers, 2560 embed)  
**GPU:** Apple M2 Max  
**Profiling tool:** `cargo run --release --example profile_loading`

### Bottleneck Breakdown

| Phase                  | Time (ms) | % of Total | Notes                                              |
| ---------------------- | --------- | ---------- | -------------------------------------------------- |
| **FFN matrices**       | 2500.9    | 44.6%      | `ffn.key.weight` + `ffn.value.weight` × 32 layers  |
| **ATT large matrices** | 1113.7    | 19.9%      | `key`, `value`, `receptance`, `output` × 32 layers |
| **Head weight (GPU)**  | 920.4     | 16.4%      | Single large matrix (65536 × 2560)                 |
| **Embed weight (CPU)** | 698.6     | 12.5%      | CPU-side F16 conversion, kept on CPU               |
| ATT small matrices     | 132.6     | 2.4%       | `w1`, `w2`, `a1`, `a2`, `g1`, `g2`, `v1`, `v2`     |
| GPU sync #1            | 94.3      | 1.7%       | Wait for embed/head upload                         |
| Context create         | 69.0      | 1.2%       | wgpu adapter/device creation                       |
| ATT layer norms        | 9.5       | 0.2%       | Small vectors                                      |
| ATT vectors            | 4.8       | 0.1%       | `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g`           |
| Parse header           | 3.8       | 0.1%       | GGUF header parsing                                |
| Other                  | ~2.0      | <0.1%      | File open, mmap, FFN vectors/LN                    |
| **TOTAL**              | 5601.9    | 100%       |                                                    |

### Per-Layer Breakdown (sample)

| Layer | ATT LN | ATT vec | ATT small mat | ATT large mat | FFN LN | FFN vec | FFN mat | Total   |
| ----- | ------ | ------- | ------------- | ------------- | ------ | ------- | ------- | ------- |
| 0     | 0.1ms  | 0.3ms   | 5.8ms         | 43.3ms        | 0.1ms  | 0.0ms   | 94.8ms  | 144.3ms |
| 1     | 0.1ms  | 0.1ms   | 3.8ms         | 33.9ms        | 0.1ms  | 0.0ms   | 80.7ms  | 118.7ms |
| 2     | 0.1ms  | 0.2ms   | 4.5ms         | 33.6ms        | 0.1ms  | 0.0ms   | 77.8ms  | 116.2ms |

### Root Cause Analysis

The bottleneck is **NOT** GPU upload or shader compilation. It's **CPU-side tensor processing**:

1. **Q4_K dequantization** - Converting Q4_K blocks to F16 for GPU upload
2. **Memory allocation** - Creating intermediate `TensorCpu<f16>` buffers
3. **Data copying** - Multiple copies during tensor creation pipeline

The `load_matrix()` path for Q4_K tensors:

```markdown
GGUF Q4_K data → dequantize to F32 → convert to F16 → TensorCpu → GPU upload
```

For native Q4_K path (already implemented but not used for all tensors):

```markdown
GGUF Q4_K data → direct GPU upload (no CPU conversion)
```

### Why Native Q4_K Loading is Faster

The codebase already has native Q4_K loading (`Matrix::Q4K`) that bypasses dequantization:

-   Uploads raw Q4_K blocks directly to GPU
-   Dequantizes on-the-fly during matmul shader execution
-   Should be ~10x faster for loading

**Current issue:** The native path may not be triggering for all matrices, or there's overhead in the fallback path.

## Why llama.cpp Loads 40x Faster

### 1. Memory-Mapped Files (mmap)

llama.cpp uses `mmap()` to map the GGUF file directly into virtual memory:

```c
// Simplified llama.cpp approach
void* model_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
```

Benefits:

-   OS handles paging - only loads data when accessed
-   No explicit read() calls needed
-   Kernel manages caching efficiently
-   Near-instant "load" time (just sets up page tables)

### 2. No Upfront Shader Compilation

llama.cpp on Metal:

-   Uses pre-compiled Metal shaders bundled with the library
-   Shader compilation happens once during library build
-   Runtime just loads compiled shader binaries

WebGPU (web-rwkv):

-   Compiles WGSL shaders at runtime
-   Each shader variant must be compiled
-   GPU driver optimization adds latency

### 3. Lazy Weight Loading

llama.cpp defers actual weight loading:

-   Weights are only read from disk when first accessed
-   GPU layers are populated on-demand
-   First inference is slightly slower, but load is instant

## Implementation

### Primary: Incremental GPU Buffer Population

Allocate GPU buffers lazily, populate during first inference. This is the main optimization.

```rust
struct LazyBuffer {
    cpu_data: Option<Arc<[f32]>>,  // Backed by mmap
    gpu_buffer: Option<Buffer>,
}

impl LazyBuffer {
    fn ensure_gpu(&mut self, device: &Device) -> &Buffer {
        if self.gpu_buffer.is_none() {
            let data = self.cpu_data.take().unwrap();
            self.gpu_buffer = Some(device.create_buffer_init(&data));
        }
        self.gpu_buffer.as_ref().unwrap()
    }
}
```

**Benefits:**

-   Only loads layers actually used
-   Spreads GPU transfer cost over time
-   Compatible with mmap approach

**Estimated Impact:** Combined with mmap, achieves ~500ms load time

### Backup: Memory-Mapped Model Files

**Concept:** Use mmap in Rust layer, stream weights to GPU.

```rust
// In web-rwkv Rust code
use memmap2::MmapOptions;

fn load_model_mmap(path: &Path) -> Result<Model> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // Parse header only (fast)
    let header = parse_header(&mmap[0..HEADER_SIZE]);

    // Weights are accessed lazily via mmap
    Model::from_mmap(mmap, header)
}
```

Use if incremental loading alone is insufficient. Provides foundation for lazy loading.

**Estimated Impact:** Reduces load time to ~1-2s (combined with incremental loading for ~500ms)

## Files to Modify

-   `src/model/loader.rs` - Add LazyBuffer and mmap support
-   `src/model/run.rs` - Update inference to call `ensure_gpu()` on first access

## References

-   [llama.cpp mmap implementation](https://github.com/ggerganov/llama.cpp/blob/master/llama.cpp#L1234)
-   [WebGPU shader compilation](https://www.w3.org/TR/webgpu/#shader-module-creation)
-   [Rust memmap2 crate](https://docs.rs/memmap2/latest/memmap2/)

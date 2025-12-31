# Model Loading Optimization

## Current State

| Backend           | Load Time | Technique                      |
| ----------------- | --------- | ------------------------------ |
| WebGPU (web-rwkv) | ~10s      | Full load + shader compilation |
| llama.cpp         | ~250ms    | Memory-mapped files (mmap)     |

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

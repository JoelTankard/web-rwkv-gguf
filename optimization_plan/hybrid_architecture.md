# Enhanced web-rwkv Architecture

## Goal

Create a single, optimized inference backend that combines:

-   **Fast tokenization** (307µs) - keep existing Rust tokenizer
-   **Fast generation** (30+ tok/s) - keep existing WebGPU WKV shaders
-   **Fast model loading** (<500ms) - port mmap + lazy loading from llama.cpp
-   **Fast state/embedding ops** (<400ms) - add dedicated embedding API

**Constraints:**

-   Single model in memory (~2GB for 2.9B model)
-   No external dependencies (no llama.cpp runtime)
-   Contribute improvements upstream to web-rwkv

## Target Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Enhanced web-rwkv Interface                     │
├─────────────────────────────────────────────────────────────────┤
│  load()  │  encode()  │  generate()  │  embed()  │  state()    │
└────┬─────┴─────┬──────┴──────┬───────┴─────┬─────┴──────┬──────┘
     │           │             │             │            │
     ▼           ▼             ▼             ▼            ▼
┌─────────┐ ┌─────────┐  ┌──────────┐  ┌──────────┐ ┌──────────┐
│ mmap +  │ │ Rust    │  │ WebGPU   │  │ Dedicated│ │ Zero-copy│
│ lazy    │ │ tokeniz │  │ WKV GPU  │  │ embed()  │ │ numpy    │
│ loader  │ │ (fast)  │  │ shaders  │  │ API      │ │ arrays   │
└─────────┘ └─────────┘  └──────────┘  └──────────┘ └──────────┘
```

## Implementation: Enhanced web-rwkv

Port the fast techniques from llama.cpp into web-rwkv.

### 1. Incremental GPU Buffer Population (Primary)

Allocate GPU buffers lazily, populate during first inference. Combined with mmap for ~500ms load time.

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

### 2. Memory-Mapped Model Files (Backup)

Use mmap for model file access if incremental loading proves insufficient.

```rust
use memmap2::MmapOptions;

pub struct MmapModelLoader {
    mmap: Mmap,
    header: ModelHeader,
}

impl MmapModelLoader {
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let header = parse_header(&mmap)?;
        Ok(Self { mmap, header })
    }

    pub fn load_layer(&self, layer_idx: usize) -> LayerWeights {
        let offset = self.header.layer_offsets[layer_idx];
        let size = self.header.layer_sizes[layer_idx];
        LayerWeights::from_bytes(&self.mmap[offset..offset+size])
    }
}
```

### 3. Dedicated Embedding Extraction

```rust
impl Model {
    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        // Run forward pass but stop before lm_head
        let mut state = self.initial_state();
        for &token in tokens {
            state = self.forward_layer(token, state);
        }
        // Return hidden state directly (skip output projection)
        state.hidden.to_vec()
    }
}
```

## Benefits

-   **Single backend** - simpler architecture, easier maintenance
-   **Low memory** - ~2GB for 2.9B model (no duplication)
-   **No external dependencies** - pure Rust/WebGPU
-   **Upstream contribution** - benefits the broader RWKV community

## Development Approach

This work should be done in the `web-rwkv-gguf` repository to benefit the community:

1. Fork/clone `https://github.com/JoelTankard/web-rwkv-gguf`
2. Create feature branch for optimizations
3. Implement changes incrementally with benchmarks
4. Submit PRs upstream

## Memory Budget

| Component              | Memory   | Notes                      |
| ---------------------- | -------- | -------------------------- |
| Model weights (Q4_K_M) | ~1.8GB   | Memory-mapped, lazy loaded |
| GPU buffers            | ~200MB   | Active layers only         |
| State/embeddings       | ~50MB    | Reused buffers             |
| **Total**              | **~2GB** | Single model, optimized    |

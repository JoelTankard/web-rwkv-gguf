# Embedding Lookup Optimization

## Current Implementation

Location: `src/runtime/v7.rs:450-486` (`RnnJob::load`)

```rust
let stack: Vec<TensorCpu<f16>> = input
    .iter()
    .map(|chunk| {
        let num_emb = self.embed.shape()[0];
        let data = self.embed.data();
        let data = chunk
            .iter()
            .map(|token| match token {
                &Token::Token(token) => {
                    let start = num_emb * token as usize;
                    let end = start + num_emb;
                    let data = data[start..end].to_vec();  // CPU allocation per token
                    TensorCpu::from_data_1d(data)
                }
                Token::Embed(tensor) => tensor.clone(),
            })
            .collect_vec();
        // ...
    })
    .collect();
```

## Problem

1. **Per-token CPU allocation**: Each token lookup creates a new `Vec<f16>` via `.to_vec()`
2. **Sequential processing**: Tokens are processed one at a time
3. **Double copy**: Data is copied to CPU tensor, then uploaded to GPU

## Optimization Ideas

### Idea 1: GPU-Side Embedding Lookup

Move the embedding table to GPU and perform lookups there:

```rust
// Store embed on GPU instead of CPU
pub struct Embed {
    pub ln: LayerNorm,
    pub w: TensorGpu<f16, ReadWrite>,  // GPU tensor instead of TensorCpu
}

// New shader: embed_lookup.wgsl
// Input: token_ids [T], embed_table [V, E]
// Output: embeddings [E, T]
```

**Benefits:**

-   Eliminates CPU→GPU transfer per inference
-   Parallel lookup across all tokens
-   Memory bandwidth: GPU VRAM is much faster than system RAM

**Considerations:**

-   Embedding table must fit in VRAM (already does for inference)
-   Need new `TensorOp::embed_lookup` shader

### Idea 2: Pre-allocated Staging Buffer

Keep CPU path but eliminate allocations:

```rust
struct EmbedCache {
    staging: Vec<f16>,  // Pre-allocated buffer for max_tokens * num_emb
}

impl RnnJob {
    fn load(&self, input: &RnnChunk) -> Result<(), RuntimeError> {
        // Reuse staging buffer instead of allocating per token
        let staging = &mut self.embed_staging;
        for (i, token) in tokens.iter().enumerate() {
            let src = &embed_data[token * num_emb..(token + 1) * num_emb];
            staging[i * num_emb..(i + 1) * num_emb].copy_from_slice(src);
        }
        // Single upload
        self.input.load_from_slice(staging)?;
    }
}
```

**Benefits:**

-   Zero allocations during inference
-   Single contiguous upload to GPU
-   Simpler than GPU-side lookup

### Idea 3: Memory-Mapped Embedding with Direct Upload

Use `mmap` for embedding table and upload directly:

```rust
// During model load, keep embedding as mmap'd region
// During inference, create GPU buffer view directly from mmap'd data
```

## Estimated Impact

| Approach       | Latency Reduction | Memory Savings | Complexity |
| -------------- | ----------------- | -------------- | ---------- |
| GPU Lookup     | ~0.5-1ms/token    | High           | Medium     |
| Staging Buffer | ~0.2-0.5ms/token  | Medium         | Low        |
| Mmap Direct    | ~0.1-0.3ms/token  | Low            | Low        |

## Recommended Experiment

Start with **Idea 2 (Staging Buffer)** as it's lowest risk and provides immediate benefit. Measure baseline first:

```bash
# Benchmark current embedding lookup time
cargo run --release --example bench -- --model $MODEL --tokens 1 --measure-load
```

## Files to Modify

-   `src/runtime/v7.rs`: `RnnJob` struct and `load()` method
-   `src/tensor/ops.rs`: Add `TensorOp::embed_lookup` if going GPU route
-   `src/shaders/embed_lookup.wgsl`: New shader for GPU lookup

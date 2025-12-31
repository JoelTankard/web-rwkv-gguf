# State & Embedding Optimization

## Current State

| Backend           | Doc State | Field Analysis | Technique                               |
| ----------------- | --------- | -------------- | --------------------------------------- |
| WebGPU (web-rwkv) | 950.8ms   | 1.84s          | Full forward pass + state serialization |
| llama.cpp         | 296.1ms   | 563.3ms        | Native embedding API                    |

## Why llama.cpp is 3x Faster for State Operations

### llama.cpp Approach

llama.cpp provides a dedicated embedding extraction API:

```python
# llama-cpp-python
embeddings = model.embed(text)  # Returns float array directly
```

Under the hood:

1. Runs forward pass up to embedding layer only
2. Skips output projection and sampling
3. Returns raw hidden states efficiently
4. Optimized for batch embedding extraction

### WebGPU (web-rwkv) Approach

Current implementation requires:

```python
# web-rwkv-py current approach
tokens = model.encode(text)
model.run(tokens)           # Full forward pass
state = model.back_state()  # Extract state
embeddings = state.to_list()  # Serialize to Python list
```

Problems:

1. Runs full forward pass including output projection
2. State serialization crosses Rust/Python boundary
3. No batch optimization
4. State format not optimized for similarity calculations

## Proposed Solutions

### Solution 1: Add Dedicated Embedding API to web-rwkv

**Concept:** Add `embed()` method that stops at hidden state layer.

```rust
// In web-rwkv Rust code
impl Model {
    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        // Run only embedding + transformer layers
        // Skip output projection (lm_head)
        let hidden = self.forward_to_hidden(tokens);
        hidden.to_vec()
    }
}
```

```python
# Python wrapper
def embed(self, text: str) -> list[float]:
    tokens = self.encode(text)
    return self._model.embed(tokens)
```

**Benefits:**

-   Skips unnecessary computation (output projection)
-   Direct return without state serialization overhead
-   API compatible with llama.cpp

**Estimated Impact:** 2-3x speedup for embedding operations

### Solution 2: Batch Embedding Support

**Concept:** Process multiple texts in single GPU dispatch.

```rust
impl Model {
    pub fn embed_batch(&self, token_batches: &[Vec<u32>]) -> Vec<Vec<f32>> {
        // Pad sequences to same length
        // Run batched forward pass
        // Return embeddings for each input
    }
}
```

```python
# Python usage
texts = ["email 1 content", "email 2 content", "email 3 content"]
embeddings = model.embed_batch(texts)
```

**Benefits:**

-   Amortizes GPU dispatch overhead
-   Better GPU utilization
-   Significant speedup for field analysis (multiple embeddings)

**Estimated Impact:** 2-4x speedup for batch operations

### Solution 3: Optimized State Serialization

**Concept:** Use zero-copy transfer between Rust and Python.

Current approach (slow):

```rust
// Copies data to Python list
fn back_state(&self) -> PyResult<Vec<f32>> {
    Ok(self.state.clone())
}
```

Optimized approach:

```rust
// Return numpy array view (zero-copy)
fn back_state<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
    PyArray1::from_slice(py, &self.state)
}
```

**Benefits:**

-   Eliminates data copy between Rust and Python
-   Direct numpy array access
-   Compatible with existing similarity calculations

**Estimated Impact:** 1.5-2x speedup for state operations

### Solution 4: Cached Embeddings for Repeated Texts

**Concept:** Cache embeddings for frequently used prompts.

```python
class EmbeddingCache:
    def __init__(self, model, max_size: int = 1000):
        self._model = model
        self._cache = {}
        self._max_size = max_size

    def embed(self, text: str) -> list[float]:
        cache_key = hash(text)
        if cache_key not in self._cache:
            if len(self._cache) >= self._max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = self._model.embed(text)
        return self._cache[cache_key]
```

**Benefits:**

-   Instant retrieval for repeated prompts
-   Useful for field analysis with template prompts
-   Simple to implement

**Estimated Impact:** Near-instant for cached texts

## Implementation Priority

1. **Solution 3 (Zero-copy serialization)** - Quick win, minimal changes
2. **Solution 1 (Dedicated embed API)** - Medium effort, big impact
3. **Solution 4 (Caching)** - Easy to add on top
4. **Solution 2 (Batch support)** - Highest effort, best for scaling

## Files to Modify

-   `benchmark_tui/web-rwkv-py/src/lib.rs` - Add embed() method, zero-copy arrays
-   `benchmark_tui/model.py` - Add Python wrapper for embed()
-   `web-rwkv/crates/web-rwkv/src/model/run.rs` - Core embedding extraction

## RWKV-Specific Considerations

RWKV's state is different from transformer hidden states:

-   RWKV state is recurrent (carries information across tokens)
-   State size is fixed regardless of sequence length
-   State can be saved/restored for continuation

This is actually an advantage - we can:

1. Pre-compute state for common prefixes
2. Resume from saved states without re-processing
3. Use state directly as "embedding" for similarity

## References

-   [llama.cpp embedding implementation](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp)
-   [PyO3 numpy integration](https://docs.rs/numpy/latest/numpy/)
-   [RWKV state management](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/src/model.py)

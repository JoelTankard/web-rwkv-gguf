# Python Bindings Optimization Handoff

## UPDATE: Bindings Now in web-rwkv-gguf

The Python bindings have been moved into the `web-rwkv-gguf` repo at `crates/web-rwkv-py/`.

**To set up RWKV-email-agent to use the new bindings, see the prompt at the bottom of this file.**

---

## Context

The `web-rwkv-gguf` Rust library has been optimized with:

1. **Optimized `State::back()`** - Single GPU buffer + single GPU→CPU transfer (~6ms)
2. **`RnnOption::EmbedLast/EmbedFull`** - Skip head projection for embedding extraction

However, benchmarks show minimal improvement because the bottleneck is in the **Python bindings** (`web_rwkv_py`):

| Metric         | Current | Target | llama.cpp |
| -------------- | ------- | ------ | --------- |
| Doc State      | 938ms   | <300ms | 296ms     |
| Field Analysis | 1.83s   | <600ms | 563ms     |

## Root Cause

The current `back_state().to_list()` pattern is extremely slow:

```python
# Current (slow) - from benchmark_tui/dashboard.py lines 919, 937
final_doc_state = np.array(self._runner.back_state().to_list(), dtype=np.float32)
```

This converts the entire state tensor (~5.4M floats for the 2.9B model) to a Python list, then to numpy. The Python list intermediate is the killer.

## Required Changes

### 1. Add `to_numpy()` Method to State (Critical - 3x speedup)

**File:** `src/lib.rs` (or wherever `State` is exposed to Python)

Replace the `to_list()` approach with direct numpy array creation using PyO3's numpy support:

```rust
use numpy::{PyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pymethods]
impl State {
    /// Returns state data as a numpy array (zero-copy when possible)
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f32>> {
        // Get the raw f32 data from the TensorCpu
        let data: Vec<f32> = self.inner.clone().into();  // or however you access the data
        Ok(data.into_pyarray(py))
    }
}
```

**Cargo.toml addition:**

```toml
[dependencies]
numpy = "0.20"  # or latest compatible version
```

### 2. Expose `RnnOption::EmbedLast/EmbedFull` (1.5x speedup)

The Rust library now has these options that skip the expensive head projection:

```rust
pub enum RnnOption {
    Last,       // Output logits for last token
    Full,       // Output logits for all tokens
    EmbedLast,  // Output hidden state for last token (NEW - skips head projection)
    EmbedFull,  // Output hidden states for all tokens (NEW - skips head projection)
}
```

**Expose in Python bindings:**

```rust
#[pyclass]
#[derive(Clone)]
pub enum PyRnnOption {
    Last,
    Full,
    EmbedLast,
    EmbedFull,
}

impl From<PyRnnOption> for RnnOption {
    fn from(opt: PyRnnOption) -> Self {
        match opt {
            PyRnnOption::Last => RnnOption::Last,
            PyRnnOption::Full => RnnOption::Full,
            PyRnnOption::EmbedLast => RnnOption::EmbedLast,
            PyRnnOption::EmbedFull => RnnOption::EmbedFull,
        }
    }
}
```

**Update `run()` method signature:**

```rust
#[pymethods]
impl Model {
    #[pyo3(signature = (tokens, token_chunk_size=128, option=None))]
    fn run(
        &mut self,
        tokens: Vec<u32>,
        token_chunk_size: usize,
        option: Option<PyRnnOption>,
    ) -> PyResult<Vec<f32>> {
        let option = option.map(RnnOption::from).unwrap_or(RnnOption::Last);
        // ... use option in RnnInputBatch::new(tokens, option)
    }
}
```

### 3. Add `embed()` Convenience Method

```rust
#[pymethods]
impl Model {
    /// Extract embeddings without running head projection (faster than run + back_state)
    #[pyo3(signature = (tokens, token_chunk_size=128, last_only=true))]
    fn embed<'py>(
        &mut self,
        py: Python<'py>,
        tokens: Vec<u32>,
        token_chunk_size: usize,
        last_only: bool,
    ) -> PyResult<&'py PyArray1<f32>> {
        let option = if last_only { RnnOption::EmbedLast } else { RnnOption::EmbedFull };
        // Run inference with embed option
        // Return state as numpy array directly
    }
}
```

## Python Usage After Changes

```python
# Before (slow):
self._runner.run(doc_tokens, token_chunk_size=128)
final_doc_state = np.array(self._runner.back_state().to_list(), dtype=np.float32)

# After (fast):
self._runner.run(doc_tokens, token_chunk_size=128)
final_doc_state = self._runner.back_state().to_numpy()  # Direct numpy, no list conversion

# Or even faster with embed():
final_doc_state = self._runner.embed(doc_tokens)  # Skips head projection entirely
```

## Expected Impact

| Change       | Speedup | Doc State       | Field Analysis  |
| ------------ | ------- | --------------- | --------------- |
| `to_numpy()` | 3x      | 938ms → ~300ms  | 1.83s → ~600ms  |
| `EmbedLast`  | 1.5x    | ~300ms → ~200ms | ~600ms → ~400ms |
| Combined     | 4-5x    | **~200ms**      | **~400ms**      |

## Testing

After implementing, run the benchmark TUI:

```bash
cd /Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui
python dashboard.py
```

Load the GGUF model and run extraction. Check:

-   Doc State should be <300ms (target: ~200ms)
-   Field Analysis should be <700ms (target: ~400ms)

## Files to Modify

1. `Cargo.toml` - Add `numpy` dependency
2. `src/lib.rs` (or state module) - Add `to_numpy()` method
3. `src/lib.rs` (or model module) - Expose `RnnOption` variants, add `embed()` method

## Reference: Rust-side Changes Already Made

In `web-rwkv-gguf`:

-   `src/runtime/infer/rnn.rs` - Added `RnnOption::EmbedLast`, `RnnOption::EmbedFull`, `is_embed_only()`
-   `src/runtime/v7.rs` - Modified `dispatch_header()` to skip head projection when `embed_only=true`
-   `src/runtime/v6.rs` - Same changes for v6 models
-   `examples/test_embed_api.rs` - Test demonstrating the new functionality

The Rust library is ready. The Python bindings just need to expose these features and add zero-copy numpy support.

---

# RWKV-email-agent Setup Prompt

Copy this prompt to the RWKV-email-agent repo to set up the new optimized Python bindings:

---

## Task: Switch to Optimized Python Bindings from web-rwkv-gguf

The Python bindings have been moved into the `web-rwkv-gguf` repo with optimizations for state extraction and embedding. Update this project to use them.

### Step 1: Update Cargo.toml

Change the workspace to point to the new bindings location:

```toml
[workspace]
members = ["benchmark_tui/web-rwkv-py"]  # REMOVE THIS LINE
exclude = ["web-rwkv", "web-rwkv-gguf", "legacy_app"]
resolver = "2"
```

Replace with:

```toml
[workspace]
resolver = "2"
# Python bindings are now built from web-rwkv-gguf/crates/web-rwkv-py
```

### Step 2: Build the New Bindings

```bash
cd /Users/joel/Dev/Experimental/web-rwkv-gguf/crates/web-rwkv-py
pip install maturin
maturin develop --release
```

### Step 3: Update benchmark_tui/dashboard.py

Replace slow `to_list()` calls with optimized `to_numpy()` or `embed()`:

**Line ~919 (Doc State):**

```python
# Before:
final_doc_state = np.array(self._runner.back_state().to_list(), dtype=np.float32)

# After:
final_doc_state = self._runner.back_state_numpy()
```

**Line ~937 (Field Analysis loop):**

```python
# Before:
query_emb = np.array(self._runner.back_state().to_list(), dtype=np.float32)

# After:
query_emb = self._runner.back_state_numpy()
```

### Step 4: Update benchmark_tui/model.py

The `RWKVCompletionRunner` class wraps the bindings. Add convenience methods:

```python
def back_state_numpy(self):
    """Get state as numpy array directly (faster than back_state().to_list())"""
    return self._model.back_state_numpy()

def embed(self, tokens: list[int], token_chunk_size: int = 128) -> np.ndarray:
    """Get embeddings without running head projection (faster for similarity)"""
    tokens_u16 = [t if t < 65536 else 0 for t in tokens]
    return self._model.embed(tokens_u16, token_chunk_size=token_chunk_size)
```

### Step 5: (Optional) Use embed() for Field Analysis

For even faster field analysis, use `embed()` which skips the head projection entirely:

```python
# In _do_extraction, replace the field analysis loop:
for field_name, field_description in schema.items():
    field_query = f"{field_name}: {field_description}"
    field_query_tokens = self._runner.tokenize(field_query)

    self._runner.reset_state()
    # Use embed() instead of run() + back_state()
    query_emb = self._runner.embed(field_query_tokens)
    # ... rest of similarity calculation
```

### Expected Results

| Metric         | Before | After  | Improvement |
| -------------- | ------ | ------ | ----------- |
| Doc State      | 938ms  | ~200ms | 4.7x faster |
| Field Analysis | 1.83s  | ~400ms | 4.6x faster |

### New APIs Available

-   `model.back_state_numpy()` - Returns state as numpy array directly
-   `model.embed(tokens, last_only=True)` - Get embeddings, skips head projection
-   `state.to_numpy()` - Convert state to numpy (if you have a State object)
-   `PyRnnOption.EmbedLast` / `PyRnnOption.EmbedFull` - For explicit control

### Cleanup

You can remove the old `benchmark_tui/web-rwkv-py/` directory since the bindings are now in `web-rwkv-gguf/crates/web-rwkv-py/`.

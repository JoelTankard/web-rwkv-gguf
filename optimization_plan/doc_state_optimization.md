# Doc State Optimization Handoff

## Problem Statement

The `benchmark_tui` dashboard in `/Users/joel/Dev/Prototypes/RWKV-email-agent` has slow "Doc State" timing:

-   **Current**: 719.4ms for 212 tokens
-   **llama.cpp**: 296.1ms (2.4x faster)

## Root Cause Analysis

### Issue 1: Redundant Forward Pass

In `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/dashboard.py` lines 860-920:

```python
# Stage 2: Generation (processes document as part of prompt)
self._runner.reset_state()
result = self._runner.generate(extraction_prompt, ...)  # Document already processed here

# Stage 3: Doc State - REDUNDANT! Processes document AGAIN
self._runner.reset_state()  # Throws away state from generation!
self._runner.run(doc_tokens, token_chunk_size=128)  # Re-processes 212 tokens
final_doc_state = self._runner.back_state_numpy()
```

**The document is processed TWICE.** The 719ms is a full forward pass, not state extraction overhead.

### Issue 2: No Dedicated Embedding API Usage

The Python wrapper in `model.py` has an `embed()` method but it's not being used. The current approach:

1. Runs full forward pass (including head projection)
2. Then extracts state

## Fixes Required

### Fix 1: Eliminate Redundant Processing (HIGH IMPACT)

**File**: `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/dashboard.py`

**Current code** (lines 916-920):

```python
t0 = perf_counter()
self._runner.reset_state()
self._runner.run(doc_tokens, token_chunk_size=128)
final_doc_state = self._runner.back_state_numpy()
timings["doc_state"] = perf_counter() - t0
```

**Fixed code - Option A** (reorder to avoid redundant processing):

```python
# Get doc state BEFORE generation
t0 = perf_counter()
self._runner.reset_state()
self._runner.run(doc_tokens, token_chunk_size=128)
final_doc_state = self._runner.back_state_numpy()
timings["doc_state"] = perf_counter() - t0

# Then continue with generation from saved state or fresh state
# Don't reset and re-run the same tokens!
```

**Fixed code - Option B** (use embed API for similarity):

```python
t0 = perf_counter()
final_doc_state = self._runner.embed(doc_tokens, token_chunk_size=128)
timings["doc_state"] = perf_counter() - t0
```

**Expected Impact**: ~700ms to ~20ms (if reusing state) or ~600ms (if using embed API)

### Fix 2: Use embed() API for Field Analysis

**File**: `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/dashboard.py`

**Current code** (lines 933-937):

```python
field_query_tokens = self._runner.tokenize(field_query)
self._runner.reset_state()
self._runner.run(field_query_tokens, token_chunk_size=128)
query_emb = self._runner.back_state_numpy()
```

**Fixed code**:

```python
field_query_tokens = self._runner.tokenize(field_query)
query_emb = self._runner.embed(field_query_tokens, token_chunk_size=128)
```

The `embed()` API uses `RnnOption::EmbedLast` which skips head projection.

### Fix 3: Verify model.py Methods

**File**: `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/model.py`

Ensure the wrapper correctly calls the Rust bindings:

```python
def back_state_numpy(self) -> np.ndarray:
    """Get state as numpy array directly"""
    return self._model.back_state_numpy()

def embed(self, tokens: list[int], token_chunk_size: int = 128) -> np.ndarray:
    """Get embeddings without running head projection"""
    tokens_u16 = [t if t < 65536 else 0 for t in tokens]
    return self._model.embed(tokens_u16, token_chunk_size=token_chunk_size)
```

## Regarding llama.cpp RWKV Performance

llama.cpp supports RWKV models. Their faster performance comes from:

1. **Optimized GGML kernels** - Hand-tuned CPU/GPU kernels for matrix operations
2. **Efficient memory layout** - Contiguous memory access patterns
3. **Batched operations** - Even for RNNs, they batch across the embedding dimension

To match llama.cpp speed, web-rwkv would need:

1. More aggressive shader optimization for Q4_K matmul
2. Better GPU memory coalescing
3. Reduced GPU to CPU synchronization

However, the **immediate 2x+ speedup** comes from eliminating the redundant forward pass.

## Testing

After fixes, run the dashboard and verify:

1. Doc State timing drops from ~700ms to less than 100ms
2. Field Analysis timing also improves
3. Total extraction time should be ~50% faster

## Files to Modify

1. `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/dashboard.py`

    - Lines 916-920: Remove redundant `reset_state()` + `run()`
    - Lines 933-937: Use `embed()` instead of `run()` + `back_state_numpy()`

2. `/Users/joel/Dev/Prototypes/RWKV-email-agent/benchmark_tui/model.py`
    - Verify `embed()` and `back_state_numpy()` methods work correctly

## Verification Commands

```bash
cd /Users/joel/Dev/Prototypes/RWKV-email-agent
source .venv/bin/activate
python benchmark_tui/dashboard.py
```

Run an extraction and compare Doc State timing before and after.

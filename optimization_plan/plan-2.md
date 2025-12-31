# Optimization Plan Phase 2: Prefill Performance

## Context

Phase 1 optimizations (embed API, zero-copy transfers) are complete but prefill is still 2.3x slower than target.

## Benchmarking

**CRITICAL:** Run the benchmark tool BEFORE and AFTER each change to measure improvements.

```bash
# Before making changes
cargo run --release --example bench -- \
  --model /Users/joel/Dev/Experimental/web-rwkv-gguf/assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf \
  --title "Feature Name" \
  --change "Baseline before changes"

# After making changes (use SAME title to group results)
cargo run --release --example bench -- \
  --model /Users/joel/Dev/Experimental/web-rwkv-gguf/assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf \
  --title "Feature Name" \
  --change "After implementing X"
```

**Important:** Use the same `--title` for before/after comparisons - results are grouped by title in `benchmarks.md`.

**Metrics tracked:**

-   **Load Time (ms)** - Model loading speed
-   **Prefill (tok/s)** - Prompt processing speed (this is what we're optimizing)
-   **Generation (tok/s)** - Token generation speed (must maintain >30 tok/s)
-   **Quality Hash** - Deterministic output hash to verify correctness

Record results in `optimization_plan/benchmarks.md` for each change.

### Current Performance Comparison

| Metric              | web-rwkv (WebGPU) | llama.cpp  | Winner                   |
| ------------------- | ----------------- | ---------- | ------------------------ |
| Generation          | 31.5 tok/s        | 11.4 tok/s | **web-rwkv 3x faster**   |
| Doc State (prefill) | 682.5ms           | 296.1ms    | llama.cpp 2.3x faster    |
| Field Analysis      | 327.1ms           | 563.3ms    | **web-rwkv 1.7x faster** |

### Key Insight

WebGPU generation is **3x faster** than llama.cpp. The bottleneck is specifically **prefill** (processing multiple tokens at once), not single-token generation.

### Constraints

-   **Must use WebGPU** - No hybrid approach (double model loading unacceptable)
-   **F16 compute** - Native quant compute (Q4_K shaders) tested slower than F16 dequant-then-compute
-   **Maintain generation speed** - Don't regress the 31.5 tok/s

## Root Cause Analysis

### Why Prefill is Slow

Prefill processes 212 tokens in ~680ms = **3.2ms per token**.
Generation processes 1 token in ~32ms = **32ms per token**.

Wait - that means **prefill is actually 10x faster per token than generation**. The issue isn't prefill efficiency, it's that we're still doing a full forward pass.

Let me recalculate:

-   212 tokens in 682ms = 310 tok/s prefill
-   1 token in 32ms = 31.5 tok/s generation

**Prefill IS faster** (310 tok/s vs 31.5 tok/s). The question is: can we make it even faster?

llama.cpp achieves: 212 tokens in 296ms = **715 tok/s prefill**

So the gap is: **310 tok/s (web-rwkv) vs 715 tok/s (llama.cpp)** = 2.3x difference in prefill throughput.

## Optimization Targets

| Metric                 | Current   | Target    | Improvement Needed |
| ---------------------- | --------- | --------- | ------------------ |
| Prefill throughput     | 310 tok/s | 700 tok/s | 2.3x               |
| Doc State (212 tokens) | 682ms     | 300ms     | 2.3x               |
| Field Analysis         | 327ms     | 150ms     | 2.2x               |

## Action Plan

### Phase 2.1: Profile the Forward Pass (1-2 days)

**Goal:** Identify which operations consume the most time during prefill.

#### Task 2.1.1: Add GPU Timing

```rust
// In src/runtime/v7.rs or context.rs
// Use wgpu timestamp queries to measure each operation
let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
    ty: wgpu::QueryType::Timestamp,
    count: 64,
});
```

**Metrics to capture:**

-   Embedding lookup time
-   Per-layer time breakdown:
    -   Layer norm
    -   Token shift
    -   Attention (time_mix)
    -   FFN (channel_mix)
    -   Matrix multiplications
-   Head projection time (should be skipped for embed)

#### Task 2.1.2: Identify Bottlenecks

Expected bottlenecks based on RWKV v7 architecture:

1. **Matrix multiplications** - Most compute-heavy
2. **Token shift operations** - Sequential dependency
3. **Time mixing** - Complex attention mechanism
4. **GPU dispatch overhead** - Many small dispatches

### Phase 2.2: Optimize Matrix Multiplications (2-3 days)

**Goal:** Speed up matmul operations which dominate compute time.

#### Task 2.2.1: Optimize Chunk Size

Current: `token_chunk_size = 128`

Test different chunk sizes:

-   64, 128, 256, 512
-   Find optimal balance between parallelism and memory

#### Task 2.2.2: Shader Workgroup Optimization

Review matmul shaders for:

-   Workgroup size tuning (currently likely 8x8 or 16x16)
-   Shared memory usage
-   Memory coalescing patterns

**Files to review:**

-   `src/shaders/matmul_vec_fp16.wgsl`
-   `src/shaders/matmul_mat_fp16.wgsl`

#### Task 2.2.3: Reduce Dispatch Overhead

Current architecture dispatches many small operations. Consider:

-   Fusing layer norm + first matmul
-   Fusing activation + next operation
-   Batching multiple layers into single dispatch where possible

### Phase 2.3: Optimize Token Processing (2-3 days)

**Goal:** Reduce overhead in token-level operations.

#### Task 2.3.1: Token Shift Optimization

The `token_shift` operation has sequential dependency but can be optimized:

-   Pre-compute shift coefficients
-   Use parallel prefix scan for multi-token shifts
-   Reduce memory bandwidth

**File:** `src/shaders/token_shift.wgsl`

#### Task 2.3.2: Time Mix Optimization

RWKV v7 time mixing is the core attention mechanism:

-   Review `time_mix.wgsl` for optimization opportunities
-   Consider fusing with adjacent operations
-   Optimize memory access patterns

### Phase 2.4: Reduce GPU-CPU Synchronization (1-2 days)

**Goal:** Minimize round-trips between GPU and CPU.

#### Task 2.4.1: Pipeline Command Buffers

Current: Build and submit command buffer per chunk
Optimized: Pre-build command buffers, submit in pipeline

#### Task 2.4.2: Async State Readback

For embed operations, state readback can be async:

-   Don't block on GPU completion
-   Use mapped buffer callbacks
-   Pipeline next operation while waiting

### Phase 2.5: Batch Embedding Operations (2-3 days)

**Goal:** Process multiple embedding requests in single GPU pass.

#### Task 2.5.1: Batch State Computation

For field analysis (6 fields), currently runs 6 separate forward passes.
Optimize by batching:

```rust
// Instead of 6 separate calls:
for field in fields {
    embed(field_tokens)  // 6 GPU dispatches
}

// Batch into single call:
embed_batch([field1_tokens, field2_tokens, ...])  // 1 GPU dispatch
```

**Impact:** Field Analysis 327ms → ~100ms (3x improvement)

#### Task 2.5.2: Parallel State Extraction

When extracting multiple embeddings, parallelize the GPU work:

-   Pad sequences to same length
-   Run batched forward pass
-   Extract all embeddings at once

## Implementation Priority

1. **Phase 2.1 (Profiling)** - Must do first to identify actual bottlenecks
2. **Phase 2.5 (Batch Embedding)** - Quick win for Field Analysis
3. **Phase 2.2 (Matmul Optimization)** - Highest potential impact
4. **Phase 2.3 (Token Processing)** - Medium impact
5. **Phase 2.4 (Sync Reduction)** - Polish

## Timeline

```
Week 1:
├── Day 1: Task 2.1.1-2.1.2 (Profiling)
├── Day 2-3: Task 2.5.1-2.5.2 (Batch Embedding)
└── Day 4-5: Task 2.2.1-2.2.2 (Matmul Optimization)

Week 2:
├── Day 1-2: Task 2.2.3 (Dispatch Fusion)
├── Day 3-4: Task 2.3.1-2.3.2 (Token Processing)
└── Day 5: Task 2.4.1-2.4.2 (Sync Reduction)
```

## Success Metrics

| Metric             | Current    | Target    | Status      |
| ------------------ | ---------- | --------- | ----------- |
| Prefill throughput | 310 tok/s  | 700 tok/s | Pending     |
| Doc State          | 682ms      | 300ms     | Pending     |
| Field Analysis     | 327ms      | 100ms     | Pending     |
| Generation         | 31.5 tok/s | >30 tok/s | ✅ Maintain |

## Files to Modify

### Profiling

-   `src/context.rs` - Add timestamp queries
-   `src/runtime/v7.rs` - Add timing instrumentation

### Shader Optimization

-   `src/shaders/matmul_vec_fp16.wgsl`
-   `src/shaders/matmul_mat_fp16.wgsl`
-   `src/shaders/token_shift.wgsl`
-   `src/shaders/time_mix.wgsl`

### Batching

-   `src/runtime/infer/rnn.rs` - Batch input handling
-   `crates/web-rwkv-py/src/lib.rs` - Python batch API

## Notes

-   F16 compute outperforms native Q4_K shaders by 3x (tested)
-   WebGPU generation is 3x faster than llama.cpp - preserve this
-   Focus on prefill-specific optimizations
-   Profile before optimizing - don't guess at bottlenecks

## References

-   [WebGPU Best Practices](https://toji.dev/webgpu-best-practices/)
-   [wgpu Profiling](https://docs.rs/wgpu/latest/wgpu/struct.Device.html#method.create_query_set)
-   [RWKV v7 Architecture](https://github.com/BlinkDL/RWKV-LM)

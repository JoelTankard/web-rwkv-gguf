# Optimization Plan: Enhanced web-rwkv

## Executive Summary

Enhance the web-rwkv inference backend to achieve:

-   **<500ms model load** (incremental GPU buffer population + mmap)
-   **30+ tok/s generation** (keep existing WebGPU WKV shaders)
-   **<400ms state/embedding ops** (dedicated embedding API + zero-copy)
-   **<500µs tokenization** (keep existing Rust tokenizer)

**Constraints:**

-   Single model in memory (~2GB for 2.9B model)
-   No external dependencies (no llama.cpp runtime)
-   Contribute improvements upstream to web-rwkv-gguf

## Related Documents

| Document                                             | Description                                  |
| ---------------------------------------------------- | -------------------------------------------- |
| [performance_analysis.md](./performance_analysis.md) | Benchmark comparison and root cause analysis |
| [model_loading.md](./model_loading.md)               | Fast model loading solutions                 |
| [state_embeddings.md](./state_embeddings.md)         | State and embedding optimization             |
| [hybrid_architecture.md](./hybrid_architecture.md)   | Enhanced web-rwkv architecture               |

## Repository Setup

```bash
git clone https://github.com/JoelTankard/web-rwkv-gguf
cd web-rwkv-gguf
git checkout -b feature/batched-gpu-uploads
```

## Benchmarking

**IMPORTANT:** Run the benchmark tool BEFORE and AFTER making changes to track performance impact.

```bash
# Before making changes
cargo run --release --example benchmark -- \
  --model assets/models/rwkv7-g1a-0.1b-20250728-ctx4096.Q4_K_M.gguf \
  --title "<Feature Name>" \
  --change "Baseline before <description>"

# After making changes (use SAME title to group results)
cargo run --release --example benchmark -- \
  --model assets/models/rwkv7-g1a-0.1b-20250728-ctx4096.Q4_K_M.gguf \
  --title "<Feature Name>" \
  --change "After <description of changes>"
```

Results are written to `optimization_plan/benchmarks.md` and grouped by title for easy comparison.

**Metrics tracked:**

-   **Load Time (ms)** - Model loading speed
-   **Prefill (tok/s)** - Prompt processing speed
-   **Generation (tok/s)** - Token generation speed
-   **Quality Hash** - Deterministic output hash to verify correctness

## Action Plan

### Phase 1: Model Loading Optimization (3-5 days)

**Goal:** Reduce model load time from 10s to <500ms.

#### Task 1.1: Incremental GPU Buffer Population (Primary)

-   **File:** `src/model/loader.rs` (or equivalent in web-rwkv-gguf)
-   **Change:** Lazy GPU buffer allocation, populate on first access
-   **Impact:** ~500ms load time (combined with mmap)
-   **Effort:** High

```rust
struct LazyBuffer {
    cpu_data: Option<Arc<[f32]>>,
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

#### Task 1.2: Memory-Mapped Model Files (Backup)

-   **File:** `src/model/loader.rs`
-   **Change:** Use mmap for model file access
-   **Impact:** Foundation for lazy loading
-   **Effort:** Medium

### Phase 2: State & Embedding Optimization (3-5 days)

**Goal:** Reduce state/embedding ops from 950ms to <400ms.

#### Task 2.1: Dedicated Embedding API

-   **File:** `src/model/run.rs` (or Python bindings)
-   **Change:** Add `embed()` method that skips output projection
-   **Impact:** 2-3x faster embedding extraction
-   **Effort:** Medium

#### Task 2.2: Zero-Copy State Transfer

-   **File:** Python bindings (`src/lib.rs` in PyO3)
-   **Change:** Return numpy arrays directly from Rust
-   **Impact:** 1.5-2x faster state operations
-   **Effort:** Medium

#### Task 2.3: Embedding Cache

-   **File:** Python wrapper layer
-   **Change:** LRU cache for repeated embeddings
-   **Impact:** Instant retrieval for cached prompts
-   **Effort:** Low

### Phase 3: Validation & Polish (2-3 days)

**Goal:** Verify targets met, clean up for upstream contribution.

#### Task 3.1: Benchmark Suite

-   **Change:** Create comprehensive benchmarks
-   **Impact:** Verify optimization success
-   **Effort:** Low

#### Task 3.2: Documentation

-   **Change:** Document new APIs and optimizations
-   **Impact:** Enable upstream contribution
-   **Effort:** Low

## Timeline

```text
Week 1:
├── Day 1-2: Task 1.1 (Incremental GPU buffers)
├── Day 3-4: Task 1.2 (mmap if needed)
└── Day 5: Integration testing

Week 2:
├── Day 1-2: Task 2.1 (Embedding API)
├── Day 3: Task 2.2 (Zero-copy transfer)
├── Day 4: Task 2.3 (Embedding cache)
└── Day 5: Phase 3 (Validation & docs)
```

## Success Metrics

| Metric         | Current (WebGPU) | Target    | Improvement |
| -------------- | ---------------- | --------- | ----------- |
| Model Load     | 10s              | <500ms    | 20x         |
| Tokenize       | 307µs            | <500µs    | maintain    |
| Generation     | 30.6 tok/s       | >30 tok/s | maintain    |
| Doc State      | 950ms            | <400ms    | 2.4x        |
| Field Analysis | 1.84s            | <700ms    | 2.6x        |

## Risk Assessment

| Risk                                    | Likelihood | Impact | Mitigation                                 |
| --------------------------------------- | ---------- | ------ | ------------------------------------------ |
| Incremental loading breaks inference    | Low        | High   | Extensive testing, fallback to full load   |
| mmap incompatible with GPU buffers      | Low        | Medium | Use standard file I/O with lazy allocation |
| First inference slower due to lazy load | Low        | Low    | Acceptable tradeoff for fast startup       |

## Getting Started

1. **Clone** the web-rwkv-gguf repository
2. **Create** feature branch `feature/optimized-loading`
3. **Read** [performance_analysis.md](./performance_analysis.md) for context
4. **Start** with Task 1.1 (Incremental GPU Buffer Population)
5. **Benchmark** after each change to verify improvement

## Notes

-   All changes should be designed for upstream contribution
-   Keep existing API compatibility where possible
-   Add benchmarks for each optimization
-   Document any breaking changes

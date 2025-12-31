# Handoff Prompt: Enhanced web-rwkv Optimization

Use this prompt to start a new chat session for implementing the optimizations.

---

## Prompt

I need help optimizing the web-rwkv inference engine for RWKV models. The goal is to achieve fast, accurate, low-memory inference by porting techniques from llama.cpp into the existing WebGPU-based web-rwkv implementation.

### Background

I've benchmarked two inference backends for RWKV7 models:

| Metric     | WebGPU (web-rwkv) | llama.cpp  | Target    |
| ---------- | ----------------- | ---------- | --------- |
| Model Load | 10s               | 250ms      | <500ms    |
| Tokenize   | 307µs             | 936µs      | <500µs    |
| Generation | 30.6 tok/s        | 11.4 tok/s | >30 tok/s |

### Constraints

-   **Single model in memory** (~2GB for 2.9B model) - no dual-backend approach
-   **No external dependencies** - no llama.cpp runtime required
-   **Contribute upstream** - changes should benefit the web-rwkv-gguf community

### Planning Documents

Read these documents in `benchmark_tui/optimization_plan/`:

1. **plan.md** - Main action plan with phased tasks
2. **performance_analysis.md** - Benchmark comparison and root cause analysis
3. **model_loading.md** - Fast model loading (Primary: Incremental GPU Buffer, Backup: mmap)
4. **state_embeddings.md** - State and embedding optimization
5. **hybrid_architecture.md** - Enhanced web-rwkv architecture

### Phase 1: Model Loading Optimization

**Primary approach:** Incremental GPU Buffer Population

-   Lazy GPU buffer allocation
-   Populate buffers on first access during inference
-   Combined with mmap for ~500ms total load time

**Backup approach:** Memory-Mapped Model Files

-   Use mmap for model file access if incremental loading alone is insufficient

### Phase 2: State & Embedding Optimization

1. **Dedicated Embedding API** - Add `embed()` method that skips output projection
2. **Zero-Copy State Transfer** - Return numpy arrays directly from Rust
3. **Embedding Cache** - LRU cache for repeated embeddings

### What I Need

1. Start by reading the planning documents
2. Clone `https://github.com/JoelTankard/web-rwkv-gguf` and create branch `feature/optimized-loading`
3. Begin with Task 1.1: Incremental GPU Buffer Population
4. Benchmark after each change to verify improvement

---

## Files to Include

When starting the new chat, reference these files:

```
@benchmark_tui/optimization_plan/plan.md
@benchmark_tui/optimization_plan/performance_analysis.md
@benchmark_tui/optimization_plan/model_loading.md
@benchmark_tui/optimization_plan/state_embeddings.md
@benchmark_tui/optimization_plan/hybrid_architecture.md
```

Or if working in web-rwkv-gguf repo, copy the `optimization_plan/` folder there first.

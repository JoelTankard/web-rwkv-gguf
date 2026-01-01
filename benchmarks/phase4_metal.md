# Benchmark Results

This file tracks performance benchmarks across changes to the codebase.

**How to use:**
1. Run benchmark BEFORE making changes with a descriptive title
2. Make your changes
3. Run benchmark AFTER with the SAME title to group results

```bash
cargo run --release --example benchmark -- --model <path> --title "<title>" --change "<description>"
```

---

## Phase 4 Baseline

### 2026-01-01 05:40:53 - Before Metal backend implementation

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4367.84 ms** |
| **Prefill** | **151.93 tok/s** |
| **Generation** | **44.20 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


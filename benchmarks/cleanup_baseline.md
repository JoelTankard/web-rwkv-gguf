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

## Pre-cleanup Q8_0

### 2026-01-03 22:30:47 - Baseline before cleanup

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **8898.28 ms** |
| **Prefill** | **207.32 tok/s** |
| **Generation** | **52.82 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Pre-cleanup F16

### 2026-01-03 22:31:57 - Baseline before cleanup

| Metric | Value |
|--------|-------|
| Model | 2.9b-f16.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **17418.61 ms** |
| **Prefill** | **218.48 tok/s** |
| **Generation** | **44.37 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Post-cleanup Q8_0

### 2026-01-03 22:40:48 - After cleanup

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **8513.76 ms** |
| **Prefill** | **206.64 tok/s** |
| **Generation** | **53.26 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Post-cleanup F16

### 2026-01-03 22:42:00 - After cleanup

| Metric | Value |
|--------|-------|
| Model | 2.9b-f16.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **16862.48 ms** |
| **Prefill** | **219.56 tok/s** |
| **Generation** | **44.71 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>


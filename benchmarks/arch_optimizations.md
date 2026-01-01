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

## Baseline

### 2026-01-01 05:06:38 - Before RWKV architecture optimizations

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4460.77 ms** |
| **Prefill** | **152.25 tok/s** |
| **Generation** | **44.53 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Shader Opt 1

### 2026-01-01 05:08:27 - Optimized time_mix_v7 state access pattern

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4455.62 ms** |
| **Prefill** | **151.80 tok/s** |
| **Generation** | **44.10 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Workgroup Size 64

### 2026-01-01 05:10:09 - Increased time_mix_v7 BLOCK_SIZE from 32 to 64

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4515.94 ms** |
| **Prefill** | **150.93 tok/s** |
| **Generation** | **43.58 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Fused State Loop

### 2026-01-01 05:14:11 - Fused two loops in time_mix_v7 to read state once per j iteration

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4468.99 ms** |
| **Prefill** | **154.70 tok/s** |
| **Generation** | **44.36 tok/s** |
| Quality Hash | `bcbb65bac1fd4a96` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 3319, 153, 129, 3319, 153, 129, 3319, 153, 129, 3319, 153, 129]
```

</details>

## Reverted

### 2026-01-01 05:15:21 - Reverted to original time_mix_v7 shader

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4354.14 ms** |
| **Prefill** | **152.02 tok/s** |
| **Generation** | **44.27 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


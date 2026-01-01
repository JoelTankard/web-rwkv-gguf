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

## Fused WKV Kernel

### 2026-01-01 00:01:16 - Baseline before fusion

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **5242.61 ms** |
| **Prefill** | **147.82 tok/s** |
| **Generation** | **42.82 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-01 00:09:40 - pack_kvakk: fuse 4 blit ops into single kernel

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4284.98 ms** |
| **Prefill** | **152.37 tok/s** |
| **Generation** | **44.08 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## baseline

### 2026-01-01 05:59:51 - none

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4641.46 ms** |
| **Prefill** | **152.34 tok/s** |
| **Generation** | **44.17 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Metal Integration

### 2026-01-01 06:24:16 - Phase 1 Metal buffer creation during Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3483.24 ms** |
| **Prefill** | **121.39 tok/s** |
| **Generation** | **15.12 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Metal Dispatch

### 2026-01-01 06:37:11 - Q4K matmul emits MetalMatmulQ4K ops

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3631.35 ms** |
| **Prefill** | **132.95 tok/s** |
| **Generation** | **96.96 tok/s** |
| Quality Hash | `f8abdfcaacccd600` |

<details><summary>Quality tokens (first 16)</summary>

```
[267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267]
```

</details>

## WebGPU Baseline

### 2026-01-01 06:38:35 - Native Q4K loading with WebGPU shader

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3560.27 ms** |
| **Prefill** | **137.58 tok/s** |
| **Generation** | **15.09 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-01 06:50:19 - Metal disabled, WebGPU Q4K shader

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3556.45 ms** |
| **Prefill** | **137.64 tok/s** |
| **Generation** | **15.10 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


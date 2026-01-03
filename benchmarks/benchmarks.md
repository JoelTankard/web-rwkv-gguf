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

## Metal Test

### 2026-01-02 14:20:16 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4673.85 ms** |
| **Prefill** | **219.30 tok/s** |
| **Generation** | **44.60 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:21:19 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4351.37 ms** |
| **Prefill** | **218.31 tok/s** |
| **Generation** | **44.50 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:21:52 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4337.05 ms** |
| **Prefill** | **219.55 tok/s** |
| **Generation** | **44.55 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:22:53 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4461.05 ms** |
| **Prefill** | **219.17 tok/s** |
| **Generation** | **44.31 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:23:48 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4329.54 ms** |
| **Prefill** | **219.89 tok/s** |
| **Generation** | **44.49 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:25:56 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4453.29 ms** |
| **Prefill** | **219.12 tok/s** |
| **Generation** | **44.26 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:26:28 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4367.73 ms** |
| **Prefill** | **219.58 tok/s** |
| **Generation** | **44.36 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:27:49 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4351.81 ms** |
| **Prefill** | **219.51 tok/s** |
| **Generation** | **44.33 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:28:46 - Testing Metal V7

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4527.02 ms** |
| **Prefill** | **219.02 tok/s** |
| **Generation** | **44.65 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Metal V7 Test

### 2026-01-02 14:31:44 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3262.12 ms** |
| **Prefill** | **95.08 tok/s** |
| **Generation** | **15.25 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:32:51 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3252.48 ms** |
| **Prefill** | **94.45 tok/s** |
| **Generation** | **15.26 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:33:57 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3302.40 ms** |
| **Prefill** | **95.21 tok/s** |
| **Generation** | **15.22 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:34:57 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3407.94 ms** |
| **Prefill** | **94.70 tok/s** |
| **Generation** | **15.21 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:35:36 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3255.59 ms** |
| **Prefill** | **94.90 tok/s** |
| **Generation** | **15.17 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:36:16 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3313.96 ms** |
| **Prefill** | **94.95 tok/s** |
| **Generation** | **15.22 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2026-01-02 14:37:30 - Enabled Q4K loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3320.59 ms** |
| **Prefill** | **94.78 tok/s** |
| **Generation** | **15.13 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Baseline

### 2026-01-02 14:41:58 - Restored F16 loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4807.82 ms** |
| **Prefill** | **216.50 tok/s** |
| **Generation** | **43.80 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Q4K Metal Test

### 2026-01-02 14:47:27 - Testing with Q4K loading enabled

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **3896.94 ms** |
| **Prefill** | **95.04 tok/s** |
| **Generation** | **15.25 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Q8_0 Metal Test

### 2026-01-02 14:54:55 - Int8 Metal support

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6887.40 ms** |
| **Prefill** | **207.45 tok/s** |
| **Generation** | **53.02 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>


### 2026-01-02 14:57:13 - Int8+Fp16 Metal support

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6927.24 ms** |
| **Prefill** | **207.10 tok/s** |
| **Generation** | **53.15 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Q8_0 Metal V7

### 2026-01-02 14:58:06 - Final test

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6850.77 ms** |
| **Prefill** | **207.46 tok/s** |
| **Generation** | **53.43 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## test

### 2026-01-03 10:40:04 - test

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7119.54 ms** |
| **Prefill** | **208.53 tok/s** |
| **Generation** | **54.03 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Q8_0 test

### 2026-01-03 11:02:45 - baseline

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6814.97 ms** |
| **Prefill** | **205.74 tok/s** |
| **Generation** | **52.42 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Q4K F16 fallback

### 2026-01-03 11:59:12 - disabled native Q4K

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **5023.83 ms** |
| **Prefill** | **6.15 tok/s** |
| **Generation** | **1.37 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


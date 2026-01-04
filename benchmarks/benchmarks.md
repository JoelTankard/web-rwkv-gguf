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

### 2026-01-04 08:38:14 - Q8_0 Metal kernel

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **9428.01 ms** |
| **Prefill** | **596.39 tok/s** |
| **Generation** | **83.54 tok/s** |
| Quality Hash | `438287259638b200` |

<details><summary>Quality tokens (first 16)</summary>

```
[2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361]
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

### 2026-01-04 05:37:23 - No fusion - separate layer norm and token shifts

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6810.33 ms** |
| **Prefill** | **207.21 tok/s** |
| **Generation** | **53.55 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>


### 2026-01-04 05:44:51 - No fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6905.47 ms** |
| **Prefill** | **206.88 tok/s** |
| **Generation** | **52.65 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>


### 2026-01-04 05:59:14 - No fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7155.32 ms** |
| **Prefill** | **206.07 tok/s** |
| **Generation** | **52.97 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
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

### 2026-01-03 22:45:24 - test

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **8482.25 ms** |
| **Prefill** | **209.16 tok/s** |
| **Generation** | **53.79 tok/s** |
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

## Fused Test

### 2026-01-04 05:36:40 - Testing fused token shift + layer norm with increased buffer limit

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6985.49 ms** |
| **Prefill** | **207.20 tok/s** |
| **Generation** | **55.60 tok/s** |
| Quality Hash | `9885271c252a7220` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 227, 33, 3319, 33, 3319, 33, 3319, 33, 3319, 33, 3319, 33, 3319, 33]
```

</details>

## Fused Fixed

### 2026-01-04 05:44:06 - Fixed in-place normalization

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7072.35 ms** |
| **Prefill** | **206.30 tok/s** |
| **Generation** | **53.95 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Fused ATT+FFN

### 2026-01-04 05:58:31 - Fused ATT LN+6TS and FFN LN+TS

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **9135.10 ms** |
| **Prefill** | **200.30 tok/s** |
| **Generation** | **51.33 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Fused ATT Only

### 2026-01-04 06:00:23 - Only ATT LN+6TS fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7328.72 ms** |
| **Prefill** | **195.95 tok/s** |
| **Generation** | **50.69 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Baseline 1

### 2026-01-04 06:01:05 - No fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6982.19 ms** |
| **Prefill** | **205.04 tok/s** |
| **Generation** | **49.22 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Baseline 2

### 2026-01-04 06:01:40 - No fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6945.50 ms** |
| **Prefill** | **208.48 tok/s** |
| **Generation** | **54.20 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Baseline 3

### 2026-01-04 06:02:15 - No fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7021.89 ms** |
| **Prefill** | **209.30 tok/s** |
| **Generation** | **53.99 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Fused 1

### 2026-01-04 06:02:59 - ATT fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7214.15 ms** |
| **Prefill** | **201.95 tok/s** |
| **Generation** | **54.72 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Fused 2

### 2026-01-04 06:03:34 - ATT fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6852.12 ms** |
| **Prefill** | **209.59 tok/s** |
| **Generation** | **54.95 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Fused 3

### 2026-01-04 06:04:08 - ATT fusion

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6951.45 ms** |
| **Prefill** | **209.67 tok/s** |
| **Generation** | **55.20 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## Async GPU-CPU Overlap

### 2026-01-04 06:42:40 - Baseline before optimization

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **9168.66 ms** |
| **Prefill** | **208.60 tok/s** |
| **Generation** | **54.11 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## GPU Compute Opt

### 2026-01-04 07:40:33 - Testing fused vs non-fused

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7458.22 ms** |
| **Prefill** | **203.49 tok/s** |
| **Generation** | **53.19 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>


### 2026-01-04 07:41:17 - Testing non-fused baseline

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6885.09 ms** |
| **Prefill** | **205.44 tok/s** |
| **Generation** | **53.08 tok/s** |
| Quality Hash | `c55251c506e49c98` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157, 33, 236, 149, 157]
```

</details>

## TensorOp Caching

### 2026-01-04 08:13:31 - Added TensorOp tree caching

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7187.30 ms** |
| **Prefill** | **207.62 tok/s** |
| **Generation** | **54.31 tok/s** |
| Quality Hash | `552b965f4d00419e` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 65535, 64865, 64865, 65535, 64865, 64865, 65535, 64865, 64865, 65535, 64865, 64865, 65535, 64865, 64865]
```

</details>

## TensorOp Caching Verify

### 2026-01-04 08:14:15 - Verify correctness

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7148.29 ms** |
| **Prefill** | **207.11 tok/s** |
| **Generation** | **53.92 tok/s** |
| Quality Hash | `4dbb661be0c4c8f4` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 21818, 65535, 21818, 65535, 21818, 21818, 65535, 21818, 21818, 65535, 21818, 21818, 65535, 21818, 21818]
```

</details>

## WebGPU Baseline

### 2026-01-04 08:39:08 - No Metal

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7517.54 ms** |
| **Prefill** | **205.30 tok/s** |
| **Generation** | **53.76 tok/s** |
| Quality Hash | `ce40c707fa5590eb` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 65535, 116, 65535, 116, 21265, 65535, 116, 21265, 22590, 21265, 21265, 22590, 21265, 21265]
```

</details>

## Metal Q8_0

### 2026-01-04 09:00:15 - Native Q8_0 loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4646.32 ms** |
| **Prefill** | **328.02 tok/s** |
| **Generation** | **87.75 tok/s** |
| Quality Hash | `f8abdfcaacccd600` |

<details><summary>Quality tokens (first 16)</summary>

```
[267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267, 267]
```

</details>


### 2026-01-04 09:01:44 - Fixed dispatch

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4547.13 ms** |
| **Prefill** | **332.12 tok/s** |
| **Generation** | **99.92 tok/s** |
| Quality Hash | `fc01fbe3ced0ed58` |

<details><summary>Quality tokens (first 16)</summary>

```
[267, 267, 65535, 267, 65535, 267, 2779, 65535, 267, 51284, 65535, 267, 45, 65535, 267, 51284]
```

</details>


### 2026-01-04 09:14:45 - Fixed block layout

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4919.69 ms** |
| **Prefill** | **88.80 tok/s** |
| **Generation** | **39.26 tok/s** |
| Quality Hash | `16a7da583c362c1e` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 332, 46, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332]
```

</details>


### 2026-01-04 09:17:10 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4654.32 ms** |
| **Prefill** | **88.55 tok/s** |
| **Generation** | **39.69 tok/s** |
| Quality Hash | `ecc6ad6c8a4205cd` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 332, 65535, 332, 65535, 332, 332, 65535, 332, 332, 65535, 332, 332, 65535, 332, 332]
```

</details>


### 2026-01-04 09:17:56 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4524.61 ms** |
| **Prefill** | **87.61 tok/s** |
| **Generation** | **39.00 tok/s** |
| Quality Hash | `f091452164ed0fc` |

<details><summary>Quality tokens (first 16)</summary>

```
[46, 332, 46, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332]
```

</details>


### 2026-01-04 09:19:43 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4618.54 ms** |
| **Prefill** | **87.50 tok/s** |
| **Generation** | **38.35 tok/s** |
| Quality Hash | `bc667ec83d759800` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]
```

</details>


### 2026-01-04 09:20:21 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4536.80 ms** |
| **Prefill** | **88.06 tok/s** |
| **Generation** | **38.88 tok/s** |
| Quality Hash | `16a7da583c362c1e` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 332, 46, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332]
```

</details>


### 2026-01-04 09:21:35 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4530.02 ms** |
| **Prefill** | **86.56 tok/s** |
| **Generation** | **38.57 tok/s** |
| Quality Hash | `f091452164ed0fc` |

<details><summary>Quality tokens (first 16)</summary>

```
[46, 332, 46, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332, 46, 332, 332]
```

</details>


### 2026-01-04 09:22:39 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4619.32 ms** |
| **Prefill** | **86.69 tok/s** |
| **Generation** | **38.73 tok/s** |
| Quality Hash | `b4c7b8c2178e3cde` |

<details><summary>Quality tokens (first 16)</summary>

```
[46, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]
```

</details>


### 2026-01-04 09:23:37 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4515.55 ms** |
| **Prefill** | **86.68 tok/s** |
| **Generation** | **38.76 tok/s** |
| Quality Hash | `b4c7b8c2178e3cde` |

<details><summary>Quality tokens (first 16)</summary>

```
[46, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]
```

</details>


### 2026-01-04 09:24:18 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4514.77 ms** |
| **Prefill** | **86.43 tok/s** |
| **Generation** | **38.93 tok/s** |
| Quality Hash | `b4c7b8c2178e3cde` |

<details><summary>Quality tokens (first 16)</summary>

```
[46, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]
```

</details>


### 2026-01-04 09:25:57 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4427.64 ms** |
| **Prefill** | **88.73 tok/s** |
| **Generation** | **39.63 tok/s** |
| Quality Hash | `702229209b4f10d1` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 46, 65535, 46, 65535, 46, 46, 65535, 46, 46, 65535, 46, 46, 65535, 46, 46]
```

</details>


### 2026-01-04 09:26:36 - Debug

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **4549.89 ms** |
| **Prefill** | **88.78 tok/s** |
| **Generation** | **39.30 tok/s** |
| Quality Hash | `bc667ec83d759800` |

<details><summary>Quality tokens (first 16)</summary>

```
[332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332, 332]
```

</details>


### 2026-01-04 09:27:59 - Fixed discount loading

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **2603.25 ms** |
| **Prefill** | **389.75 tok/s** |
| **Generation** | **101.68 tok/s** |
| Quality Hash | `438287259638b200` |

<details><summary>Quality tokens (first 16)</summary>

```
[2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361]
```

</details>


### 2026-01-04 09:31:35 - Fixed threadgroup position

| Metric | Value |
|--------|-------|
| Model | 2.9b-Q8_0.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **2779.35 ms** |
| **Prefill** | **61.30 tok/s** |
| **Generation** | **103.63 tok/s** |
| Quality Hash | `4382872596391595` |

<details><summary>Quality tokens (first 16)</summary>

```
[2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361, 2361]
```

</details>


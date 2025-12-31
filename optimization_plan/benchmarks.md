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

## Batched GPU Uploads

### 2025-12-31 02:16:41 - Baseline after removing per-layer GPU sync

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7519.07 ms** |
| **Prefill** | **151.74 tok/s** |
| **Generation** | **43.52 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Lazy Loading

### 2025-12-31 02:21:46 - Baseline before lazy mmap implementation

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6477.41 ms** |
| **Prefill** | **151.58 tok/s** |
| **Generation** | **43.54 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2025-12-31 02:27:15 - After implementing lazy mmap GPU buffers

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6251.56 ms** |
| **Prefill** | **151.78 tok/s** |
| **Generation** | **43.58 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


### 2025-12-31 02:32:35 - After implementing lazy GPU buffer loading with LazyMatrix

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7016.52 ms** |
| **Prefill** | **151.66 tok/s** |
| **Generation** | **43.40 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Zero-Copy Lazy Loading

### 2025-12-31 03:46:26 - Using Arc mmap for zero-copy lazy loading

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7032.45 ms** |
| **Prefill** | **152.38 tok/s** |
| **Generation** | **44.30 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

### 2025-12-31 04:42:19 - Using Arc<Mmap> for zero-copy lazy loading

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7557.06 ms** |
| **Prefill** | **152.59 tok/s** |
| **Generation** | **43.90 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


## Zero-Copy Mmap

### 2025-12-31 04:16:28 - Using Arc<Mmap> for true zero-copy lazy loading

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7361.74 ms** |
| **Prefill** | **151.82 tok/s** |
| **Generation** | **43.68 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Zero-Copy Check

### 2025-12-31 04:18:05 - With pointer validation

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6612.49 ms** |
| **Prefill** | **151.53 tok/s** |
| **Generation** | **44.48 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## True Lazy Loading

### 2025-12-31 04:46:56 - Deferred GPU upload with LazyMatrix in Att/Ffn

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7794.49 ms** |
| **Prefill** | **151.50 tok/s** |
| **Generation** | **43.50 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Lazy Head Weight

### 2025-12-31 04:49:21 - Deferred GPU upload for head.weight (168M elements)

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6517.50 ms** |
| **Prefill** | **152.02 tok/s** |
| **Generation** | **43.18 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## Fp16 Zero-Copy

### 2025-12-31 05:00:35 - Added Fp16ZeroCopy for deferred Fp16 matrix loading

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **7348.68 ms** |
| **Prefill** | **145.94 tok/s** |
| **Generation** | **39.59 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>

## More Lazy Matrices

### 2025-12-31 05:04:58 - w1,w2,a1,a2,g1,g2,v1,v2 now use LazyMatrix

| Metric | Value |
|--------|-------|
| Model | rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf |
| Version | V7 |
| Layers | 32 |
| Embedding | 2560 |
| GPU | Apple M2 Max |
| **Load Time** | **6661.41 ms** |
| **Prefill** | **146.06 tok/s** |
| **Generation** | **39.89 tok/s** |
| Quality Hash | `3895ad4add71cff0` |

<details><summary>Quality tokens (first 16)</summary>

```
[33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129, 33, 3319, 153, 129]
```

</details>


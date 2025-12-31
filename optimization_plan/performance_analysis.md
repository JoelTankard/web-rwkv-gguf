# Performance Analysis: WebGPU vs llama.cpp

## Benchmark Results

| Metric             | WebGPU (web-rwkv)  | llama.cpp          | Difference           |
| ------------------ | ------------------ | ------------------ | -------------------- |
| **Model Load**     | ~10s               | ~250ms             | llama.cpp 40x faster |
| **Tokenize**       | 307µs              | 936µs              | WebGPU 3x faster     |
| **Generation**     | 6.54s (30.6 tok/s) | 9.00s (11.4 tok/s) | WebGPU 3x faster     |
| **Doc State**      | 950.8ms            | 296.1ms            | llama.cpp 3x faster  |
| **Field Analysis** | 1.84s              | 563.3ms            | llama.cpp 3x faster  |

## Root Cause Analysis

### 1. Model Loading (llama.cpp wins: 40x)

**Why llama.cpp is faster:**

-   Uses memory-mapped files (mmap) - the OS lazily loads model weights as needed
-   No upfront GPU shader compilation
-   GGUF format is designed for fast loading

**Why WebGPU is slower:**

-   Compiles WGSL compute shaders at startup
-   Allocates GPU buffers upfront
-   Transfers model weights to GPU memory

**Opportunity:** WebGPU could potentially use lazy loading or pre-compiled shader caches.

### 2. Tokenization (WebGPU wins: 3x)

**Why WebGPU is faster:**

-   Uses a pre-compiled Rust tokenizer (via web-rwkv-py)
-   Direct memory access, no Python overhead

**Why llama.cpp is slower:**

-   Python bindings add overhead
-   Tokenizer runs through ctypes/cffi layer

**Opportunity:** This is already optimal in WebGPU. llama.cpp could improve with native tokenizer calls.

### 3. Token Generation (WebGPU wins: 3x)

**Why WebGPU is faster:**

-   Custom WGSL compute shaders optimized specifically for RWKV's WKV (linear attention) operation
-   Hand-tuned for RWKV's unique architecture
-   Efficient GPU memory access patterns

**Why llama.cpp is slower:**

-   Uses general-purpose Metal kernels designed for transformers
-   RWKV support is newer and less optimized
-   WKV operation not as efficiently implemented

**Opportunity:** This is the core strength of web-rwkv. Keep using it for generation.

### 4. Doc State & Field Analysis (llama.cpp wins: 3x)

**Why llama.cpp is faster:**

-   Native embedding extraction via `model.embed()`
-   Efficient batch processing
-   Optimized for embedding operations

**Why WebGPU is slower:**

-   State extraction requires full forward pass
-   Custom state serialization overhead
-   Not optimized for embedding-style operations

**Opportunity:** Investigate how llama.cpp handles embeddings and apply similar techniques to web-rwkv.

## Key Insights

1. **Different strengths for different operations** - Neither backend is universally better
2. **Architecture matters** - web-rwkv is purpose-built for RWKV, llama.cpp is general-purpose
3. **Loading vs Runtime tradeoff** - llama.cpp trades upfront work for lazy loading
4. **Embedding operations** - llama.cpp has better embedding infrastructure

## Recommendations

See [plan.md](./plan.md) for the action plan to bring the best of both worlds together.

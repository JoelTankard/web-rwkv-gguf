# GGUF Support Implementation Plan

## Goal

Add GGUF file format support to reduce memory footprint during model loading by enabling direct loading of pre-quantized weights without FP16 staging.

## Benefits

-   **Reduced peak VRAM**: No FP16 staging buffer needed during quantization
-   **Smaller disk/download size**: Pre-quantized models are 2-4x smaller
-   **Lower system RAM pressure**: Smaller mmap footprint
-   **Ecosystem compatibility**: Share models with llama.cpp users

## Architecture Overview

### Current Flow (SafeTensors)

```
Disk (FP16) → mmap → GPU (FP16) → Quantize on GPU → GPU (Quantized)
Peak VRAM: FP16 + Quantized buffer
```

### New Flow (GGUF Pre-Quantized)

```
Disk (Quantized) → mmap → GPU (Quantized)
Peak VRAM: Quantized only
```

## Implementation Phases

### Phase 1: GGUF Parser Core ✅

-   [x] Create `src/runtime/gguf.rs` module
-   [x] Implement GGUF header parsing (magic, version, tensor_count, metadata_kv_count)
-   [x] Implement metadata key-value parsing (all value types)
-   [x] Implement tensor info parsing (name, dimensions, type, offset)
-   [x] Handle alignment and padding

### Phase 2: Reader Trait Implementation ✅

-   [x] Implement `Reader` trait for `GgufReader`
-   [x] Map GGUF tensor names to SafeTensors-style names (RWKV naming convention)
-   [x] Handle tensor data access via mmap offsets

### Phase 3: Basic Quantization Support ✅

-   [x] F16/BF16/F32 native loading
-   [x] Q8_0 dequantization to F16 (CPU-side)
-   [x] Q4_0 dequantization to F16 (CPU-side)
-   [x] Benchmark example comparing .st vs .gguf

### Phase 4: K-Quant Support ✅

-   [x] Implement Q4_K dequantization (super-blocks with nested quantization)
-   [x] Implement Q5_K dequantization
-   [x] Implement Q6_K dequantization
-   [x] Implement Q3_K dequantization
-   [x] Implement Q2_K dequantization
-   [ ] Update conversion script to support K-quants

### Phase 5: Direct GPU Quantized Loading ✅

-   [x] Analyze block size differences:
    -   GGUF Q8_0: 32 elements/block (34 bytes: 2B scale + 32B data)
    -   web-rwkv Int8: 128 elements/block (min/max per block)
-   [x] Option A: Repack GGUF Q8_0 → web-rwkv Int8 on CPU (saves F16 staging)
-   [x] Repack GGUF Q4_0 → web-rwkv NF4 on CPU
-   [x] Repack GGUF K-quants (Q4_K, Q5_K, Q6_K) → web-rwkv Int8 on CPU
-   [x] Add `quantized_tensor` method to Reader trait
-   [x] Add `try_load_matrix_direct` to Loader for direct quantized loading
-   [x] Bypass F16 staging for Q8_0→Int8, Q4_0→NF4, and K-quants→Int8

### Phase 6: Integration & Testing ✅

-   [x] Add GGUF detection in examples (chat)
-   [x] Test with RWKV GGUF models (Q8_0 tested, working)
-   [x] Benchmark memory usage vs SafeTensors path
-   [x] Document GGUF usage in README

### Phase 7: Native Quantized Matmul Shaders

**Goal:** Implement true quantized matrix multiplication that operates directly on packed 4-bit data without full dequantization, achieving actual compute speedup (not just memory savings).

**Current Problem:** GGUF Q4_K tensors are dequantized to F16 at load time, so inference runs at F16 speed despite smaller file/RAM footprint.

**Approach:** Extend existing web-rwkv WebGPU/WGSL infrastructure (NOT a new engine):

-   Add `Matrix::Q4K` variant alongside existing `Fp16`, `Int8`, `Fp4`
-   Create `matmul_mat_q4k.wgsl` following patterns in `src/shaders/`
-   Integrate via `TensorOp::matmul_mat_q4k` like existing quantized matmul ops

**Reference (for dequant math only):** llama.cpp Vulkan shaders, MLC-LLM/WebLLM

**Implementation Steps:**

-   [ ] Create `Matrix::Q4K` variant in `tensor/matrix.rs`
-   [ ] Add `matmul_mat_q4k.wgsl` shader with inline dequantization
-   [ ] Implement `TensorOp::matmul_mat_q4k` in `tensor/ops.rs`
-   [ ] Update loader to create `Matrix::Q4K` directly from GGUF
-   [ ] Benchmark against F16 path (target: 1.5-2x speedup on memory-bound ops)
-   [ ] Extend to Q5_K, Q6_K, Q8_0 formats

## GGUF File Structure Reference

```
┌─────────────────────────────────────┐
│ Header                              │
│   magic: u32 (0x46554747 "GGUF")    │
│   version: u32 (3)                  │
│   tensor_count: u64                 │
│   metadata_kv_count: u64            │
├─────────────────────────────────────┤
│ Metadata KV Pairs                   │
│   [key: string, type: u32, value]   │
├─────────────────────────────────────┤
│ Tensor Infos                        │
│   [name, n_dims, dims[], type, off] │
├─────────────────────────────────────┤
│ Padding to ALIGNMENT                │
├─────────────────────────────────────┤
│ Tensor Data (aligned)               │
└─────────────────────────────────────┘
```

## GGUF Quantization Types

| GGUF Type | Value | Block Size | Description    | Status             |
| --------- | ----- | ---------- | -------------- | ------------------ |
| F32       | 0     | 1          | 32-bit float   | ✅ Supported       |
| F16       | 1     | 1          | 16-bit float   | ✅ Supported       |
| Q4_0      | 2     | 32         | 4-bit (simple) | ✅ Dequant to F16  |
| Q4_1      | 3     | 32         | 4-bit + min    | ❌ Not implemented |
| Q5_0      | 6     | 32         | 5-bit (simple) | ❌ Not implemented |
| Q5_1      | 7     | 32         | 5-bit + min    | ❌ Not implemented |
| Q8_0      | 8     | 32         | 8-bit (simple) | ✅ Dequant to F16  |
| Q8_1      | 9     | 32         | 8-bit + min    | ❌ Not implemented |
| Q2_K      | 10    | 256        | 2-bit K-quant  | ✅ Dequant to F16  |
| Q3_K      | 11    | 256        | 3-bit K-quant  | ✅ Dequant to F16  |
| Q4_K      | 12    | 256        | 4-bit K-quant  | ✅ Dequant to F16  |
| Q5_K      | 13    | 256        | 5-bit K-quant  | ✅ Dequant to F16  |
| Q6_K      | 14    | 256        | 6-bit K-quant  | ✅ Dequant to F16  |
| Q8_K      | 15    | 256        | 8-bit K-quant  | ❌ Not implemented |
| BF16      | 30    | 1          | bfloat16       | ✅ Supported       |

## RWKV Tensor Name Mapping

GGUF uses different naming than SafeTensors. Need to map:

| SafeTensors Name          | GGUF Name (likely)       |
| ------------------------- | ------------------------ |
| `emb.weight`              | `token_embd.weight`      |
| `blocks.N.ln1.weight`     | `blk.N.attn_norm.weight` |
| `blocks.N.att.key.weight` | `blk.N.attn_k.weight`    |
| `blocks.N.ffn.key.weight` | `blk.N.ffn_up.weight`    |
| `ln_out.weight`           | `output_norm.weight`     |
| `head.weight`             | `output.weight`          |

_Note: Actual RWKV GGUF naming may vary - will verify with real files._

## Files to Create/Modify

### New Files

-   `src/runtime/gguf.rs` - GGUF parser and Reader implementation

### Modified Files

-   `src/runtime/mod.rs` - Export gguf module
-   `src/runtime/loader.rs` - Add quantized loading methods
-   `examples/chat.rs` - Support .gguf file extension
-   `Cargo.toml` - No new dependencies needed (using bytemuck)

## Risk Mitigation

1. **RWKV GGUF availability**: Check HuggingFace for existing RWKV GGUF models
2. **Quantization format mismatch**: GGUF Q8_0 block format may differ from web-rwkv Int8
    - Solution: Implement conversion layer if needed
3. **Tensor naming**: RWKV may use custom GGUF tensor names
    - Solution: Support both standard and RWKV-specific names

## Current Status

**Implemented:**

-   GGUF file format parser (header, metadata, tensor info)
-   Reader trait implementation for GgufReader
-   RWKV tensor name mapping (GGUF ↔ SafeTensors)
-   Virtual tensor slicing for fused `time_mix_lerp_fused` → individual `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` tensors
-   Tensor shape handling:
    -   2D+ tensors: dimension reversal for SafeTensors convention
    -   1D tensors: convert to `[x, 1]` for proper `from_slice_rev` handling
    -   `r_k` tensor: reshape from 1D `[768]` to 2D `[num_head, head_dim]` by inferring from `a1` tensor
-   F32/BF16 → F16 conversion during tensor loading
-   Q8_0/Q4_0 → F16 dequantization during tensor loading (supports pre-quantized GGUF models)
-   K-quant dequantization (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) → F16 for better compression ratios
-   Direct quantized loading: Q8_0/Q4_K/Q5_K/Q6_K → Int8, Q4_0 → NF4 (bypasses F16 staging)
-   Chat example updated with streaming output and tokens/second display
-   Benchmark example (`bench_format`) for comparing .st vs .gguf performance
-   All tests passing, model produces coherent output

**Pending:**

-   Update conversion script to support K-quants

## Benchmark Results (M2 Max, 0.1B model)

| Format      | File Size | Load Time | RAM Δ    | Prefill  | Generation |
| ----------- | --------- | --------- | -------- | -------- | ---------- |
| SafeTensors | 364.5 MB  | 1896 ms   | 865.9 MB | 2971 t/s | 188.7 t/s  |
| GGUF Q8_0   | 198.8 MB  | 2043 ms   | 221.1 MB | 2984 t/s | 189.1 t/s  |

**Key findings:**

-   **45.5% smaller** file size with Q8_0
-   **74.5% less RAM** during loading
-   **No performance regression** (identical inference speed)
-   Load time ~8% slower due to CPU dequantization

## Success Criteria

-   [x] Parse GGUF file format correctly
-   [x] Load RWKV F16 GGUF models
-   [x] Inference produces coherent output matching SafeTensors quality
-   [x] No performance regression (~140 tok/s on M2 Max)
-   [x] Load RWKV pre-quantized GGUF models (Q8_0, Q4_0) via dequantization
-   [x] K-quant support (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
-   [x] Direct quantized loading to bypass F16 staging (Q8_0→Int8, Q4_0→NF4)

## Usage

```bash
# Load a GGUF model (F16)
cargo run --release --example chat -- --model /path/to/model.gguf

# Load a SafeTensors model (existing)
cargo run --release --example chat -- --model /path/to/model.st
```

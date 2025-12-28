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

### Phase 3: Quantization Type Mapping (Future)

-   [ ] Map GGUF types to web-rwkv Matrix variants:
    -   `GGML_TYPE_F16` → Load as FP16, use existing path ✅
    -   `GGML_TYPE_Q8_0` → Direct load to `Matrix::Int8`
    -   `GGML_TYPE_Q4_0/Q4_K` → Direct load to `Matrix::Fp4`
-   [ ] Implement dequantization for unsupported types (fallback to FP16)

### Phase 4: Direct Quantized Loading (Future)

-   [ ] Add `load_matrix_quantized` methods to `Loader`
-   [ ] Bypass FP16 staging for compatible quantization types
-   [ ] Handle block size and absmax differences between GGUF and web-rwkv formats

### Phase 5: Integration & Testing

-   [x] Add GGUF detection in examples (chat)
-   [ ] Test with RWKV GGUF models from HuggingFace
-   [ ] Benchmark memory usage vs SafeTensors path
-   [ ] Document GGUF usage in README

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

## GGUF Quantization Types (Relevant)

| GGUF Type | Value | Description    | web-rwkv Mapping       |
| --------- | ----- | -------------- | ---------------------- |
| F32       | 0     | 32-bit float   | Convert to F16         |
| F16       | 1     | 16-bit float   | Matrix::Fp16           |
| Q4_0      | 2     | 4-bit (simple) | Matrix::Fp4 (convert)  |
| Q8_0      | 8     | 8-bit (simple) | Matrix::Int8 (convert) |
| BF16      | 30    | bfloat16       | Convert to F16         |

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
-   Chat example updated to detect and load .gguf files
-   All tests passing

**Pending (for full memory benefit):**

-   Direct loading of pre-quantized GGUF tensors (Q4_K, Q8_0)
-   Currently loads F16 GGUF files through existing path

## Success Criteria

-   [x] Parse GGUF file format correctly
-   [x] Load RWKV F16 GGUF models
-   [ ] Load RWKV pre-quantized GGUF models without FP16 staging
-   [ ] Peak VRAM reduced by ~50% for Q4 models
-   [ ] Inference produces identical results to SafeTensors path
-   [ ] No performance regression in inference speed

## Usage

```bash
# Load a GGUF model (F16)
cargo run --release --example chat -- --model /path/to/model.gguf

# Load a SafeTensors model (existing)
cargo run --release --example chat -- --model /path/to/model.st
```

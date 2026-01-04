# Inference Optimization Research Overview

This folder contains research documents for optimizing the web-rwkv-gguf inference pipeline.

## Inference Flow Summary

```
User Input → Tokenize → RnnInput
                           ↓
                    RnnJob::load()     [CPU→GPU: embed lookup]
                           ↓
                    dispatch()          [Build GPU commands]
                           ↓
                    submit()            [Execute on GPU]
                           ↓
                    back()              [GPU→CPU: read logits]
                           ↓
                    Sample → Token → Output
```

## Research Documents

| #   | Topic                                                            | Potential Impact | Complexity  | Priority     |
| --- | ---------------------------------------------------------------- | ---------------- | ----------- | ------------ |
| 01  | [Embedding Lookup](./01_embedding_lookup_optimization.md)        | 0.2-1ms/token    | Low-Medium  | Medium       |
| 02  | [Command Buffer Batching](./02_command_buffer_batching.md)       | 5-20%            | Low         | High         |
| 03  | [Fused Layer Operations](./03_fused_layer_operations.md)         | 20-50%           | Medium-High | High         |
| 04  | [Matmul Shader Optimization](./04_matmul_shader_optimization.md) | 10-100% batch    | Medium      | High         |
| 05  | [Async GPU-CPU Overlap](./05_async_gpu_cpu_overlap.md)           | 10-40%           | Medium-High | Medium       |
| 06  | [State Management](./06_state_management_optimization.md)        | 5-20%            | Low-Medium  | Medium       |
| 07  | [Memory Layout](./07_memory_layout_optimization.md)              | 5-25%            | Medium      | Low          |
| 08  | [Metal Backend](./08_metal_backend_integration.md)               | 2-4x (macOS)     | High        | High (macOS) |
| 09  | [Quantization](./09_quantization_improvements.md)                | 10-40%           | Medium      | High         |
| 10  | [Model Loading](./10_model_loading_optimization.md)              | 30-80% load      | Medium      | Medium       |

## Quick Wins (Low Effort, High Impact)

1. **Remove Sep markers** (02) - Test if `TensorOp::Sep` is actually needed
2. **SIMD F32→F16** (10) - Speed up model loading conversion
3. **NF4 LUT optimization** (09) - Use constant lookup table

## High Impact (Medium-High Effort)

1. **Fused token shift + LayerNorm** (03) - 7 dispatches → 1
2. **Tiled matmul with shared memory** (04) - Major batch speedup
3. **GPU-side sampling** (05) - Avoid 260KB readback per token
4. **Pure Metal layer execution** (08) - Amortize sync overhead

## Experimental (High Effort, Uncertain)

1. **Full Metal forward pass** (08) - Maximum macOS performance
2. **Persistent kernel** (03) - Single kernel per layer
3. **Continuous batching** (05) - Throughput optimization

## Benchmarking

Before implementing any optimization, establish baseline:

```bash
# Single token latency
cargo run --release --example bench -- \
    --model /Users/joel/Dev/Experimental/web-rwkv-gguf/assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf \
    --prompt "Hello" \
    --tokens 100

# Batch throughput
cargo run --release --example bench -- \
    --model $MODEL \
    --prompt "Hello" \
    --tokens 100 \
    --batch 64
```

## Implementation Order Recommendation

### Phase 1: Quick Wins

1. Test removing Sep markers
2. Implement SIMD conversion
3. Optimize NF4 lookup table

### Phase 2: Core Optimizations

1. Fused token shift + LayerNorm shader
2. Tiled matmul for batch inference
3. GPU-side token sampling

### Phase 3: Platform-Specific

1. Pure Metal layer execution (macOS)
2. Re-enable Q4_K native shaders
3. Parallel model loading

### Phase 4: Advanced

1. Continuous batching
2. State compression
3. Full Metal forward pass

## Related Existing Work

-   `optimization_plan/` - Previous optimization attempts
-   `optimization_plan_v2/` - HAL migration analysis
-   `optimization_plan_v3/metal_integration_handoff.md` - Metal backend status
-   `research/quantization_speedup/` - Quantization research
-   `benchmarks/` - Existing benchmark results

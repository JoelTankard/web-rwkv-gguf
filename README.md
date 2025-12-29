# Web-RWKV-GGUF

A fork of [web-rwkv](https://github.com/cryscan/web-rwkv) adding **GGUF format support** for RWKV models.

## What's New in This Fork

This fork adds the ability to load RWKV models in [GGUF format](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md), the model format used by llama.cpp.

### Key Features

-   **GGUF file format parsing** — Full parser for GGUF v3 headers, metadata, and tensor info
-   **RWKV tensor name mapping** — Automatic translation between GGUF and SafeTensors naming conventions
-   **Multiple data types** — Supports F16, BF16, F32, Q8_0, and Q4_0 GGUF tensors
-   **RWKV V7 support** — Handles V7-specific tensors including fused `time_mix_lerp_fused` → individual `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` slicing
-   **Drop-in replacement** — Use `.gguf` files anywhere you'd use `.st` files

### Current Status

| Feature                      | Status                           |
| ---------------------------- | -------------------------------- |
| GGUF parsing                 | ✅ Complete                      |
| F16/BF16/F32 loading         | ✅ Complete                      |
| Q8_0/Q4_0 dequantization     | ✅ Complete (dequantizes to F16) |
| RWKV V4-V7 compatibility     | ✅ Tested                        |
| Direct quantized GPU loading | 🔜 Planned                       |

> **Note:** Pre-quantized GGUF tensors (Q8_0, Q4_0) are currently dequantized to F16 during loading, then optionally re-quantized on GPU. Direct loading to web-rwkv's Int8/Fp4 format is planned to reduce peak VRAM usage.

## Quick Start

```bash
# Load a GGUF model
cargo run --release --example chat -- --model /path/to/model.gguf

# Load a SafeTensors model (unchanged from upstream)
cargo run --release --example chat -- --model /path/to/model.st
```

## Converting Models to GGUF

Use the included conversion script:

```bash
python assets/scripts/convert_hf_to_gguf.py /path/to/model.pth /path/to/rwkv_vocab.txt --outtype f16
```

Supported output types: `f16`, `bf16`, `f32`, `q8_0`

## Upstream Project

This fork is based on **[web-rwkv](https://github.com/cryscan/web-rwkv)** by [@cryscan](https://github.com/cryscan) — a pure WebGPU inference engine for RWKV language models.

For full documentation on web-rwkv features (batched inference, hooks, quantization, WASM support, etc.), see the [upstream README](https://github.com/cryscan/web-rwkv#readme).

### Upstream Features (Inherited)

-   No CUDA/Python dependencies
-   Nvidia/AMD/Intel GPU support (Vulkan/Dx12/OpenGL)
-   WASM/browser support
-   Batched inference
-   Int8 and Float4 quantization
-   LoRA merging at load time
-   RWKV V4–V7 support
-   Inference hooks

## License

Same as upstream — see [LICENSE](./LICENSE).

## Credits

-   **[web-rwkv](https://github.com/cryscan/web-rwkv)** — The original project this fork is based on
-   **[@cryscan](https://github.com/cryscan)** — Author of web-rwkv
-   **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — GGUF format specification

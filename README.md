# Web-RWKV-GGUF

A fork of [web-rwkv](https://github.com/cryscan/web-rwkv) adding **GGUF format support** for RWKV models.

## What's New in This Fork

This fork adds the ability to load RWKV models in [GGUF format](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md), the model format used by llama.cpp.

### Key Features

-   **GGUF file format parsing** — Full parser for GGUF v3 headers, metadata, and tensor info
-   **RWKV tensor name mapping** — Automatic translation between GGUF and SafeTensors naming conventions
-   **Multiple data types** — Supports F16, BF16, F32, and quantized GGUF tensors
-   **RWKV V7 support** — Handles V7-specific tensors including fused `time_mix_lerp_fused` → individual `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` slicing
-   **Drop-in replacement** — Use `.gguf` files anywhere you'd use `.st` files

### Supported Quantizations

| GGUF Type | Status     | Notes                           |
| --------- | ---------- | ------------------------------- |
| F16       | ✅ Native  | Direct loading                  |
| BF16      | ✅ Native  | Converts to F16                 |
| F32       | ✅ Native  | Converts to F16                 |
| Q4_K      | ✅ Native  | Inline dequantization in shader |
| Q5_K      | ❔ Native  | Inline dequantization in shader |
| Q6_K      | ❔ Native  | Inline dequantization in shader |
| Q8_0      | ❔ Native  | Inline dequantization in shader |
| Q2_K      | ❔ Dequant | Dequantizes to F16 at load time |
| Q3_K      | ❔ Dequant | Dequantizes to F16 at load time |
| Q4_0      | ❔ Dequant | Dequantizes to F16 at load time |

> **Recommended:** Q4_K_M or Q5_K_M for best balance of quality and VRAM usage.

## Quick Start

```bash
# Load a GGUF model
cargo run --release --example chat -- --model /path/to/model.gguf

# Load a SafeTensors model (unchanged from upstream)
cargo run --release --example chat -- --model /path/to/model.st
```

## Converting Models

### `.pth` to Q4_K_M GGUF (Recommended)

To get a Q4_K_M quantized GGUF from a PyTorch `.pth` file:

```bash
# 1. Install dependencies
pip install torch numpy gguf

# 2. Convert .pth to F16 GGUF
python assets/scripts/convert_hf_to_gguf.py /path/to/model.pth assets/models/rwkv_vocab_v20230424.json --outtype f16

# 3. Quantize to Q4_K_M using llama.cpp's llama-quantize
llama-quantize /path/to/model-F16.gguf /path/to/model-Q4_K_M.gguf Q4_K_M
```

Supported `--outtype` values: `f16`, `bf16`, `f32`, `q8_0`, `auto`

### `.pth` to SafeTensors (for bench_format testing)

To convert a `.pth` file to SafeTensors format for benchmarking:

```bash
# Convert .pth to .st
python assets/scripts/convert_safetensors.py --input /path/to/model.pth --output /path/to/model.st

# Run bench_format to compare performance
cargo run --release --example bench_format -- --model /path/to/model.st
```

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

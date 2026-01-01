# WebGPU Fallback and Cross-Platform Support

## Design Philosophy

WebGPU is the universal fallback that ensures the library works everywhere:

-   All desktop platforms (macOS, Windows, Linux)
-   All browsers with WebGPU support (Chrome 113+, Firefox 121+, Safari 17+)
-   Any GPU vendor (NVIDIA, AMD, Intel, Apple, Qualcomm)

## When WebGPU is Used

| Scenario                        | Backend Used                          |
| ------------------------------- | ------------------------------------- |
| macOS + `metal-backend` feature | Metal for Q4K matmul, WebGPU for rest |
| macOS without feature           | WebGPU only                           |
| Windows/Linux (no CUDA)         | WebGPU only                           |
| Browser (any platform)          | WebGPU only                           |
| VM without GPU passthrough      | WebGPU (software renderer)            |

## WebGPU Optimizations

Even when using WebGPU as fallback, these optimizations apply:

### 1. Subgroup Operations

When available, use subgroup operations for faster reductions:

```rust
// Check at context creation
let supports_subgroups = adapter.features().contains(Features::SUBGROUP);

// Use optimized shader variant
if supports_subgroups {
    include_str!("shaders/matmul_vec_q4k_subgroup.wgsl")
} else {
    include_str!("shaders/matmul_vec_q4k.wgsl")
}
```

### 2. F16 Support

Use f16 when available for reduced memory bandwidth:

```rust
let supports_f16 = adapter.features().contains(Features::SHADER_F16);

// F16 reduces memory traffic by 2x for activations
if supports_f16 {
    include_str!("shaders/matmul_vec_q4k_f16.wgsl")
}
```

### 3. Workgroup Size Tuning

Optimal workgroup size varies by GPU:

```rust
fn optimal_workgroup_size(adapter: &Adapter) -> u32 {
    let info = adapter.get_info();
    match info.backend {
        Backend::Metal => 256,      // Apple GPUs prefer larger
        Backend::Vulkan => 128,     // NVIDIA/AMD
        Backend::Dx12 => 128,       // Windows
        _ => 64,                    // Safe default
    }
}
```

## Browser-Specific Considerations

### Feature Detection

```javascript
// Check WebGPU availability
if (!navigator.gpu) {
    console.error("WebGPU not supported");
    return;
}

const adapter = await navigator.gpu.requestAdapter();
const features = adapter.features;

// Check for optional features
const hasF16 = features.has("shader-f16");
const hasSubgroups = features.has("subgroups");
```

### Memory Limits

Browsers impose stricter memory limits:

```rust
// Query limits
let limits = adapter.limits();
let max_buffer = limits.max_buffer_size;  // Often 256MB in browser

// For large models, may need to:
// 1. Stream weights from disk/network
// 2. Use smaller batch sizes
// 3. Quantize more aggressively
```

### WASM Considerations

When compiling to WASM:

```toml
[features]
web = []  # Excludes tokio, uses wasm-bindgen

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
```

## Graceful Degradation

The system should degrade gracefully:

```rust
pub fn select_best_available(adapter: &Adapter) -> InferenceConfig {
    let mut config = InferenceConfig::default();

    // Try native acceleration first
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    if let Some(metal) = MetalContext::new() {
        config.matmul_backend = Backend::Metal;
        log::info!("Using Metal backend (4-5x speedup expected)");
        return config;
    }

    // Check WebGPU capabilities
    let features = adapter.features();

    if features.contains(Features::SUBGROUP) {
        config.use_subgroups = true;
        log::info!("Subgroup operations enabled");
    }

    if features.contains(Features::SHADER_F16) {
        config.use_f16 = true;
        log::info!("F16 shaders enabled");
    }

    config.matmul_backend = Backend::WebGpu;
    log::info!("Using WebGPU backend");
    config
}
```

## Performance Expectations

| Platform               | Backend | Expected Speed |
| ---------------------- | ------- | -------------- |
| macOS M-series         | Metal   | 150-200 tok/s  |
| macOS M-series         | WebGPU  | 40-50 tok/s    |
| Windows RTX 4090       | WebGPU  | 60-80 tok/s    |
| Windows RTX 3080       | WebGPU  | 40-60 tok/s    |
| Browser (M-series)     | WebGPU  | 30-40 tok/s    |
| Browser (discrete GPU) | WebGPU  | 20-40 tok/s    |

_Note: These are estimates for the 2.9B Q4_K model. Actual performance varies._

## Testing Cross-Platform

```bash
# Test on macOS with Metal
cargo run --release --features metal-backend --example benchmark

# Test on macOS without Metal (WebGPU only)
cargo run --release --example benchmark

# Test WASM build
wasm-pack build --target web
# Serve and test in browser
```

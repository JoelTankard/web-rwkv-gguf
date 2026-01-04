# Model Loading Optimization

## Current Implementation

Location: `src/runtime/loader.rs`, `src/runtime/v7.rs:1298-1474` (`build_v7`)

Loading flow:

1. Parse GGUF/SafeTensors header
2. For each tensor: read → convert (F32→F16) → upload to GPU
3. Apply LoRA blending if present
4. Optionally quantize (Int8/NF4)

## Problem

1. **Sequential loading**: Tensors loaded one at a time
2. **CPU conversion bottleneck**: F32→F16 conversion is slow (~600ms for embeddings)
3. **Memory spikes**: Full tensor in CPU memory during upload
4. **No streaming**: Must load entire model before inference

## Optimization Ideas

### Idea 1: Parallel Tensor Loading

Load multiple tensors concurrently:

```rust
async fn build_v7_parallel(self) -> Result<Model, LoaderError> {
    let loader = Arc::new(self.loader);

    // Load all layers in parallel
    let layer_futures: Vec<_> = (0..info.num_layer)
        .map(|layer| {
            let loader = loader.clone();
            tokio::spawn(async move {
                load_layer(&loader, layer).await
            })
        })
        .collect();

    // Wait for all layers
    let layers: Vec<Layer> = futures::future::try_join_all(layer_futures).await?;

    Ok(Model { layers, .. })
}
```

**Benefit**: Utilize multiple CPU cores for conversion

### Idea 2: Streaming GPU Upload

Upload to GPU while still reading from disk:

```rust
async fn load_tensor_streaming(
    &self,
    name: &str,
) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
    let context = &self.context;

    // Get tensor metadata without loading data
    let shape = self.model.shape(name)?;
    let gpu_tensor: TensorGpu<f16, _> = context.tensor_init(shape);

    // Stream in chunks
    const CHUNK_SIZE: usize = 1024 * 1024; // 1MB chunks
    let mut offset = 0;

    while offset < shape.len() {
        let chunk_end = (offset + CHUNK_SIZE).min(shape.len());

        // Read chunk from disk
        let chunk = self.model.tensor_chunk(name, offset, chunk_end)?;

        // Convert F32→F16 (can be SIMD optimized)
        let chunk_f16 = convert_f32_to_f16_simd(&chunk);

        // Upload chunk to GPU (async)
        gpu_tensor.load_chunk(&chunk_f16, offset)?;

        offset = chunk_end;
    }

    Ok(gpu_tensor)
}
```

**Benefit**: Lower peak memory, faster perceived loading

### Idea 3: SIMD F32→F16 Conversion

Optimize the conversion bottleneck:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn convert_f32_to_f16_simd(input: &[f32]) -> Vec<f16> {
    let mut output = vec![f16::ZERO; input.len()];

    // Process 8 floats at a time with AVX
    let chunks = input.len() / 8;
    for i in 0..chunks {
        unsafe {
            let f32_vec = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let f16_vec = _mm256_cvtps_ph(f32_vec, 0);
            _mm_storeu_si128(
                output.as_mut_ptr().add(i * 8) as *mut __m128i,
                f16_vec,
            );
        }
    }

    // Handle remainder
    for i in (chunks * 8)..input.len() {
        output[i] = f16::from_f32(input[i]);
    }

    output
}
```

**Benefit**: 4-8x faster F32→F16 conversion

### Idea 4: Memory-Mapped Loading

Use mmap for zero-copy access to model file:

```rust
use memmap2::Mmap;

struct MmapLoader {
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
}

impl MmapLoader {
    fn tensor_data(&self, name: &str) -> &[u8] {
        let info = &self.tensors[name];
        &self.mmap[info.offset..info.offset + info.size]
    }

    fn load_to_gpu(&self, name: &str, context: &Context) -> TensorGpu<f16, ReadWrite> {
        let data = self.tensor_data(name);

        // For F16 tensors: direct upload (zero-copy)
        if info.dtype == Dtype::F16 {
            let f16_data: &[f16] = bytemuck::cast_slice(data);
            return TensorCpu::from_data(info.shape, f16_data).to(context);
        }

        // For F32: convert and upload
        // ...
    }
}
```

**Benefit**: Eliminates file read syscalls, OS handles caching

### Idea 5: Lazy Layer Loading

Load layers on-demand during first inference:

```rust
pub struct LazyModel {
    loader: Arc<Loader>,
    layers: Vec<OnceCell<Layer>>,
}

impl LazyModel {
    fn get_layer(&self, index: usize) -> &Layer {
        self.layers[index].get_or_init(|| {
            self.loader.load_layer(index).expect("failed to load layer")
        })
    }
}
```

**Benefit**: Fast startup, load during first prefill
**Tradeoff**: First inference is slower (but previous experiments showed 7% inference regression)

### Idea 6: Pre-quantized Model Cache

Cache quantized models to skip runtime quantization:

```rust
struct ModelCache {
    cache_dir: PathBuf,
}

impl ModelCache {
    fn get_or_quantize(
        &self,
        model_path: &Path,
        quant: Quant,
    ) -> Result<Model, LoaderError> {
        let cache_key = format!(
            "{}-{:?}-{}",
            model_path.file_name().unwrap().to_str().unwrap(),
            quant,
            model_hash(model_path),
        );
        let cache_path = self.cache_dir.join(&cache_key);

        if cache_path.exists() {
            // Load pre-quantized model (fast)
            return self.load_cached(&cache_path);
        }

        // Quantize and cache
        let model = self.load_and_quantize(model_path, quant)?;
        self.save_cached(&cache_path, &model)?;

        Ok(model)
    }
}
```

**Benefit**: Subsequent loads are much faster

### Idea 7: Direct GGUF Quantized Loading

For GGUF models, load quantized weights directly without dequantization:

```rust
// Already partially implemented - extend to more formats
fn try_load_matrix_direct(
    &self,
    context: &Context,
    name: &str,
    gguf_type: u32,
    raw_data: &[u8],
    quant: Quant,
) -> Result<Option<Matrix>, LoaderError> {
    match (GgmlType::from(gguf_type), quant) {
        // Existing: Q8_0 → Int8, Q4_0 → NF4

        // Add: Q4_K → native Q4K (skip dequant entirely)
        (GgmlType::Q4K, Quant::None) => {
            let matrix = load_q4k_native(context, raw_data, shape)?;
            Ok(Some(matrix))
        }

        // Add: Q5_K, Q6_K native support
        (GgmlType::Q5K, Quant::None) => { ... }
        (GgmlType::Q6K, Quant::None) => { ... }

        _ => Ok(None),
    }
}
```

**Benefit**: Skip dequant→requant cycle for GGUF models

## Estimated Impact

| Optimization        | Load Time Reduction | Memory Reduction | Complexity |
| ------------------- | ------------------- | ---------------- | ---------- |
| Parallel Loading    | 30-50%              | 0%               | Medium     |
| Streaming Upload    | 10-20%              | 50%+             | Medium     |
| SIMD Conversion     | 20-30%              | 0%               | Low        |
| Memory-Mapped       | 10-20%              | 30%              | Low        |
| Lazy Loading        | 80%+ startup        | 0%               | Medium     |
| Pre-quantized Cache | 50-70%              | 0%               | Medium     |
| Direct GGUF         | 40-60%              | 0%               | Medium     |

## Recommended Experiment

1. **First**: SIMD F32→F16 conversion - simple, big impact on conversion bottleneck
2. **Second**: Parallel tensor loading - utilize all CPU cores
3. **Third**: Direct GGUF loading for Q4_K - skip unnecessary conversions

## Profiling

```bash
# Measure current load time
time cargo run --release --example bench -- --model $MODEL --tokens 1

# Profile loading phases
RUST_LOG=info cargo run --release --example bench 2>&1 | grep -E "(load|convert|upload)"
```

## Files to Modify

-   `src/runtime/loader.rs`: Parallel loading, SIMD conversion
-   `src/runtime/gguf.rs`: Direct quantized loading
-   `src/runtime/v7.rs`: `build_v7` parallelization
-   `src/tensor/mod.rs`: Streaming upload support

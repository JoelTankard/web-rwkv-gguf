# Quantization Improvements

## Current Implementation

Supported quantization formats:

-   `Matrix::Fp16` - Full precision F16
-   `Matrix::Int8` - 8-bit integer with min/max scaling
-   `Matrix::Fp4` (NF4) - 4-bit with lookup table
-   `Matrix::Q8_0` - GGML Q8_0 format (native)

Q4_K support exists but is disabled (see memory about native Q4_K shaders).

## Problem

1. **Dequantization overhead**: Quantized matmul spends significant time dequantizing
2. **Limited format support**: Only a few GGML formats supported natively
3. **No mixed precision**: All layers use same quantization
4. **Activation quantization**: Activations always F16/F32

## Optimization Ideas

### Idea 1: Native Q4_K Matmul (Re-enable)

Q4_K shaders were implemented but may have issues. Re-enable and fix:

```rust
// In matrix.rs, add Q4K variant
pub enum Matrix {
    // ... existing variants
    Q4K {
        w: TensorGpu<u8, ReadWrite>,  // Raw Q4_K blocks
        s: TensorGpu<u8, ReadWrite>,  // Shape metadata
    },
}

// Q4_K block structure (144 bytes per 256 elements):
// - d: f16 (super-block scale)
// - dmin: f16 (super-block min)
// - scales: 12 bytes (8 sub-block scales/mins, 6-bit each)
// - qs: 128 bytes (256 4-bit values)
```

Shader optimization:

```wgsl
// Vectorized Q4_K dequantization
fn dequant_q4k_vec8(block: ptr<Q4KBlock>, idx: u32) -> array<f16, 8> {
    let d = block.d;
    let dmin = block.dmin;

    // Decode sub-block scale/min (6-bit packed)
    let sub_idx = idx / 32;
    let scale = decode_scale(block.scales, sub_idx);
    let min = decode_min(block.scales, sub_idx);

    // Unpack 8 4-bit values at once
    let packed = block.qs[idx / 2];
    let q4_lo = (packed >> vec4<u32>(0, 4, 8, 12)) & vec4<u32>(0xF);
    let q4_hi = (packed >> vec4<u32>(16, 20, 24, 28)) & vec4<u32>(0xF);

    // Dequantize: val = d * scale * q4 - dmin * min
    var result: array<f16, 8>;
    for (var i = 0u; i < 4u; i++) {
        result[i] = d * scale * f16(q4_lo[i]) - dmin * min;
        result[i + 4] = d * scale * f16(q4_hi[i]) - dmin * min;
    }
    return result;
}
```

### Idea 2: Fused Dequant-Matmul with Shared Memory

Cache dequantized values in shared memory:

```wgsl
var<workgroup> dequant_cache: array<vec4<f16>, 1024>;

@compute @workgroup_size(256)
fn matmul_q4k_cached() {
    let local_id = local_invocation_index;

    // Phase 1: Cooperative dequantization into shared memory
    for (var i = local_id; i < 1024u; i += 256u) {
        dequant_cache[i] = dequant_q4k_vec4(matrix, i);
    }
    workgroupBarrier();

    // Phase 2: Matmul from shared memory (fast)
    var acc = vec4<f32>(0.0);
    for (var k = 0u; k < K; k += 4u) {
        acc += dequant_cache[k/4] * input[k/4];
    }

    // ...
}
```

**Benefit**: Dequantize once, use multiple times for different output rows

### Idea 3: Mixed-Precision Layers

Use different quantization for different layers:

```rust
pub struct MixedPrecisionConfig {
    // First/last layers: higher precision (more sensitive)
    pub embed_quant: Quant::Fp16,
    pub head_quant: Quant::Fp16,

    // Middle layers: aggressive quantization
    pub layer_quant: HashMap<usize, Quant>,
}

impl MixedPrecisionConfig {
    pub fn default_for_model(num_layers: usize) -> Self {
        let mut layer_quant = HashMap::new();
        for i in 0..num_layers {
            let quant = if i < 2 || i >= num_layers - 2 {
                Quant::Int8  // First/last 2 layers: Int8
            } else {
                Quant::NF4   // Middle layers: NF4
            };
            layer_quant.insert(i, quant);
        }
        Self { layer_quant, .. }
    }
}
```

**Benefit**: Better quality/speed tradeoff

### Idea 4: Dynamic Quantization

Quantize activations on-the-fly:

```wgsl
// Dynamic INT8 activation quantization
fn quantize_activation(x: vec4<f32>) -> vec4<i8> {
    // Compute scale from max absolute value
    let max_abs = max(abs(x));
    let scale = 127.0 / max_abs;

    // Quantize
    return vec4<i8>(round(x * scale));
}

// INT8 matmul with dynamic activation quantization
@compute @workgroup_size(128)
fn matmul_int8_dynamic() {
    // Quantize input activation
    let input_q = quantize_activation(input[token]);
    let input_scale = compute_scale(input[token]);

    // INT8 dot product (fast on many GPUs)
    var acc = 0i;
    for (var k = 0u; k < K; k++) {
        acc += i32(weight_q[k]) * i32(input_q[k]);
    }

    // Dequantize output
    output[row] = f32(acc) * weight_scale * input_scale;
}
```

**Benefit**: Reduced memory bandwidth for activations

### Idea 5: Lookup Table Optimization

For NF4, optimize the lookup table access:

```wgsl
// Current: 16-element lookup table in uniform buffer
@group(0) @binding(3) var<uniform> quant: array<f32, 16>;

// Optimized: Pack into vec4s for better cache behavior
const QUANT_TABLE: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -0.696, -0.525, -0.395),
    vec4<f32>(-0.284, -0.185, -0.091, 0.0),
    vec4<f32>(0.080, 0.161, 0.246, 0.338),
    vec4<f32>(0.441, 0.563, 0.723, 1.0),
);

fn lookup_nf4(idx: u32) -> f32 {
    let vec_idx = idx >> 2u;
    let elem_idx = idx & 3u;
    return QUANT_TABLE[vec_idx][elem_idx];
}
```

**Benefit**: Constant lookup table, no memory access

### Idea 6: GGML Q5_K / Q6_K Support

Add support for higher-quality GGML formats:

```rust
// Q5_K: 5.5 bits per weight (better quality than Q4_K)
pub struct Q5KBlock {
    d: f16,           // Super-block scale
    dmin: f16,        // Super-block min
    scales: [u8; 12], // Sub-block scales (6-bit)
    qh: [u8; 32],     // High bits (1 bit per value)
    qs: [u8; 128],    // Low bits (4 bits per value)
}

// Q6_K: 6.5 bits per weight (near-lossless)
pub struct Q6KBlock {
    ql: [u8; 128],    // Low 4 bits
    qh: [u8; 64],     // High 2 bits
    scales: [i8; 16], // Sub-block scales
    d: f16,           // Super-block scale
}
```

## Estimated Impact

| Optimization     | Speed Improvement | Quality Impact | Complexity |
| ---------------- | ----------------- | -------------- | ---------- |
| Native Q4_K      | +20-40%           | None           | Medium     |
| Fused Dequant    | +10-20%           | None           | Medium     |
| Mixed Precision  | +10-15%           | Minimal        | Low        |
| Dynamic Quant    | +15-25%           | Small          | High       |
| LUT Optimization | +5-10%            | None           | Low        |
| Q5_K/Q6_K        | -10% (larger)     | Better         | Medium     |

## Recommended Experiment

1. **First**: Re-enable Q4_K with fixes - already implemented
2. **Second**: LUT optimization for NF4 - simple change
3. **Third**: Fused dequant-matmul for batch inference

## Files to Modify

-   `src/tensor/matrix.rs`: Q4K variant, mixed precision
-   `src/shaders/matmul_*_q4k.wgsl`: Optimized dequantization
-   `src/runtime/loader.rs`: Q4K loading path
-   `src/runtime/gguf.rs`: Q5_K/Q6_K parsing

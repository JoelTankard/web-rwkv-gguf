# Memory Layout Optimization

## Current Implementation

Tensor memory layout follows standard row-major ordering with shape `[dim0, dim1, dim2, dim3]`.

Key tensors during inference:

-   Input/Output: `[num_emb, num_token, 1, 1]`
-   Matrices: `[K, M, 1, 1]` where K=input_dim, M=output_dim
-   State: `[num_emb, head_size+2, num_batch, 1]`

## Problem

1. **Non-coalesced memory access**: Matrix multiplication may not access memory in optimal pattern
2. **Padding overhead**: Tensors padded to multiples of 8 for alignment
3. **Cache thrashing**: Large matrices don't fit in GPU cache
4. **Strided access**: Token dimension in middle causes strided access patterns

## Optimization Ideas

### Idea 1: Transposed Weight Storage

Store weights transposed for better access pattern during matmul:

```rust
// Current: weights stored as [K, M]
// Access pattern: for each output row, read entire input column
// Problem: strided access across K dimension

// Optimized: weights stored as [M, K] (transposed)
// Access pattern: for each output row, read contiguous K values
// Benefit: coalesced memory access
```

Implementation:

```rust
impl Matrix {
    pub fn transpose_for_inference(&self) -> Self {
        match self {
            Matrix::Fp16(w) => {
                let transposed = TensorOp::transpose(w)?;
                Matrix::Fp16Transposed(transposed)
            }
            // ... other variants
        }
    }
}
```

### Idea 2: Block-Interleaved Layout for Quantized Weights

For NF4/INT8, interleave blocks for better cache utilization:

```
Current layout (NF4):
[block0_row0][block1_row0][block2_row0]...
[block0_row1][block1_row1][block2_row1]...

Optimized layout (block-interleaved):
[block0_row0][block0_row1][block0_row2][block0_row3]  // 4 rows of same block
[block1_row0][block1_row1][block1_row2][block1_row3]
...
```

Benefits:

-   Better L2 cache utilization
-   Reduces cache line conflicts
-   Enables 4-row parallel processing

### Idea 3: Swizzled Tensor Layout

Use Morton/Z-order curve for 2D locality:

```rust
fn morton_index(x: u32, y: u32) -> u32 {
    // Interleave bits of x and y
    let mut result = 0u32;
    for i in 0..16 {
        result |= ((x >> i) & 1) << (2 * i);
        result |= ((y >> i) & 1) << (2 * i + 1);
    }
    result
}

// Store matrix in Morton order
fn store_swizzled(matrix: &[f16], rows: usize, cols: usize) -> Vec<f16> {
    let mut swizzled = vec![f16::ZERO; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let linear = r * cols + c;
            let morton = morton_index(r as u32, c as u32) as usize;
            swizzled[morton] = matrix[linear];
        }
    }
    swizzled
}
```

Benefits:

-   Better 2D cache locality for tiled matmul
-   Reduces cache misses for non-sequential access

### Idea 4: Packed Activation Storage

Pack multiple activations together to reduce memory traffic:

```rust
// Current: separate tensors for each intermediate
pub struct Runtime {
    pub att_rx: TensorGpu<F, ReadWrite>,
    pub att_wx: TensorGpu<F, ReadWrite>,
    pub att_kx: TensorGpu<F, ReadWrite>,
    pub att_vx: TensorGpu<F, ReadWrite>,
    pub att_ax: TensorGpu<F, ReadWrite>,
    pub att_gx: TensorGpu<F, ReadWrite>,
}

// Optimized: packed into single tensor
pub struct PackedRuntime {
    // [num_emb, num_token, 6, 1] - all 6 token-shifted values packed
    pub att_shifted: TensorGpu<F, ReadWrite>,
}
```

Benefits:

-   Single allocation instead of 6
-   Better memory locality
-   Enables fused operations

### Idea 5: Aligned Tensor Dimensions

Ensure tensor dimensions align with GPU cache lines and workgroup sizes:

```rust
fn align_dimension(dim: usize, alignment: usize) -> usize {
    (dim + alignment - 1) / alignment * alignment
}

impl TensorGpu {
    fn new_aligned(shape: Shape, alignment: usize) -> Self {
        let aligned_shape = Shape::new(
            align_dimension(shape[0], alignment),
            align_dimension(shape[1], alignment),
            shape[2],
            shape[3],
        );
        // Store original shape for bounds checking
        // Use aligned shape for allocation
    }
}
```

Recommended alignments:

-   FP16: 128 bytes (64 elements) for cache line
-   INT8/NF4: 256 bytes for quantization block alignment
-   Workgroup: 256 elements for typical workgroup size

### Idea 6: Separate Hot/Cold Data

Separate frequently accessed data from rarely accessed:

```rust
pub struct LayerWeights {
    // Hot path: accessed every token
    pub hot: HotWeights,
    // Cold path: accessed only during prefill or special operations
    pub cold: ColdWeights,
}

pub struct HotWeights {
    // Main projection matrices (always needed)
    pub w_k: Matrix,
    pub w_v: Matrix,
    pub w_r: Matrix,
    pub w_o: Matrix,
    pub ffn_k: Matrix,
    pub ffn_v: Matrix,
}

pub struct ColdWeights {
    // LoRA matrices (small, but separate allocation)
    pub w1: Matrix,
    pub w2: Matrix,
    // ... etc
}
```

Benefits:

-   Hot weights stay in GPU cache
-   Cold weights don't pollute cache during generation

## Estimated Impact

| Optimization        | Memory Bandwidth | Cache Hit Rate | Complexity |
| ------------------- | ---------------- | -------------- | ---------- |
| Transposed Weights  | +10-20%          | +5%            | Low        |
| Block-Interleaved   | +15-25%          | +15%           | Medium     |
| Swizzled Layout     | +10-15%          | +20%           | High       |
| Packed Activations  | +5-10%           | +10%           | Low        |
| Aligned Dimensions  | +5-10%           | +5%            | Low        |
| Hot/Cold Separation | +5%              | +15%           | Medium     |

## Recommended Experiment

1. **First**: Aligned dimensions - simple change, no algorithm modification
2. **Second**: Packed activations - enables fused operations
3. **Third**: Transposed weights for single-token inference

## Profiling

```bash
# Memory bandwidth utilization
WGPU_BACKEND=metal cargo run --release --example bench -- --model $MODEL --tokens 100

# Cache statistics (if available)
xcrun xctrace record --template "Metal System Trace" --launch -- cargo run --release
```

## Files to Modify

-   `src/tensor/shape.rs`: Alignment utilities
-   `src/tensor/mod.rs`: Packed tensor types
-   `src/runtime/loader.rs`: Transposed weight loading
-   `src/runtime/v7.rs`: Use packed activations

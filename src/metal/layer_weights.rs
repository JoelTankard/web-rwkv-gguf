//! Metal layer weights for pure Metal layer execution.
//!
//! This module holds all layer weights as Metal buffers for zero-copy access
//! during pure Metal layer execution.

use metal::Buffer;

use crate::metal::buffer_bridge::get_metal_buffer_from_wgpu;
use crate::runtime::v7::{Layer, LayerNorm};
use crate::tensor::matrix::Matrix;
use crate::tensor::TensorShape;

/// Metal buffers for a layer normalization.
#[derive(Debug)]
pub struct MetalLayerNorm {
    pub w: Buffer,
    pub b: Buffer,
}

impl MetalLayerNorm {
    pub fn from_wgpu(ln: &LayerNorm) -> Option<Self> {
        Some(Self {
            w: get_metal_buffer_from_wgpu(&ln.w.buffer)?,
            b: get_metal_buffer_from_wgpu(&ln.b.buffer)?,
        })
    }
}

/// Metal buffers for attention weights.
#[derive(Debug)]
pub struct MetalAttWeights {
    pub ln: MetalLayerNorm,

    // Token shift mix factors
    pub x_r: Buffer,
    pub x_w: Buffer,
    pub x_k: Buffer,
    pub x_v: Buffer,
    pub x_a: Buffer,
    pub x_g: Buffer,

    // Adapt biases
    pub w0: Buffer,
    pub a0: Buffer,
    pub v0: Buffer,

    // LoRA matrices (FP16)
    pub w1: Buffer,
    pub w2: Buffer,
    pub a1: Buffer,
    pub a2: Buffer,
    pub g1: Buffer,
    pub g2: Buffer,
    pub v1: Buffer,
    pub v2: Buffer,

    // Control vectors
    pub r_k: Buffer,
    pub k_k: Buffer,
    pub k_a: Buffer,

    // Main weight matrices (Q4K or FP16)
    pub w_k: MetalMatrixWeights,
    pub w_v: MetalMatrixWeights,
    pub w_r: MetalMatrixWeights,
    pub w_o: MetalMatrixWeights,

    // Group norm
    pub gn: MetalLayerNorm,
}

/// Metal buffers for FFN weights.
#[derive(Debug)]
pub struct MetalFfnWeights {
    pub ln: MetalLayerNorm,
    pub x_k: Buffer,
    pub w_k: MetalMatrixWeights,
    pub w_v: MetalMatrixWeights,
}

/// Metal representation of a weight matrix.
/// Supports both FP16 and Q4K quantized formats.
#[derive(Debug)]
pub enum MetalMatrixWeights {
    Fp16 {
        w: Buffer,
        k: u32, // input dimension
        m: u32, // output dimension
    },
    Q4K {
        w: Buffer, // raw Q4K blocks
        k: u32,    // input dimension (must be multiple of 256)
        m: u32,    // output dimension
    },
}

impl MetalMatrixWeights {
    pub fn from_matrix(matrix: &Matrix) -> Option<Self> {
        match matrix {
            Matrix::Fp16(tensor) => {
                let shape = tensor.shape();
                Some(MetalMatrixWeights::Fp16 {
                    w: get_metal_buffer_from_wgpu(&tensor.buffer)?,
                    k: shape[0] as u32,
                    m: shape[1] as u32,
                })
            }
            Matrix::Q4K { w, s } => {
                let shape = s.shape();
                Some(MetalMatrixWeights::Q4K {
                    w: get_metal_buffer_from_wgpu(&w.buffer)?,
                    k: shape[0] as u32,
                    m: shape[1] as u32,
                })
            }
            // For other quantization types, we'll fall back to wgpu
            _ => None,
        }
    }

    pub fn k(&self) -> u32 {
        match self {
            MetalMatrixWeights::Fp16 { k, .. } => *k,
            MetalMatrixWeights::Q4K { k, .. } => *k,
        }
    }

    pub fn m(&self) -> u32 {
        match self {
            MetalMatrixWeights::Fp16 { m, .. } => *m,
            MetalMatrixWeights::Q4K { m, .. } => *m,
        }
    }
}

/// All Metal weights for a single layer.
#[derive(Debug)]
pub struct MetalLayerWeights {
    pub att: MetalAttWeights,
    pub ffn: MetalFfnWeights,
}

impl MetalLayerWeights {
    /// Create Metal layer weights from wgpu layer.
    /// Returns None if any buffer extraction fails.
    pub fn from_layer(layer: &Layer) -> Option<Self> {
        let att = &layer.att;
        let ffn = &layer.ffn;

        // Extract attention weights
        let att_weights = MetalAttWeights {
            ln: MetalLayerNorm::from_wgpu(&layer.att_ln)?,
            x_r: get_metal_buffer_from_wgpu(&att.x_r.buffer)?,
            x_w: get_metal_buffer_from_wgpu(&att.x_w.buffer)?,
            x_k: get_metal_buffer_from_wgpu(&att.x_k.buffer)?,
            x_v: get_metal_buffer_from_wgpu(&att.x_v.buffer)?,
            x_a: get_metal_buffer_from_wgpu(&att.x_a.buffer)?,
            x_g: get_metal_buffer_from_wgpu(&att.x_g.buffer)?,
            w0: get_metal_buffer_from_wgpu(&att.w0.buffer)?,
            a0: get_metal_buffer_from_wgpu(&att.a0.buffer)?,
            v0: get_metal_buffer_from_wgpu(&att.v0.buffer)?,
            w1: extract_fp16_matrix(&att.w1)?,
            w2: extract_fp16_matrix(&att.w2)?,
            a1: extract_fp16_matrix(&att.a1)?,
            a2: extract_fp16_matrix(&att.a2)?,
            g1: extract_fp16_matrix(&att.g1)?,
            g2: extract_fp16_matrix(&att.g2)?,
            v1: extract_fp16_matrix(&att.v1)?,
            v2: extract_fp16_matrix(&att.v2)?,
            r_k: get_metal_buffer_from_wgpu(&att.r_k.buffer)?,
            k_k: get_metal_buffer_from_wgpu(&att.k_k.buffer)?,
            k_a: get_metal_buffer_from_wgpu(&att.k_a.buffer)?,
            w_k: MetalMatrixWeights::from_matrix(&att.w_k)?,
            w_v: MetalMatrixWeights::from_matrix(&att.w_v)?,
            w_r: MetalMatrixWeights::from_matrix(&att.w_r)?,
            w_o: MetalMatrixWeights::from_matrix(&att.w_o)?,
            gn: MetalLayerNorm::from_wgpu(&att.gn)?,
        };

        // Extract FFN weights
        let ffn_weights = MetalFfnWeights {
            ln: MetalLayerNorm::from_wgpu(&layer.ffn_ln)?,
            x_k: get_metal_buffer_from_wgpu(&ffn.x_k.buffer)?,
            w_k: MetalMatrixWeights::from_matrix(&ffn.w_k)?,
            w_v: MetalMatrixWeights::from_matrix(&ffn.w_v)?,
        };

        Some(MetalLayerWeights {
            att: att_weights,
            ffn: ffn_weights,
        })
    }
}

/// Helper to extract FP16 matrix buffer (for LoRA matrices which are always FP16).
fn extract_fp16_matrix(matrix: &Matrix) -> Option<Buffer> {
    match matrix {
        Matrix::Fp16(tensor) => get_metal_buffer_from_wgpu(&tensor.buffer),
        _ => None,
    }
}

/// All Metal weights for the entire model.
#[derive(Debug)]
pub struct MetalModelWeights {
    pub layers: Vec<MetalLayerWeights>,
    pub embed_ln: MetalLayerNorm,
    pub head_ln: MetalLayerNorm,
    pub head_w: MetalMatrixWeights,
}

impl MetalModelWeights {
    /// Create Metal model weights from wgpu model.
    /// Returns None if any buffer extraction fails.
    pub fn from_model(
        layers: &[Layer],
        embed_ln: &LayerNorm,
        head_ln: &LayerNorm,
        head_w: &Matrix,
    ) -> Option<Self> {
        let metal_layers: Option<Vec<_>> =
            layers.iter().map(MetalLayerWeights::from_layer).collect();

        Some(MetalModelWeights {
            layers: metal_layers?,
            embed_ln: MetalLayerNorm::from_wgpu(embed_ln)?,
            head_ln: MetalLayerNorm::from_wgpu(head_ln)?,
            head_w: MetalMatrixWeights::from_matrix(head_w)?,
        })
    }
}

use std::borrow::Cow;

use half::f16;
use itertools::Itertools;
use regex::Regex;
use safetensors::{Dtype, SafeTensorError, SafeTensors};
use thiserror::Error;
use web_rwkv_derive::{Deref, DerefMut};

use super::model::{ModelCustomInfo, ModelInfo, ModelVersion, Quant};
use crate::{
    context::Context,
    num::Scalar,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorOp},
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorErrorKind, TensorGpu, TensorInit, TensorInto, TensorReshape,
        TensorShape,
    },
};

pub const PAD_VEC: [usize; 4] = [8, 1, 1, 1];
pub const PAD_MAT: [usize; 4] = [8, 8, 1, 1];

const Q4K_BLOCK_BYTES: usize = 144;

/// Transpose Q4K layout from row-major to column-major for coalesced memory access.
/// Original layout: block[row][sb_k] - threads reading different rows access far-apart memory
/// Transposed layout: block[sb_k][row] - consecutive threads read consecutive super-blocks
///
/// This improves memory bandwidth utilization during matmul by ensuring that
/// threads in a workgroup access consecutive memory addresses.
fn transpose_q4k_layout(raw_data: &[u8], num_rows: usize, num_sb_k: usize) -> Vec<u8> {
    let mut transposed = vec![0u8; raw_data.len()];
    for row in 0..num_rows {
        for sb_k in 0..num_sb_k {
            let src_offset = (row * num_sb_k + sb_k) * Q4K_BLOCK_BYTES;
            let dst_offset = (sb_k * num_rows + row) * Q4K_BLOCK_BYTES;
            transposed[dst_offset..dst_offset + Q4K_BLOCK_BYTES]
                .copy_from_slice(&raw_data[src_offset..src_offset + Q4K_BLOCK_BYTES]);
        }
    }
    transposed
}

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("invalid model version")]
    InvalidVersion,
    #[error("tensor error")]
    TensorError(#[from] TensorError),
    #[error("failed to load safe tensor")]
    SafeTensor(#[from] safetensors::SafeTensorError),
    #[error("failed to parse int")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("failed to parse regex")]
    RegexError(#[from] regex::Error),
}

pub type ReaderTensor<'a> = (Dtype, Vec<usize>, Cow<'a, [u8]>);

/// Interface accessing a safetensors data blob.
pub trait Reader {
    fn names(&self) -> Vec<&str>;
    fn contains(&self, name: &str) -> bool;
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError>;
    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError>;

    /// Get raw quantized tensor data for direct loading (bypasses dequantization).
    /// Returns (ggml_type_id, raw_data) if the tensor is quantized, None otherwise.
    fn quantized_tensor(&self, _name: &str) -> Option<(u32, &[u8])> {
        None
    }
}

impl Reader for SafeTensors<'_> {
    #[inline]
    fn names(&self) -> Vec<&str> {
        self.names().into_iter().map(AsRef::as_ref).collect()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.names().contains(&name)
    }

    #[inline]
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        Ok(self.tensor(name)?.shape().to_vec())
    }

    #[inline]
    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        let tensor = SafeTensors::tensor(self, name)?;
        let shape = tensor.shape().to_vec();
        let data = tensor.data().into();
        Ok((tensor.dtype(), shape, data))
    }
}

pub trait TensorFromReader<T: Scalar> {
    /// Create a tensor from safetensors reader.
    fn from_reader(reader: ReaderTensor) -> Result<TensorCpu<T>, TensorError>;
}

impl<T: Scalar> TensorFromReader<T> for TensorCpu<T> {
    fn from_reader((dt, shape, data): ReaderTensor) -> Result<Self, TensorError> {
        if T::DATA_TYPE != dt {
            Err(TensorErrorKind::Type)?;
        }
        let shape = Shape::from_slice_rev(&shape)?;
        match data {
            Cow::Borrowed(data) => Self::from_data(shape, bytemuck::cast_slice(data)),
            Cow::Owned(data) => {
                let data = bytemuck::cast_slice(&data);
                let data = Cow::Owned(data.to_vec());
                Self::from_data(shape, data)
            }
        }
    }
}

pub fn tensor_f16_from_reader(reader: ReaderTensor) -> Result<TensorCpu<f16>, TensorError> {
    let (dt, shape, data) = reader;
    let shape = Shape::from_slice_rev(&shape)?;

    match dt {
        Dtype::F16 => match data {
            Cow::Borrowed(data) => TensorCpu::from_data(shape, bytemuck::cast_slice(data)),
            Cow::Owned(data) => {
                let data = bytemuck::cast_slice(&data);
                let data = Cow::Owned(data.to_vec());
                TensorCpu::from_data(shape, data)
            }
        },
        Dtype::F32 => {
            let f32_data: &[f32] = bytemuck::cast_slice(&data);
            let f16_data: Vec<f16> = f32_data.iter().map(|&x| f16::from_f32(x)).collect();
            TensorCpu::from_data(shape, f16_data)
        }
        Dtype::BF16 => {
            let bf16_data: &[half::bf16] = bytemuck::cast_slice(&data);
            let f16_data: Vec<f16> = bf16_data
                .iter()
                .map(|x| f16::from_f32(x.to_f32()))
                .collect();
            TensorCpu::from_data(shape, f16_data)
        }
        _ => Err(TensorErrorKind::Type)?,
    }
}

/// A LoRA that adds to the model when loading.
#[derive(Clone)]
pub struct Lora<R> {
    /// Binary safetensors LoRA content.
    pub data: R,
    /// A list of LoRA blend patterns.
    /// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
    /// When applying the patterns, they are applied in order.
    pub blend: LoraBlend,
}

/// A list of LoRA blend patterns.
#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    /// Build a blend pattern that replaces all vectors, and adds to all matrices with `alpha`.
    #[inline]
    pub fn full(alpha: f32) -> Self {
        Self::default().add_nominal(1.0).add_matrices(alpha)
    }

    /// Add a blend pattern that interpolates tensors with factor `alpha` from 0 to 1.
    #[inline]
    pub fn add_nominal(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r".+", alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices with `alpha`.
    #[inline]
    pub fn add_matrices(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(
            r"blocks\.([0-9]+)\.(att|ffn)\.(key|value|receptance|gate|output)\.weight",
            alpha,
        )
        .unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that interpolates tensors in a layer with factor `alpha` from 0 to 1.
    pub fn add_layer_nominal(mut self, layer: usize, alpha: f32) -> Self {
        let pattern = format!(r"blocks\.{layer}");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices in a layer with `alpha`.
    pub fn add_layer_matrices(mut self, layer: usize, alpha: f32) -> Self {
        let pattern =
            format!(r"blocks\.{layer}\.(att|ffn)\.(key|value|receptance|gate|output)\.weight");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }
}

/// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
#[derive(Debug, Clone)]
pub struct LoraBlendPattern {
    /// A regex pattern that matches tensors in the model.
    pattern: Regex,
    /// The blend factor.
    alpha: f32,
}

impl LoraBlendPattern {
    #[inline]
    pub fn new(pattern: &str, alpha: f32) -> Result<Self, LoaderError> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            alpha,
        })
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

struct LoraVector {
    tensor: TensorGpu<f16, ReadWrite>,
    alpha: f32,
}

struct LoraMatrix {
    x: TensorGpu<f16, ReadWrite>,
    y: TensorGpu<f16, ReadWrite>,
    rank: usize,
    alpha: f32,
}

#[derive(Clone)]
pub struct Loader<R> {
    pub context: Context,
    pub model: R,
    pub lora: Vec<Lora<R>>,
}

impl<R: Reader> Loader<R> {
    pub fn info(model: &R) -> Result<ModelInfo, LoaderError> {
        let num_layer = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "blocks.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };

        let embed = model.shape("emb.weight")?;
        let ffn = model.shape("blocks.0.ffn.key.weight")?;

        let v4 = [
            "blocks.0.att.time_decay",
            "blocks.0.att.time_first",
            "blocks.0.att.time_mix_k",
            "blocks.0.att.time_mix_v",
            "blocks.0.att.time_mix_r",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v5 = [
            "blocks.0.att.gate.weight",
            "blocks.0.att.ln_x.weight",
            "blocks.0.att.ln_x.bias",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v6 = [
            "blocks.0.att.time_mix_x",
            "blocks.0.att.time_mix_w",
            "blocks.0.att.time_mix_k",
            "blocks.0.att.time_mix_v",
            "blocks.0.att.time_mix_r",
            "blocks.0.att.time_mix_g",
            "blocks.0.att.time_mix_w1",
            "blocks.0.att.time_mix_w2",
            "blocks.0.att.time_decay_w1",
            "blocks.0.att.time_decay_w2",
            "blocks.0.ffn.time_mix_k",
            "blocks.0.ffn.time_mix_r",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v7_separate = [
            "blocks.0.att.x_r",
            "blocks.0.att.x_w",
            "blocks.0.att.x_k",
            "blocks.0.att.x_v",
            "blocks.0.att.x_a",
            "blocks.0.att.x_g",
            "blocks.0.att.w0",
            "blocks.0.att.w1",
            "blocks.0.att.w2",
            "blocks.0.att.a0",
            "blocks.0.att.a1",
            "blocks.0.att.a2",
            "blocks.0.att.g1",
            "blocks.0.att.g2",
            "blocks.0.att.r_k",
            "blocks.0.att.k_k",
            "blocks.0.att.k_a",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v7_fused = [
            "blocks.0.att.time_maa",
            "blocks.0.att.w0",
            "blocks.0.att.w1",
            "blocks.0.att.w2",
            "blocks.0.att.a0",
            "blocks.0.att.a1",
            "blocks.0.att.a2",
            "blocks.0.att.g1",
            "blocks.0.att.g2",
            "blocks.0.att.r_k",
            "blocks.0.att.k_k",
            "blocks.0.att.k_a",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v7 = v7_separate || v7_fused;

        let version = match (v4, v5, v6, v7) {
            (true, false, false, false) => ModelVersion::V4,
            (_, true, false, false) => ModelVersion::V5,
            (_, _, true, false) => ModelVersion::V6,
            (_, _, _, true) => ModelVersion::V7,
            _ => return Err(LoaderError::InvalidVersion),
        };

        let num_emb = embed[1];
        let num_hidden = ffn[0];
        let num_vocab = embed[0];

        let num_head = match version {
            ModelVersion::V4 => 1,
            ModelVersion::V5 | ModelVersion::V6 => model.shape("blocks.0.att.time_first")?[0],
            ModelVersion::V7 => model.shape("blocks.0.att.r_k")?[0],
        };

        let custom = match version {
            ModelVersion::V6 => {
                let time_mix = model.shape("blocks.0.att.time_mix_w1")?[0] / 5;
                let time_decay = model.shape("blocks.0.att.time_decay_w1")?[0];
                ModelCustomInfo::V6(super::v6::CustomInfo {
                    time_mix,
                    time_decay,
                })
            }
            ModelVersion::V7 => {
                let w = model.shape("blocks.0.att.w1")?[0];
                let a = model.shape("blocks.0.att.a1")?[0];
                let g = model.shape("blocks.0.att.g1")?[0];
                let v = model.shape("blocks.1.att.v1")?[0];
                ModelCustomInfo::V7(super::v7::CustomInfo { w, a, g, v })
            }
            _ => ModelCustomInfo::None,
        };

        Ok(ModelInfo {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
            custom,
        })
    }

    /// Load all lora and blend factors about the vector with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    fn lora_vectors(&self, name: impl AsRef<str>) -> Result<Vec<LoraVector>, LoaderError> {
        let context = &self.context;
        let name = name.as_ref();

        let mut vectors = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .filter(|blend| blend.pattern.is_match(name))
                .next_back()
            else {
                continue;
            };

            let Ok(tensor) = lora.data.tensor(name) else {
                continue;
            };
            let tensor = TensorCpu::from_reader(tensor)?.to(context);
            let alpha = blend.alpha;
            vectors.push(LoraVector { tensor, alpha });

            log::info!("vector (LoRA) {name}, alpha: {alpha}");
        }
        Ok(vectors)
    }

    /// Load all lora and blend factors about the matrix with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    fn lora_matrices(&self, name: impl AsRef<str>) -> Result<Vec<LoraMatrix>, LoaderError> {
        let context = &self.context;
        let name = name.as_ref();

        let mut matrices = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .filter(|blend| blend.pattern.is_match(name))
                .next_back()
            else {
                continue;
            };

            let name = name.split('.').filter(|x| !x.contains("weight")).join(".");
            let Ok(x) = lora.data.tensor(&format!("{name}.lora.0")) else {
                continue;
            };
            let Ok(y) = lora.data.tensor(&format!("{name}.lora.1")) else {
                continue;
            };

            let rank = x.1[1];
            let alpha = blend.alpha;
            let x = TensorCpu::from_reader(x)?.to(context);
            let y = TensorCpu::from_reader(y)?.to(context);
            matrices.push(LoraMatrix { x, y, rank, alpha });

            log::info!("matrix (LoRA) {name}, alpha: {alpha}, rank: {rank}");
        }
        Ok(matrices)
    }

    pub fn tensor_shape(&self, name: impl AsRef<str>) -> Result<Shape, LoaderError> {
        let shape = self.model.shape(name.as_ref())?;
        Ok(Shape::from_slice_rev(&shape)?)
    }

    pub fn load_vector_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = tensor_f16_from_reader(tensor)?
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = tensor_f16_from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        let op = TensorOp::activate(&tensor, Activation::OppositeExp)?;
        ops.push(op);

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_exp_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = tensor_f16_from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            // .map(|x| x.exp())
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        let op = TensorOp::activate(&tensor, Activation::StableExp)?;
        ops.push(op);

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let lora = self.lora_vectors(name.as_ref())?;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = if lora.is_empty() {
            tensor_f16_from_reader(tensor)?
                .reshape(
                    TensorDimension::Auto,
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                )?
                .to(context)
        } else {
            let tensor_f32: TensorGpu<f32, _> = tensor_f16_from_reader(tensor)?
                .map(|x| x.to_f32())
                .reshape(
                    TensorDimension::Auto,
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                )?
                .to(context);
            let tensor_f16: TensorGpu<f16, _> = context.tensor_init(tensor_f32.shape());

            let mut ops = vec![];
            for lora in lora {
                let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
                let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

                let shape = lora.tensor.shape();
                let tensor = tensor_f32.reshape(
                    TensorDimension::Size(shape[0]),
                    TensorDimension::Size(shape[1]),
                    TensorDimension::Size(shape[2]),
                    TensorDimension::Size(shape[3]),
                )?;

                let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
                ops.push(op);
            }

            let op = TensorOp::blit(&tensor_f32, &tensor_f16)?;
            ops.push(op);

            context.queue.submit(context.encode(&TensorOp::List(ops)));
            tensor_f16
        };
        Ok(tensor)
    }

    pub fn load_matrix_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = tensor_f16_from_reader(tensor)?.to(context);

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, &tensor)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_matrix_f16_discount(
        &self,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = tensor_f16_from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, &tensor)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_in_place_matrix_f16(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
    ) -> Result<(), LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = tensor_f16_from_reader(tensor)?;
        matrix.load(&tensor)?;

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, matrix)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(())
    }

    pub fn load_in_place_matrix_f16_discount(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<(), LoaderError> {
        let context = &self.context;

        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = tensor_f16_from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .reshape(
                TensorDimension::Full,
                TensorDimension::Full,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?;
        matrix.load(&tensor)?;

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, matrix)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(())
    }

    pub fn load_matrix_f16_padded_cpu(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorCpu<f16>, LoaderError> {
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = tensor_f16_from_reader(tensor)?.pad(PAD_MAT);
        Ok(tensor)
    }

    pub fn load_matrix_f16_padded(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = tensor_f16_from_reader(tensor)?.pad(PAD_MAT).to(context);
        Ok(tensor)
    }

    pub fn load_matrix(&self, name: String, quant: Quant) -> Result<Matrix, LoaderError> {
        let context = &self.context;

        // Try direct quantized loading for GGUF tensors (bypasses F16 staging)
        if let Some((gguf_type, raw_data)) = self.model.quantized_tensor(&name) {
            if let Some(matrix) =
                self.try_load_matrix_direct(context, &name, gguf_type, raw_data, quant)?
            {
                return Ok(matrix);
            }
        }

        match quant {
            Quant::None => Ok(Matrix::Fp16(self.load_matrix_f16(name)?)),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
            Quant::SF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_sf4(&buffer, 5.0)?)
            }
        }
    }

    /// Try to load a matrix directly from GGUF quantized data without F16 staging.
    /// Returns None if direct loading is not supported for this combination.
    fn try_load_matrix_direct(
        &self,
        context: &Context,
        name: &str,
        gguf_type: u32,
        raw_data: &[u8],
        quant: Quant,
    ) -> Result<Option<Matrix>, LoaderError> {
        use super::gguf::{repack_q4_0_to_nf4, repack_q8_0_to_int8, GgmlType};
        use crate::tensor::matrix::Float4Quant;

        let shape = self.tensor_shape(name)?;
        let num_elements = shape.len();

        match (GgmlType::from(gguf_type), quant) {
            (GgmlType::Q8_0, Quant::Int8) => {
                // Direct Q8_0 -> Int8 repacking
                let (weights, minmax) = repack_q8_0_to_int8(raw_data, num_elements);

                // Create GPU tensors
                let w: TensorGpu<u8, ReadWrite> = TensorCpu::from_data(shape, weights)?.to(context);
                let m_shape = Shape::new(minmax.len(), 1, 1, 1);
                let m: TensorGpu<f16, ReadWrite> =
                    TensorCpu::from_data(m_shape, minmax)?.to(context);

                log::info!("direct load (Q8_0 -> Int8): {name}");
                Ok(Some(Matrix::Int8 { w, m }))
            }
            (GgmlType::Q4K, Quant::Int8) | (GgmlType::Q4K, Quant::None) => {
                // Native Q4_K loading - store raw blocks for inline dequantization during matmul
                // This is faster than repacking to Int8 because it avoids dequant/requant overhead
                let actual_elements = (raw_data.len() / 144) * 256;
                if actual_elements != num_elements || num_elements % 256 != 0 {
                    return Ok(None); // Fall back to F16 path
                }

                // Transpose Q4K layout for coalesced memory access
                // Original: block[row][sb_k] - row-major, threads reading different rows access far-apart memory
                // Transposed: block[sb_k][row] - consecutive threads read consecutive super-blocks
                let k = shape[0]; // K dimension (input features)
                let m = shape[1]; // M dimension (output features/rows)
                let num_sb_k = k / 256; // Number of super-blocks along K
                let transposed = transpose_q4k_layout(raw_data, m, num_sb_k);

                // Create GPU tensor with transposed Q4_K block data
                let block_data_shape = Shape::new(transposed.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorGpu::from_data_u8(context, block_data_shape, &transposed)?;

                // Create Metal buffer for Q4K weights (if Metal backend is enabled)
                #[cfg(all(target_os = "macos", feature = "metal-backend"))]
                if let Some(metal_ctx) = context.metal() {
                    use crate::metal::BufferId;
                    let buffer_id = BufferId::from_tensor_id(w.id());
                    metal_ctx.get_or_create_buffer(buffer_id, &transposed);
                    log::debug!("Created Metal buffer for Q4K tensor: {name}");
                }

                // Create a dummy tensor with the logical matrix shape for metadata
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);

                log::info!(
                    "native Q4_K load: {name} ({} elements, {} bytes)",
                    num_elements,
                    raw_data.len()
                );
                Ok(Some(Matrix::Q4K { w, s }))
            }
            (GgmlType::Q5K, Quant::Int8) | (GgmlType::Q5K, Quant::None) => {
                // Native Q5_K loading - store raw blocks for inline dequantization during matmul
                let actual_elements = (raw_data.len() / 176) * 256;
                if actual_elements != num_elements || num_elements % 256 != 0 {
                    return Ok(None); // Fall back to F16 path
                }

                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);

                log::info!(
                    "native Q5_K load: {name} ({} elements, {} bytes)",
                    num_elements,
                    raw_data.len()
                );
                Ok(Some(Matrix::Q5K { w, s }))
            }
            (GgmlType::Q6K, Quant::Int8) | (GgmlType::Q6K, Quant::None) => {
                // Native Q6_K loading - store raw blocks for inline dequantization during matmul
                let actual_elements = (raw_data.len() / 210) * 256;
                if actual_elements != num_elements || num_elements % 256 != 0 {
                    return Ok(None); // Fall back to F16 path
                }

                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);

                log::info!(
                    "native Q6_K load: {name} ({} elements, {} bytes)",
                    num_elements,
                    raw_data.len()
                );
                Ok(Some(Matrix::Q6K { w, s }))
            }
            (GgmlType::Q8_0, Quant::None) => {
                // Native Q8_0 loading - store raw blocks for inline dequantization during matmul
                let actual_elements = (raw_data.len() / 34) * 32;
                if actual_elements != num_elements || num_elements % 32 != 0 {
                    return Ok(None); // Fall back to F16 path
                }

                let block_data_shape = Shape::new(raw_data.len(), 1, 1, 1);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorGpu::from_data_u8(context, block_data_shape, raw_data)?;
                let s: TensorGpu<u8, ReadWrite> = context.tensor_init(shape);

                log::info!(
                    "native Q8_0 load: {name} ({} elements, {} bytes)",
                    num_elements,
                    raw_data.len()
                );
                Ok(Some(Matrix::Q8_0 { w, s }))
            }
            (GgmlType::Q4_0, Quant::NF4) => {
                // Direct Q4_0 -> NF4 repacking
                let (weights, absmax) = repack_q4_0_to_nf4(raw_data, num_elements);

                // Create GPU tensors
                let matrix_shape = Shape::new(shape[0] / 2, shape[1], shape[2], shape[3]);
                let w: TensorGpu<u8, ReadWrite> =
                    TensorCpu::from_data(matrix_shape, weights)?.to(context);

                let m_shape = Shape::new(absmax.len(), 1, 1, 1);
                let m: TensorGpu<f16, ReadWrite> =
                    TensorCpu::from_data(m_shape, absmax)?.to(context);

                let q = Float4Quant::default().0.to(context);

                log::info!("direct load (Q4_0 -> NF4): {name}");
                Ok(Some(Matrix::Fp4 { w, q, m }))
            }
            _ => Ok(None), // Fall through to F16 path
        }
    }

    pub fn load_matrix_discount(
        &self,
        name: String,
        quant: Quant,
        discount: f32,
    ) -> Result<Matrix, LoaderError> {
        let context = &self.context;
        match quant {
            Quant::None => Ok(Matrix::Fp16(self.load_matrix_f16_discount(name, discount)?)),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
            Quant::SF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_sf4(&buffer, 5.0)?)
            }
        }
    }
}

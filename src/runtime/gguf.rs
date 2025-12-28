use std::{borrow::Cow, collections::HashMap};

use safetensors::{Dtype, SafeTensorError};
use thiserror::Error;

use super::loader::ReaderTensor;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("invalid magic number: expected 0x{:08X}, got 0x{:08X}", GGUF_MAGIC, .0)]
    InvalidMagic(u32),
    #[error("unsupported version: {0} (supported: {GGUF_VERSION})")]
    UnsupportedVersion(u32),
    #[error("unexpected end of file")]
    UnexpectedEof,
    #[error("invalid utf-8 string")]
    InvalidUtf8,
    #[error("invalid metadata value type: {0}")]
    InvalidValueType(u32),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("unsupported tensor type: {0:?}")]
    UnsupportedTensorType(GgmlType),
}

impl From<GgufError> for SafeTensorError {
    fn from(err: GgufError) -> Self {
        SafeTensorError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            err.to_string(),
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    Unknown(u32),
}

impl From<u32> for GgmlType {
    fn from(value: u32) -> Self {
        match value {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::IQ2XXS,
            17 => GgmlType::IQ2XS,
            18 => GgmlType::IQ3XXS,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4NL,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4XS,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            29 => GgmlType::IQ1M,
            30 => GgmlType::BF16,
            34 => GgmlType::TQ1_0,
            35 => GgmlType::TQ2_0,
            v => GgmlType::Unknown(v),
        }
    }
}

impl GgmlType {
    pub fn to_dtype(&self) -> Option<Dtype> {
        match self {
            GgmlType::F32 => Some(Dtype::F32),
            GgmlType::F16 => Some(Dtype::F16),
            GgmlType::BF16 => Some(Dtype::BF16),
            GgmlType::I8 => Some(Dtype::I8),
            GgmlType::I16 => Some(Dtype::I16),
            GgmlType::I32 => Some(Dtype::I32),
            GgmlType::I64 => Some(Dtype::I64),
            GgmlType::F64 => Some(Dtype::F64),
            _ => None, // Quantized types don't map directly to Dtype
        }
    }

    pub fn type_size(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::BF16 => 2,
            GgmlType::F64 => 8,
            GgmlType::I8 => 1,
            GgmlType::I16 => 2,
            GgmlType::I32 => 4,
            GgmlType::I64 => 8,
            GgmlType::Q4_0 => 18, // block size 32, 16 bytes data + 2 bytes scale
            GgmlType::Q4_1 => 20, // block size 32, 16 bytes data + 2 bytes scale + 2 bytes min
            GgmlType::Q5_0 => 22, // block size 32
            GgmlType::Q5_1 => 24, // block size 32
            GgmlType::Q8_0 => 34, // block size 32, 32 bytes data + 2 bytes scale
            GgmlType::Q8_1 => 36, // block size 32
            GgmlType::Q2K => 84,  // block size 256
            GgmlType::Q3K => 110, // block size 256
            GgmlType::Q4K => 144, // block size 256
            GgmlType::Q5K => 176, // block size 256
            GgmlType::Q6K => 210, // block size 256
            GgmlType::Q8K => 292, // block size 256
            _ => 0,               // Unknown or unsupported
        }
    }

    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64 => 1,
            GgmlType::I8 | GgmlType::I16 | GgmlType::I32 | GgmlType::I64 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => 32,
            GgmlType::Q8_0 | GgmlType::Q8_1 => 32,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Q8K => 256,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Uint8(v) => Some(*v as u32),
            MetadataValue::Uint16(v) => Some(*v as u32),
            MetadataValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            MetadataValue::Uint32(v) => Some(*v as u64),
            MetadataValue::Uint8(v) => Some(*v as u64),
            MetadataValue::Uint16(v) => Some(*v as u64),
            MetadataValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    pub fn data_size(&self) -> usize {
        let elements = self.num_elements() as usize;
        let block_size = self.tensor_type.block_size();
        let type_size = self.tensor_type.type_size();

        if block_size == 1 {
            elements * type_size
        } else {
            (elements / block_size) * type_size
        }
    }
}

pub struct GgufReader<'a> {
    data: &'a [u8],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub tensor_data_offset: u64,
    name_map: HashMap<String, String>,
}

fn build_rwkv_name_map(tensors: &HashMap<String, TensorInfo>) -> HashMap<String, String> {
    let mut map = HashMap::new();

    for gguf_name in tensors.keys() {
        if let Some(safetensors_name) = gguf_to_safetensors_name(gguf_name) {
            map.insert(safetensors_name, gguf_name.clone());
        }
        map.insert(gguf_name.clone(), gguf_name.clone());
    }

    map
}

fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    if gguf_name == "token_embd.weight" {
        return Some("emb.weight".to_string());
    }
    if gguf_name == "output_norm.weight" {
        return Some("ln_out.weight".to_string());
    }
    if gguf_name == "output_norm.bias" {
        return Some("ln_out.bias".to_string());
    }
    if gguf_name == "output.weight" {
        return Some("head.weight".to_string());
    }
    if gguf_name == "token_embd_norm.weight" {
        return Some("blocks.0.ln0.weight".to_string());
    }
    if gguf_name == "token_embd_norm.bias" {
        return Some("blocks.0.ln0.bias".to_string());
    }

    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let block_num = &rest[..dot_pos];
            let remainder = &rest[dot_pos + 1..];

            let mapped = match remainder {
                "attn_norm.weight" => format!("blocks.{block_num}.ln1.weight"),
                "attn_norm.bias" => format!("blocks.{block_num}.ln1.bias"),
                "attn_norm_2.weight" => format!("blocks.{block_num}.ln2.weight"),
                "attn_norm_2.bias" => format!("blocks.{block_num}.ln2.bias"),
                "ffn_norm.weight" => format!("blocks.{block_num}.ln2.weight"),
                "ffn_norm.bias" => format!("blocks.{block_num}.ln2.bias"),

                "attn_k.weight" => format!("blocks.{block_num}.att.key.weight"),
                "attn_v.weight" => format!("blocks.{block_num}.att.value.weight"),
                "attn_r.weight" => format!("blocks.{block_num}.att.receptance.weight"),
                "attn_g.weight" => format!("blocks.{block_num}.att.gate.weight"),
                "attn_output.weight" => format!("blocks.{block_num}.att.output.weight"),

                "attn_time_decay" => format!("blocks.{block_num}.att.time_decay"),
                "attn_time_first" => format!("blocks.{block_num}.att.time_first"),
                "attn_time_mix_k" => format!("blocks.{block_num}.att.time_mix_k"),
                "attn_time_mix_v" => format!("blocks.{block_num}.att.time_mix_v"),
                "attn_time_mix_r" => format!("blocks.{block_num}.att.time_mix_r"),
                "attn_time_mix_g" => format!("blocks.{block_num}.att.time_mix_g"),
                "attn_time_mix_x" => format!("blocks.{block_num}.att.time_mix_x"),
                "attn_time_mix_w" => format!("blocks.{block_num}.att.time_mix_w"),

                // V6 specific tensors
                "attn_time_mix_w1" => format!("blocks.{block_num}.att.time_mix_w1"),
                "attn_time_mix_w2" => format!("blocks.{block_num}.att.time_mix_w2"),
                "attn_time_decay_w1" => format!("blocks.{block_num}.att.time_decay_w1"),
                "attn_time_decay_w2" => format!("blocks.{block_num}.att.time_decay_w2"),
                "time_maa_w1" => format!("blocks.{block_num}.att.time_mix_w1"),
                "time_maa_w2" => format!("blocks.{block_num}.att.time_mix_w2"),
                "time_decay_w1" => format!("blocks.{block_num}.att.time_decay_w1"),
                "time_decay_w2" => format!("blocks.{block_num}.att.time_decay_w2"),

                "attn_ln_x.weight" => format!("blocks.{block_num}.att.ln_x.weight"),
                "attn_ln_x.bias" => format!("blocks.{block_num}.att.ln_x.bias"),

                "attn_time_state" => format!("blocks.{block_num}.att.time_state"),

                "ffn_k.weight" => format!("blocks.{block_num}.ffn.key.weight"),
                "ffn_v.weight" => format!("blocks.{block_num}.ffn.value.weight"),
                "ffn_r.weight" => format!("blocks.{block_num}.ffn.receptance.weight"),
                "ffn_time_mix_k" => format!("blocks.{block_num}.ffn.time_mix_k"),
                "ffn_time_mix_r" => format!("blocks.{block_num}.ffn.time_mix_r"),

                // Additional FFN mappings for RWKV v7
                "ffn.key.weight" => format!("blocks.{block_num}.ffn.key.weight"),
                "ffn.value.weight" => format!("blocks.{block_num}.ffn.value.weight"),
                "ffn.receptance.weight" => format!("blocks.{block_num}.ffn.receptance.weight"),
                "channel_mix_key.weight" => format!("blocks.{block_num}.ffn.key.weight"),
                "channel_mix_value.weight" => format!("blocks.{block_num}.ffn.value.weight"),
                "channel_mix_receptance.weight" => {
                    format!("blocks.{block_num}.ffn.receptance.weight")
                }
                "channel_mix_lerp_k.weight" => format!("blocks.{block_num}.ffn.x_k"),

                // V7 attention tensors (time_mix_ prefix from actual GGUF files)
                "time_mix_key.weight" => format!("blocks.{block_num}.att.key.weight"),
                "time_mix_value.weight" => format!("blocks.{block_num}.att.value.weight"),
                "time_mix_receptance.weight" => format!("blocks.{block_num}.att.receptance.weight"),
                "time_mix_gate.weight" => format!("blocks.{block_num}.att.gate.weight"),
                "time_mix_output.weight" => format!("blocks.{block_num}.att.output.weight"),
                "time_mix_lerp_fused.weight" => format!("blocks.{block_num}.att.time_maa"),
                "time_mix_w0.weight" => format!("blocks.{block_num}.att.w0"),
                "time_mix_w1.weight" => format!("blocks.{block_num}.att.w1"),
                "time_mix_w2.weight" => format!("blocks.{block_num}.att.w2"),
                "time_mix_a0.weight" => format!("blocks.{block_num}.att.a0"),
                "time_mix_a1.weight" => format!("blocks.{block_num}.att.a1"),
                "time_mix_a2.weight" => format!("blocks.{block_num}.att.a2"),
                "time_mix_g1.weight" => format!("blocks.{block_num}.att.g1"),
                "time_mix_g2.weight" => format!("blocks.{block_num}.att.g2"),
                "time_mix_v0.weight" => format!("blocks.{block_num}.att.v0"),
                "time_mix_v1.weight" => format!("blocks.{block_num}.att.v1"),
                "time_mix_v2.weight" => format!("blocks.{block_num}.att.v2"),
                "time_mix_r_k.weight" => format!("blocks.{block_num}.att.r_k"),
                "time_mix_k_k.weight" => format!("blocks.{block_num}.att.k_k"),
                "time_mix_k_a.weight" => format!("blocks.{block_num}.att.k_a"),
                "time_mix_ln.weight" => format!("blocks.{block_num}.att.ln_x.weight"),
                "time_mix_ln.bias" => format!("blocks.{block_num}.att.ln_x.bias"),

                "attn_x_r" => format!("blocks.{block_num}.att.x_r"),
                "attn_x_w" => format!("blocks.{block_num}.att.x_w"),
                "attn_x_k" => format!("blocks.{block_num}.att.x_k"),
                "attn_x_v" => format!("blocks.{block_num}.att.x_v"),
                "attn_x_a" => format!("blocks.{block_num}.att.x_a"),
                "attn_x_g" => format!("blocks.{block_num}.att.x_g"),
                "attn_w0" => format!("blocks.{block_num}.att.w0"),
                "attn_w1" => format!("blocks.{block_num}.att.w1"),
                "attn_w2" => format!("blocks.{block_num}.att.w2"),
                "attn_a0" => format!("blocks.{block_num}.att.a0"),
                "attn_a1" => format!("blocks.{block_num}.att.a1"),
                "attn_a2" => format!("blocks.{block_num}.att.a2"),
                "attn_g1" => format!("blocks.{block_num}.att.g1"),
                "attn_g2" => format!("blocks.{block_num}.att.g2"),
                "attn_v0" => format!("blocks.{block_num}.att.v0"),
                "attn_v1" => format!("blocks.{block_num}.att.v1"),
                "attn_v2" => format!("blocks.{block_num}.att.v2"),
                "attn_r_k" => format!("blocks.{block_num}.att.r_k"),
                "attn_k_k" => format!("blocks.{block_num}.att.k_k"),
                "attn_k_a" => format!("blocks.{block_num}.att.k_a"),

                "ffn_x_k" => format!("blocks.{block_num}.ffn.x_k"),

                // V7 attention tensors
                "att_x_r" => format!("blocks.{block_num}.att.x_r"),
                "att_x_w" => format!("blocks.{block_num}.att.x_w"),
                "att_x_k" => format!("blocks.{block_num}.att.x_k"),
                "att_x_v" => format!("blocks.{block_num}.att.x_v"),
                "att_x_a" => format!("blocks.{block_num}.att.x_a"),
                "att_x_g" => format!("blocks.{block_num}.att.x_g"),
                "att_w0" => format!("blocks.{block_num}.att.w0"),
                "att_w1" => format!("blocks.{block_num}.att.w1"),
                "att_w2" => format!("blocks.{block_num}.att.w2"),
                "att_a0" => format!("blocks.{block_num}.att.a0"),
                "att_a1" => format!("blocks.{block_num}.att.a1"),
                "att_a2" => format!("blocks.{block_num}.att.a2"),
                "att_g1" => format!("blocks.{block_num}.att.g1"),
                "att_g2" => format!("blocks.{block_num}.att.g2"),
                "att_v0" => format!("blocks.{block_num}.att.v0"),
                "att_v1" => format!("blocks.{block_num}.att.v1"),
                "att_v2" => format!("blocks.{block_num}.att.v2"),
                "att_r_k" => format!("blocks.{block_num}.att.r_k"),
                "att_k_k" => format!("blocks.{block_num}.att.k_k"),
                "att_k_a" => format!("blocks.{block_num}.att.k_a"),

                _ => return None,
            };
            return Some(mapped);
        }
    }

    None
}

impl<'a> GgufReader<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self, GgufError> {
        let mut cursor = Cursor::new(data);

        // Read header
        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = cursor.read_u32()?;
        if version < 2 || version > 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = cursor.read_u64()?;
        let metadata_kv_count = cursor.read_u64()?;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = cursor.read_string()?;
            let value = cursor.read_metadata_value()?;
            metadata.insert(key, value);
        }

        // Get alignment from metadata or use default
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Read tensor infos
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let n_dimensions = cursor.read_u32()?;

            let mut dimensions = Vec::with_capacity(n_dimensions as usize);
            for _ in 0..n_dimensions {
                dimensions.push(cursor.read_u64()?);
            }

            let tensor_type = GgmlType::from(cursor.read_u32()?);
            let offset = cursor.read_u64()?;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    dimensions,
                    tensor_type,
                    offset,
                },
            );
        }

        // Calculate tensor data offset (aligned)
        let tensor_data_offset = align_offset(cursor.position as u64, alignment);

        let name_map = build_rwkv_name_map(&tensors);

        Ok(Self {
            data,
            version,
            tensor_count,
            metadata,
            tensors,
            tensor_data_offset,
            name_map,
        })
    }

    pub fn get_tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let start = (self.tensor_data_offset + info.offset) as usize;
        let end = start + info.data_size();
        &self.data[start..end]
    }

    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }
}

fn align_offset(offset: u64, alignment: u64) -> u64 {
    offset + (alignment - (offset % alignment)) % alignment
}

struct Cursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.remaining() < n {
            return Err(GgufError::UnexpectedEof);
        }
        let bytes = &self.data[self.position..self.position + n];
        self.position += n;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0] as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let bytes = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let bytes = self.read_bytes(8)?;
        Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_metadata_value(&mut self) -> Result<MetadataValue, GgufError> {
        let value_type = self.read_u32()?;
        self.read_metadata_value_of_type(value_type)
    }

    fn read_metadata_value_of_type(&mut self, value_type: u32) -> Result<MetadataValue, GgufError> {
        match value_type {
            0 => Ok(MetadataValue::Uint8(self.read_u8()?)),
            1 => Ok(MetadataValue::Int8(self.read_i8()?)),
            2 => Ok(MetadataValue::Uint16(self.read_u16()?)),
            3 => Ok(MetadataValue::Int16(self.read_i16()?)),
            4 => Ok(MetadataValue::Uint32(self.read_u32()?)),
            5 => Ok(MetadataValue::Int32(self.read_i32()?)),
            6 => Ok(MetadataValue::Float32(self.read_f32()?)),
            7 => {
                let b = self.read_u8()?;
                Ok(MetadataValue::Bool(b != 0))
            }
            8 => Ok(MetadataValue::String(self.read_string()?)),
            9 => {
                let array_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(self.read_metadata_value_of_type(array_type)?);
                }
                Ok(MetadataValue::Array(values))
            }
            10 => Ok(MetadataValue::Uint64(self.read_u64()?)),
            11 => Ok(MetadataValue::Int64(self.read_i64()?)),
            12 => Ok(MetadataValue::Float64(self.read_f64()?)),
            _ => Err(GgufError::InvalidValueType(value_type)),
        }
    }
}

impl<'a> GgufReader<'a> {
    fn resolve_name(&self, name: &str) -> Option<&str> {
        self.name_map.get(name).map(|s| s.as_str())
    }

    fn try_get_fused_slice(&self, name: &str) -> Option<(String, usize)> {
        if !name.starts_with("blocks.") || !name.contains(".att.x_") {
            return None;
        }

        // Handle fused time_maa tensor (x_r, x_w, x_k, x_v, x_a, x_g)
        // GGUF's time_mix_lerp_fused maps to time_maa
        let suffixes = [
            (".att.x_r", 0),
            (".att.x_w", 1),
            (".att.x_k", 2),
            (".att.x_v", 3),
            (".att.x_a", 4),
            (".att.x_g", 5),
        ];

        for (suffix, index) in suffixes {
            if let Some(prefix) = name.strip_suffix(suffix) {
                let fused_name = format!("{prefix}.att.time_maa");
                if self.name_map.contains_key(&fused_name) {
                    return Some((fused_name, index));
                }
            }
        }

        None
    }
}

impl<'a> super::loader::Reader for GgufReader<'a> {
    fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.name_map.keys().map(|s| s.as_str()).collect();
        for key in self.name_map.keys() {
            // Virtual tensors from time_maa (includes time_mix_lerp_fused which maps to time_maa)
            if key.ends_with(".att.time_maa") {
                if let Some(prefix) = key.strip_suffix(".att.time_maa") {
                    for suffix in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
                        let virtual_name = format!("{prefix}.att.{suffix}");
                        if !self.name_map.contains_key(&virtual_name) {
                            names.push(Box::leak(virtual_name.into_boxed_str()));
                        }
                    }
                }
            }
        }
        names
    }

    fn contains(&self, name: &str) -> bool {
        if self.name_map.contains_key(name) {
            return true;
        }
        self.try_get_fused_slice(name).is_some()
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        if let Some((fused_name, _)) = self.try_get_fused_slice(name) {
            let gguf_name = self
                .resolve_name(&fused_name)
                .ok_or_else(|| GgufError::TensorNotFound(fused_name.clone()))?;
            let info = self
                .tensors
                .get(gguf_name)
                .ok_or_else(|| GgufError::TensorNotFound(fused_name))?;
            let emb_size = info.dimensions[0] as usize;
            // Return 1D shape for sliced vectors
            return Ok(vec![emb_size]);
        }

        let gguf_name = self
            .resolve_name(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let info = self
            .tensors
            .get(gguf_name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let mut shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();

        // Handle r_k tensor: GGUF stores as 1D [768], SafeTensors as 2D [num_head, head_dim]
        // Infer num_head by looking up a1 tensor to get head_dim
        if shape.len() == 1 && name.ends_with(".att.r_k") {
            if let Some(block_prefix) = name.strip_suffix(".att.r_k") {
                let a1_name = format!("{}.att.a1", block_prefix);
                if let Some(a1_gguf) = self.resolve_name(&a1_name) {
                    if let Some(a1_info) = self.tensors.get(a1_gguf) {
                        // a1 shape in GGUF is [emb, head_dim], so head_dim is dimensions[1]
                        let head_dim = a1_info.dimensions[1] as usize;
                        let total = shape[0];
                        let num_head = total / head_dim;
                        return Ok(vec![num_head, head_dim]);
                    }
                }
            }
        }

        // Reverse 2D+ tensor shapes to match SafeTensors convention
        // GGUF stores [768, 65536], SafeTensors expects [65536, 768]
        if shape.len() > 1 {
            shape.reverse();
        }
        Ok(shape)
    }

    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        if let Some((fused_name, slice_idx)) = self.try_get_fused_slice(name) {
            let gguf_name = self
                .resolve_name(&fused_name)
                .ok_or_else(|| GgufError::TensorNotFound(fused_name.clone()))?;
            let info = self
                .tensors
                .get(gguf_name)
                .ok_or_else(|| GgufError::TensorNotFound(fused_name))?;

            let dtype = info
                .tensor_type
                .to_dtype()
                .ok_or_else(|| GgufError::UnsupportedTensorType(info.tensor_type))?;

            let emb_size = info.dimensions[0] as usize;
            let element_size = info.tensor_type.type_size();
            let data = self.get_tensor_data(info);

            // GGUF tensor shape [768, 1, 1, 6] in row-major order means last dimension varies fastest
            // Data layout: [s0e0, s0e1, ..., s0e767, s1e0, s1e1, ..., s1e767, ...]
            // Each slice is contiguous, with emb_size elements per slice
            let slice_start = slice_idx * emb_size * element_size;
            let slice_end = slice_start + emb_size * element_size;
            let slice_data = Cow::Borrowed(&data[slice_start..slice_end]);

            // Return 2D shape [emb_size, 1] for sliced vectors to match SafeTensors convention
            // from_slice_rev([x, 1]) -> Shape::new(1, x, 1, 1) = (1, x, 1, 1)
            return Ok((dtype, vec![emb_size, 1], slice_data));
        }

        let gguf_name = self
            .resolve_name(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let info = self
            .tensors
            .get(gguf_name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;

        let dtype = info
            .tensor_type
            .to_dtype()
            .ok_or_else(|| GgufError::UnsupportedTensorType(info.tensor_type))?;

        let mut shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();

        // Handle r_k tensor: GGUF stores as 1D [768], SafeTensors as 2D [num_head, head_dim]
        if shape.len() == 1 && name.ends_with(".att.r_k") {
            if let Some(block_prefix) = name.strip_suffix(".att.r_k") {
                let a1_name = format!("{}.att.a1", block_prefix);
                if let Some(a1_gguf) = self.resolve_name(&a1_name) {
                    if let Some(a1_info) = self.tensors.get(a1_gguf) {
                        let head_dim = a1_info.dimensions[1] as usize;
                        let total = shape[0];
                        let num_head = total / head_dim;
                        // Return 2D shape [num_head, head_dim] for from_slice_rev
                        let data = self.get_tensor_data(info);
                        return Ok((dtype, vec![num_head, head_dim], Cow::Borrowed(data)));
                    }
                }
            }
        }

        if shape.len() == 1 {
            // Convert 1D [x] to 2D [x, 1] for tensor loading
            // from_slice_rev([x, 1]) -> Shape::new(1, x, 1, 1) = (1, x, 1, 1)
            shape.push(1);
        } else {
            // Reverse 2D+ tensor shapes to match SafeTensors convention
            shape.reverse();
        }
        let data = self.get_tensor_data(info);

        Ok((dtype, shape, Cow::Borrowed(data)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GgmlType::F32.type_size(), 4);
        assert_eq!(GgmlType::F16.type_size(), 2);
        assert_eq!(GgmlType::Q8_0.type_size(), 34);
        assert_eq!(GgmlType::Q4_0.type_size(), 18);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}

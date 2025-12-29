use std::{borrow::Cow, collections::HashMap};

use half::f16;
use safetensors::{Dtype, SafeTensorError};
use thiserror::Error;

use super::loader::ReaderTensor;

/// Dequantize GGUF Q8_0 data to F16.
/// Q8_0 format: blocks of 32 elements, each block is [scale: f16, data: i8[32]] = 34 bytes
fn dequantize_q8_0_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 34; // 2 bytes scale + 32 bytes data

    let num_blocks = num_elements / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_elements * 2); // f16 = 2 bytes

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block_data = &data[block_start..block_start + BLOCK_BYTES];

        // First 2 bytes are the scale (f16)
        let scale_bytes = [block_data[0], block_data[1]];
        let scale = f16::from_le_bytes(scale_bytes);
        let scale_f32 = scale.to_f32();

        // Next 32 bytes are the quantized values (i8)
        for i in 0..BLOCK_SIZE {
            let quant_val = block_data[2 + i] as i8;
            let dequant_val = (quant_val as f32) * scale_f32;
            let f16_val = f16::from_f32(dequant_val);
            output.extend_from_slice(&f16_val.to_le_bytes());
        }
    }

    output
}

/// Dequantize GGUF Q4_0 data to F16.
/// Q4_0 format: blocks of 32 elements, each block is [scale: f16, data: u8[16]] = 18 bytes
/// Each byte contains 2 4-bit values
fn dequantize_q4_0_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 bytes scale + 16 bytes data (32 4-bit values)

    let num_blocks = num_elements / BLOCK_SIZE;
    let mut output = Vec::with_capacity(num_elements * 2); // f16 = 2 bytes

    for block_idx in 0..num_blocks {
        let block_start = block_idx * BLOCK_BYTES;
        let block_data = &data[block_start..block_start + BLOCK_BYTES];

        // First 2 bytes are the scale (f16)
        let scale_bytes = [block_data[0], block_data[1]];
        let scale = f16::from_le_bytes(scale_bytes);
        let scale_f32 = scale.to_f32();

        // Next 16 bytes contain 32 4-bit values (2 per byte)
        // Q4_0 stores values as unsigned 0-15, subtract 8 to get signed -8 to 7
        for i in 0..16 {
            let byte = block_data[2 + i];
            // Low nibble first, then high nibble
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = ((byte >> 4) & 0x0F) as i8 - 8;

            let dequant_lo = (lo as f32) * scale_f32;
            let dequant_hi = (hi as f32) * scale_f32;

            output.extend_from_slice(&f16::from_f32(dequant_lo).to_le_bytes());
            output.extend_from_slice(&f16::from_f32(dequant_hi).to_le_bytes());
        }
    }

    output
}

/// Helper to extract scale and min from Q4_K/Q5_K scales array.
/// For j < 4: scale = scales[j] & 63, min = scales[j+4] & 63
/// For j >= 4: scale/min are packed across scales[j+4] and scales[j-4]/scales[j]
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize GGUF Q4_K data to F16.
/// Q4_K format: 256 elements per super-block
/// Layout: [d: f16, dmin: f16, scales: 12 bytes, qs: 128 bytes] = 144 bytes
/// 8 sub-blocks of 32 elements each, scales/mins quantized with 6 bits
fn dequantize_q4_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144; // 2 + 2 + 12 + 128

    let num_blocks = num_elements / QK_K;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse header: d (f16), dmin (f16)
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16]; // 12 bytes
        let qs = &block[16..144]; // 128 bytes

        // Process 8 sub-blocks of 32 elements (but qs packs 64 elements per 32 bytes)
        // Each byte in qs contains 2 4-bit values
        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_offset = j / 2; // 32 bytes per 64 elements

            // First 32 elements (low nibble)
            for l in 0..32 {
                let q_byte = qs[q_offset + l];
                let val = d1 * ((q_byte & 0xF) as f32) - m1_val;
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            // Next 32 elements (high nibble)
            for l in 0..32 {
                let q_byte = qs[q_offset + l];
                let val = d2 * ((q_byte >> 4) as f32) - m2_val;
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }

            is += 2;
        }
    }

    output
}

/// Dequantize GGUF Q5_K data to F16.
/// Q5_K format: 256 elements per super-block
/// Layout: [d: f16, dmin: f16, scales: 12 bytes, qh: 32 bytes, qs: 128 bytes] = 176 bytes
/// 8 sub-blocks of 32 elements each, 5-bit quantization (4 low + 1 high)
fn dequantize_q5_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176; // 2 + 2 + 12 + 32 + 128

    let num_blocks = num_elements / QK_K;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse header
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16]; // 12 bytes
        let qh = &block[16..48]; // 32 bytes - high bits
        let ql = &block[48..176]; // 128 bytes - low 4 bits

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let ql_offset = j / 2;

            // First 32 elements (low nibble + high bit from u1)
            for l in 0..32 {
                let q_low = ql[ql_offset + l] & 0xF;
                let q_high = if qh[l] & u1 != 0 { 16 } else { 0 };
                let val = d1 * ((q_low + q_high) as f32) - m1_val;
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }

            // Next 32 elements (high nibble + high bit from u2)
            for l in 0..32 {
                let q_low = ql[ql_offset + l] >> 4;
                let q_high = if qh[l] & u2 != 0 { 16 } else { 0 };
                let val = d2 * ((q_low + q_high) as f32) - m2_val;
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output
}

/// Dequantize GGUF Q6_K data to F16.
/// Q6_K format: 256 elements per super-block
/// Layout: [ql: 128 bytes, qh: 64 bytes, scales: 16 bytes, d: f16] = 210 bytes
/// 16 sub-blocks of 16 elements each, 6-bit quantization (4 low + 2 high)
fn dequantize_q6_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210; // 128 + 64 + 16 + 2

    let num_blocks = num_elements / QK_K;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse block (note: d is at the END for Q6_K)
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208]; // int8_t scales[16]
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        // Process in chunks of 128 elements (2 iterations for 256 total)
        let mut ql_idx = 0usize;
        let mut qh_idx = 0usize;
        let mut sc_idx = 0usize;

        for _n in (0..QK_K).step_by(128) {
            // Output positions 0-31
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_idx + l] & 0xF) | (((qh[qh_idx + l] >> 0) & 3) << 4)) as i8 - 32;
                let sc0 = scales[sc_idx + is] as i8;
                let val = d * (sc0 as f32) * (q1 as f32);
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            // Output positions 32-63
            for l in 0..32 {
                let is = l / 16;
                let q2 =
                    ((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) as i8 - 32;
                let sc2 = scales[sc_idx + is + 2] as i8;
                let val = d * (sc2 as f32) * (q2 as f32);
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            // Output positions 64-95
            for l in 0..32 {
                let is = l / 16;
                let q3 = ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) as i8 - 32;
                let sc4 = scales[sc_idx + is + 4] as i8;
                let val = d * (sc4 as f32) * (q3 as f32);
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }
            // Output positions 96-127
            for l in 0..32 {
                let is = l / 16;
                let q4 =
                    ((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) as i8 - 32;
                let sc6 = scales[sc_idx + is + 6] as i8;
                let val = d * (sc6 as f32) * (q4 as f32);
                output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
            }

            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
        }
    }

    output
}

/// Dequantize GGUF Q3_K data to F16.
/// Q3_K format: 256 elements per super-block
/// Layout: [hmask: 32 bytes, qs: 64 bytes, scales: 12 bytes, d: f16] = 110 bytes
/// 16 sub-blocks of 16 elements each, 3-bit quantization (2 low + 1 high)
fn dequantize_q3_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 110; // 32 + 64 + 12 + 2

    let num_blocks = num_elements / QK_K;
    let mut output = Vec::with_capacity(num_elements * 2);

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse block
        let hmask = &block[0..32]; // high bits
        let qs = &block[32..96]; // low 2 bits (packed)
        let scales_raw = &block[96..108]; // 12 bytes
        let d_all = f16::from_le_bytes([block[108], block[109]]).to_f32();

        // Unpack scales (complex bit manipulation from llama.cpp)
        let mut aux = [0u32; 4];
        aux[0] = u32::from_le_bytes([scales_raw[0], scales_raw[1], scales_raw[2], scales_raw[3]]);
        aux[1] = u32::from_le_bytes([scales_raw[4], scales_raw[5], scales_raw[6], scales_raw[7]]);
        aux[2] = u32::from_le_bytes([scales_raw[8], scales_raw[9], scales_raw[10], scales_raw[11]]);

        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

        // Convert aux to scales array (as i8)
        let scales: [i8; 16] = [
            (aux[0] & 0xFF) as i8,
            ((aux[0] >> 8) & 0xFF) as i8,
            ((aux[0] >> 16) & 0xFF) as i8,
            ((aux[0] >> 24) & 0xFF) as i8,
            (aux[1] & 0xFF) as i8,
            ((aux[1] >> 8) & 0xFF) as i8,
            ((aux[1] >> 16) & 0xFF) as i8,
            ((aux[1] >> 24) & 0xFF) as i8,
            (aux[2] & 0xFF) as i8,
            ((aux[2] >> 8) & 0xFF) as i8,
            ((aux[2] >> 16) & 0xFF) as i8,
            ((aux[2] >> 24) & 0xFF) as i8,
            (aux[3] & 0xFF) as i8,
            ((aux[3] >> 8) & 0xFF) as i8,
            ((aux[3] >> 16) & 0xFF) as i8,
            ((aux[3] >> 24) & 0xFF) as i8,
        ];

        let mut q_idx = 0usize;
        let mut is = 0usize;
        let mut m: u8 = 1;

        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0u8;
            for _j in 0..4 {
                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;

                for l in 0..16 {
                    let q_val = ((qs[q_idx + l] >> shift) & 3) as i8;
                    let h_val = if hmask[l] & m != 0 { 0i8 } else { -4i8 };
                    let val = dl * ((q_val + h_val) as f32);
                    output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
                }

                let dl = d_all * ((scales[is] as i32 - 32) as f32);
                is += 1;

                for l in 0..16 {
                    let q_val = ((qs[q_idx + l + 16] >> shift) & 3) as i8;
                    let h_val = if hmask[l + 16] & m != 0 { 0i8 } else { -4i8 };
                    let val = dl * ((q_val + h_val) as f32);
                    output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
                }

                shift += 2;
                m <<= 1;
            }
            q_idx += 32;
        }
    }

    output
}

/// Dequantize GGUF Q2_K data to F16.
/// Q2_K format: 256 elements per super-block
/// Layout: [scales: 16 bytes, qs: 64 bytes, d: f16, dmin: f16] = 84 bytes
/// 16 sub-blocks of 16 elements each, 2-bit quantization
fn dequantize_q2_k_to_f16(data: &[u8], num_elements: usize) -> Vec<u8> {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 84; // 16 + 64 + 2 + 2

    let num_blocks = num_elements / QK_K;
    let mut output = Vec::with_capacity(num_elements * 2);

    for block_idx in 0..num_blocks {
        let block = &data[block_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        // Parse block - note the layout order
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
        let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

        let mut is = 0usize;
        let mut q_idx = 0usize;

        for _n in (0..QK_K).step_by(128) {
            let mut shift = 0u8;
            for _j in 0..4 {
                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0xF) as f32);
                let ml = dmin * ((sc >> 4) as f32);

                for l in 0..16 {
                    let q_val = ((qs[q_idx + l] >> shift) & 3) as i8;
                    let val = dl * (q_val as f32) - ml;
                    output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
                }

                let sc = scales[is];
                is += 1;
                let dl = d * ((sc & 0xF) as f32);
                let ml = dmin * ((sc >> 4) as f32);

                for l in 0..16 {
                    let q_val = ((qs[q_idx + l + 16] >> shift) & 3) as i8;
                    let val = dl * (q_val as f32) - ml;
                    output.extend_from_slice(&f16::from_f32(val).to_le_bytes());
                }

                shift += 2;
            }
            q_idx += 32;
        }
    }

    output
}

/// Repack GGUF Q8_0 data to web-rwkv Int8 format.
/// GGUF Q8_0: 32 elements/block, [scale: f16, data: i8[32]] = 34 bytes
/// web-rwkv Int8: 128 elements/block, [weights: u8[128]], separate [min, max] per block
/// Returns (weights, minmax) where minmax is interleaved [min0, max0, min1, max1, ...]
pub fn repack_q8_0_to_int8(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<f16>) {
    const GGUF_BLOCK_SIZE: usize = 32;
    const GGUF_BLOCK_BYTES: usize = 34;
    const RWKV_BLOCK_SIZE: usize = 128;

    let num_gguf_blocks = num_elements / GGUF_BLOCK_SIZE;
    let num_rwkv_blocks = num_elements / RWKV_BLOCK_SIZE;

    let mut weights = Vec::with_capacity(num_elements);
    let mut minmax = Vec::with_capacity(num_rwkv_blocks * 2);

    // Process 4 GGUF blocks at a time to make 1 web-rwkv block
    for rwkv_block_idx in 0..num_rwkv_blocks {
        let gguf_start = rwkv_block_idx * 4;

        // First pass: find min/max across 4 GGUF blocks (128 elements)
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for i in 0..4 {
            let block_offset = (gguf_start + i) * GGUF_BLOCK_BYTES;
            let scale = f16::from_le_bytes([data[block_offset], data[block_offset + 1]]).to_f32();

            for j in 0..GGUF_BLOCK_SIZE {
                let q = data[block_offset + 2 + j] as i8;
                let val = (q as f32) * scale;
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Store min/max for this block
        minmax.push(f16::from_f32(min_val));
        minmax.push(f16::from_f32(max_val));

        // Second pass: rescale values to 0-255 range
        let range = max_val - min_val;
        let inv_range = if range > 0.0 { 255.0 / range } else { 0.0 };

        for i in 0..4 {
            let block_offset = (gguf_start + i) * GGUF_BLOCK_BYTES;
            let scale = f16::from_le_bytes([data[block_offset], data[block_offset + 1]]).to_f32();

            for j in 0..GGUF_BLOCK_SIZE {
                let q = data[block_offset + 2 + j] as i8;
                let val = (q as f32) * scale;
                let normalized = ((val - min_val) * inv_range).round() as u8;
                weights.push(normalized);
            }
        }
    }

    // Handle remaining elements if num_elements is not divisible by 128
    let remaining_start = num_rwkv_blocks * 4;
    if remaining_start < num_gguf_blocks {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        let mut temp_vals = Vec::new();

        for i in remaining_start..num_gguf_blocks {
            let block_offset = i * GGUF_BLOCK_BYTES;
            let scale = f16::from_le_bytes([data[block_offset], data[block_offset + 1]]).to_f32();

            for j in 0..GGUF_BLOCK_SIZE {
                let q = data[block_offset + 2 + j] as i8;
                let val = (q as f32) * scale;
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                temp_vals.push(val);
            }
        }

        if !temp_vals.is_empty() {
            minmax.push(f16::from_f32(min_val));
            minmax.push(f16::from_f32(max_val));

            let range = max_val - min_val;
            let inv_range = if range > 0.0 { 255.0 / range } else { 0.0 };

            for val in temp_vals {
                let normalized = ((val - min_val) * inv_range).round() as u8;
                weights.push(normalized);
            }

            // Pad to 128 elements
            while weights.len() % RWKV_BLOCK_SIZE != 0 {
                weights.push(0);
            }
        }
    }

    (weights, minmax)
}

/// Repack GGUF Q4_0 data to web-rwkv NF4 format.
/// GGUF Q4_0: 32 elements/block, [scale: f16, data: u8[16]] = 18 bytes (4-bit packed)
/// web-rwkv NF4: 64 elements/block, [weights: u8[32]], separate [absmax] per block
/// Note: NF4 uses a quantile-based lookup table, so this is an approximation.
/// Returns (weights, absmax)
pub fn repack_q4_0_to_nf4(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<f16>) {
    const GGUF_BLOCK_SIZE: usize = 32;
    const GGUF_BLOCK_BYTES: usize = 18;
    const NF4_BLOCK_SIZE: usize = 64;

    // NF4 quantile values (normalized to [-1, 1])
    const NF4_QUANT: [f32; 16] = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ];

    let _num_gguf_blocks = num_elements / GGUF_BLOCK_SIZE;
    let num_nf4_blocks = num_elements / NF4_BLOCK_SIZE;

    let mut weights = Vec::with_capacity(num_elements / 2); // 4-bit packed
    let mut absmax = Vec::with_capacity(num_nf4_blocks);

    // Process 2 GGUF blocks at a time to make 1 NF4 block (64 elements)
    for nf4_block_idx in 0..num_nf4_blocks {
        let gguf_start = nf4_block_idx * 2;

        // First pass: find absmax across 2 GGUF blocks (64 elements)
        let mut max_abs = 0.0f32;

        for i in 0..2 {
            let block_offset = (gguf_start + i) * GGUF_BLOCK_BYTES;
            let scale = f16::from_le_bytes([data[block_offset], data[block_offset + 1]]).to_f32();

            for j in 0..16 {
                let byte = data[block_offset + 2 + j];
                let lo = ((byte & 0x0F) as i8 - 8) as f32 * scale;
                let hi = (((byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;
                max_abs = max_abs.max(lo.abs()).max(hi.abs());
            }
        }

        absmax.push(f16::from_f32(max_abs));

        // Second pass: quantize to NF4
        let inv_absmax = if max_abs > 0.0 { 1.0 / max_abs } else { 0.0 };

        for i in 0..2 {
            let block_offset = (gguf_start + i) * GGUF_BLOCK_BYTES;
            let scale = f16::from_le_bytes([data[block_offset], data[block_offset + 1]]).to_f32();

            for j in 0..16 {
                let byte = data[block_offset + 2 + j];
                let lo = ((byte & 0x0F) as i8 - 8) as f32 * scale;
                let hi = (((byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;

                // Normalize to [-1, 1] and find closest NF4 value
                let lo_norm = lo * inv_absmax;
                let hi_norm = hi * inv_absmax;

                let lo_idx = NF4_QUANT
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - lo_norm)
                            .abs()
                            .partial_cmp(&(*b - lo_norm).abs())
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0) as u8;

                let hi_idx = NF4_QUANT
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (*a - hi_norm)
                            .abs()
                            .partial_cmp(&(*b - hi_norm).abs())
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0) as u8;

                // Pack two 4-bit values into one byte
                weights.push(lo_idx | (hi_idx << 4));
            }
        }
    }

    (weights, absmax)
}

/// Repack GGUF Q4_K data to web-rwkv Int8 format.
/// Q4_K: 256 elements per super-block, [d: f16, dmin: f16, scales: 12B, qs: 128B] = 144 bytes
/// web-rwkv Int8: 128 elements/block, [weights: u8[128]], separate [min, max] per block
pub fn repack_q4_k_to_int8(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<f16>) {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 144;
    const RWKV_BLOCK_SIZE: usize = 128;

    let num_super_blocks = num_elements / QK_K;
    let num_rwkv_blocks = num_elements / RWKV_BLOCK_SIZE;

    let mut weights = Vec::with_capacity(num_elements);
    let mut minmax = Vec::with_capacity(num_rwkv_blocks * 2);

    // Temporary buffer for dequantized values
    let mut dequant = vec![0.0f32; QK_K];

    for sb_idx in 0..num_super_blocks {
        let block = &data[sb_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qs = &block[16..144];

        // Dequantize all 256 elements
        let mut is = 0usize;
        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let q_offset = j / 2;
            for l in 0..32 {
                let q_byte = qs[q_offset + l];
                dequant[j + l] = d1 * ((q_byte & 0xF) as f32) - m1_val;
                dequant[j + 32 + l] = d2 * ((q_byte >> 4) as f32) - m2_val;
            }
            is += 2;
        }

        // Convert to Int8 format (2 blocks of 128 elements each)
        for block_idx in 0..2 {
            let start = block_idx * RWKV_BLOCK_SIZE;
            let chunk = &dequant[start..start + RWKV_BLOCK_SIZE];

            let min_val = chunk.iter().cloned().fold(f32::MAX, f32::min);
            let max_val = chunk.iter().cloned().fold(f32::MIN, f32::max);

            minmax.push(f16::from_f32(min_val));
            minmax.push(f16::from_f32(max_val));

            let range = max_val - min_val;
            let inv_range = if range > 0.0 { 255.0 / range } else { 0.0 };

            for &val in chunk {
                let normalized = ((val - min_val) * inv_range).round() as u8;
                weights.push(normalized);
            }
        }
    }

    (weights, minmax)
}

/// Repack GGUF Q5_K data to web-rwkv Int8 format.
/// Q5_K: 256 elements per super-block, [d: f16, dmin: f16, scales: 12B, qh: 32B, qs: 128B] = 176 bytes
pub fn repack_q5_k_to_int8(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<f16>) {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 176;
    const RWKV_BLOCK_SIZE: usize = 128;

    let num_super_blocks = num_elements / QK_K;
    let num_rwkv_blocks = num_elements / RWKV_BLOCK_SIZE;

    let mut weights = Vec::with_capacity(num_elements);
    let mut minmax = Vec::with_capacity(num_rwkv_blocks * 2);

    let mut dequant = vec![0.0f32; QK_K];

    for sb_idx in 0..num_super_blocks {
        let block = &data[sb_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];

        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for j in (0..QK_K).step_by(64) {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);

            let d1 = d * (sc0 as f32);
            let m1_val = dmin * (m0 as f32);
            let d2 = d * (sc1 as f32);
            let m2_val = dmin * (m1 as f32);

            let ql_offset = j / 2;

            for l in 0..32 {
                let q_low = ql[ql_offset + l] & 0xF;
                let q_high = if qh[l] & u1 != 0 { 16 } else { 0 };
                dequant[j + l] = d1 * ((q_low + q_high) as f32) - m1_val;
            }

            for l in 0..32 {
                let q_low = ql[ql_offset + l] >> 4;
                let q_high = if qh[l] & u2 != 0 { 16 } else { 0 };
                dequant[j + 32 + l] = d2 * ((q_low + q_high) as f32) - m2_val;
            }

            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        // Convert to Int8 format
        for block_idx in 0..2 {
            let start = block_idx * RWKV_BLOCK_SIZE;
            let chunk = &dequant[start..start + RWKV_BLOCK_SIZE];

            let min_val = chunk.iter().cloned().fold(f32::MAX, f32::min);
            let max_val = chunk.iter().cloned().fold(f32::MIN, f32::max);

            minmax.push(f16::from_f32(min_val));
            minmax.push(f16::from_f32(max_val));

            let range = max_val - min_val;
            let inv_range = if range > 0.0 { 255.0 / range } else { 0.0 };

            for &val in chunk {
                let normalized = ((val - min_val) * inv_range).round() as u8;
                weights.push(normalized);
            }
        }
    }

    (weights, minmax)
}

/// Repack GGUF Q6_K data to web-rwkv Int8 format.
/// Q6_K: 256 elements per super-block, [ql: 128B, qh: 64B, scales: 16B, d: f16] = 210 bytes
pub fn repack_q6_k_to_int8(data: &[u8], num_elements: usize) -> (Vec<u8>, Vec<f16>) {
    const QK_K: usize = 256;
    const BLOCK_BYTES: usize = 210;
    const RWKV_BLOCK_SIZE: usize = 128;

    let num_super_blocks = num_elements / QK_K;
    let num_rwkv_blocks = num_elements / RWKV_BLOCK_SIZE;

    let mut weights = Vec::with_capacity(num_elements);
    let mut minmax = Vec::with_capacity(num_rwkv_blocks * 2);

    let mut dequant = vec![0.0f32; QK_K];

    for sb_idx in 0..num_super_blocks {
        let block = &data[sb_idx * BLOCK_BYTES..][..BLOCK_BYTES];

        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

        let mut ql_idx = 0usize;
        let mut qh_idx = 0usize;
        let mut sc_idx = 0usize;
        let mut out_idx = 0usize;

        for _n in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_idx + l] & 0xF) | (((qh[qh_idx + l] >> 0) & 3) << 4)) as i8 - 32;
                let q2 =
                    ((ql[ql_idx + l + 32] & 0xF) | (((qh[qh_idx + l] >> 2) & 3) << 4)) as i8 - 32;
                let q3 = ((ql[ql_idx + l] >> 4) | (((qh[qh_idx + l] >> 4) & 3) << 4)) as i8 - 32;
                let q4 =
                    ((ql[ql_idx + l + 32] >> 4) | (((qh[qh_idx + l] >> 6) & 3) << 4)) as i8 - 32;

                let sc0 = scales[sc_idx + is] as i8;
                let sc2 = scales[sc_idx + is + 2] as i8;
                let sc4 = scales[sc_idx + is + 4] as i8;
                let sc6 = scales[sc_idx + is + 6] as i8;

                dequant[out_idx + l] = d * (sc0 as f32) * (q1 as f32);
                dequant[out_idx + l + 32] = d * (sc2 as f32) * (q2 as f32);
                dequant[out_idx + l + 64] = d * (sc4 as f32) * (q3 as f32);
                dequant[out_idx + l + 96] = d * (sc6 as f32) * (q4 as f32);
            }

            ql_idx += 64;
            qh_idx += 32;
            sc_idx += 8;
            out_idx += 128;
        }

        // Convert to Int8 format
        for block_idx in 0..2 {
            let start = block_idx * RWKV_BLOCK_SIZE;
            let chunk = &dequant[start..start + RWKV_BLOCK_SIZE];

            let min_val = chunk.iter().cloned().fold(f32::MAX, f32::min);
            let max_val = chunk.iter().cloned().fold(f32::MIN, f32::max);

            minmax.push(f16::from_f32(min_val));
            minmax.push(f16::from_f32(max_val));

            let range = max_val - min_val;
            let inv_range = if range > 0.0 { 255.0 / range } else { 0.0 };

            for &val in chunk {
                let normalized = ((val - min_val) * inv_range).round() as u8;
                weights.push(normalized);
            }
        }
    }

    (weights, minmax)
}

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

    pub fn to_u32(&self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q8_1 => 9,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q8K => 15,
            GgmlType::IQ2XXS => 16,
            GgmlType::IQ2XS => 17,
            GgmlType::IQ3XXS => 18,
            GgmlType::IQ1S => 19,
            GgmlType::IQ4NL => 20,
            GgmlType::IQ3S => 21,
            GgmlType::IQ2S => 22,
            GgmlType::IQ4XS => 23,
            GgmlType::I8 => 24,
            GgmlType::I16 => 25,
            GgmlType::I32 => 26,
            GgmlType::I64 => 27,
            GgmlType::F64 => 28,
            GgmlType::IQ1M => 29,
            GgmlType::BF16 => 30,
            GgmlType::TQ1_0 => 34,
            GgmlType::TQ2_0 => 35,
            GgmlType::Unknown(v) => *v,
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q8_0
                | GgmlType::Q8_1
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8K
        )
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

        let num_elements = info.num_elements() as usize;
        let mut shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();

        // Handle quantized types by dequantizing to F16
        if info.tensor_type.is_quantized() {
            let data = self.get_tensor_data(info);

            // For K-quants, calculate actual elements from raw data size
            let actual_elements = match info.tensor_type {
                GgmlType::Q4K => (data.len() / 144) * 256,
                GgmlType::Q5K => (data.len() / 176) * 256,
                GgmlType::Q6K => (data.len() / 210) * 256,
                GgmlType::Q3K => (data.len() / 110) * 256,
                GgmlType::Q2K => (data.len() / 84) * 256,
                _ => num_elements,
            };

            let mut dequantized = match info.tensor_type {
                GgmlType::Q8_0 => dequantize_q8_0_to_f16(data, num_elements),
                GgmlType::Q4_0 => dequantize_q4_0_to_f16(data, num_elements),
                GgmlType::Q4K => dequantize_q4_k_to_f16(data, actual_elements),
                GgmlType::Q5K => dequantize_q5_k_to_f16(data, actual_elements),
                GgmlType::Q6K => dequantize_q6_k_to_f16(data, actual_elements),
                GgmlType::Q3K => dequantize_q3_k_to_f16(data, actual_elements),
                GgmlType::Q2K => dequantize_q2_k_to_f16(data, actual_elements),
                _ => return Err(GgufError::UnsupportedTensorType(info.tensor_type).into()),
            };

            // Ensure output matches expected size (handle K-quant padding)
            let expected_bytes = num_elements * 2; // f16 = 2 bytes
            if dequantized.len() > expected_bytes {
                dequantized.truncate(expected_bytes);
            } else if dequantized.len() < expected_bytes {
                // Pad with zeros if needed
                dequantized.resize(expected_bytes, 0);
            }

            // Reverse 2D+ tensor shapes to match SafeTensors convention
            if shape.len() > 1 {
                shape.reverse();
            } else {
                shape.push(1);
            }

            return Ok((Dtype::F16, shape, Cow::Owned(dequantized)));
        }

        let dtype = info
            .tensor_type
            .to_dtype()
            .ok_or_else(|| GgufError::UnsupportedTensorType(info.tensor_type))?;

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

    fn quantized_tensor(&self, name: &str) -> Option<(u32, &[u8])> {
        // Don't support direct loading for virtual/sliced tensors
        if self.try_get_fused_slice(name).is_some() {
            return None;
        }

        let gguf_name = self.resolve_name(name)?;
        let info = self.tensors.get(gguf_name)?;

        // Return data for quantized types that support direct loading
        // Note: K-quants (Q4K, Q5K, Q6K) native shaders are slower than F16 dequant path
        // due to per-element dequantization overhead. Use F16 path for better performance.
        match info.tensor_type {
            GgmlType::Q8_0 | GgmlType::Q4_0 => {
                let data = self.get_tensor_data(info);
                Some((info.tensor_type.to_u32(), data))
            }
            _ => None,
        }
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
    fn test_dequantize_q8_0() {
        // Create a Q8_0 block: scale=1.0, values=[0, 1, 2, ..., 31]
        let scale = f16::from_f32(1.0);
        let mut block = Vec::with_capacity(34);
        block.extend_from_slice(&scale.to_le_bytes());
        for i in 0i8..32 {
            block.push(i as u8);
        }

        let result = dequantize_q8_0_to_f16(&block, 32);
        assert_eq!(result.len(), 64); // 32 f16 values = 64 bytes

        // Check first few values
        let val0 = f16::from_le_bytes([result[0], result[1]]);
        let val1 = f16::from_le_bytes([result[2], result[3]]);
        assert!((val0.to_f32() - 0.0).abs() < 0.01);
        assert!((val1.to_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dequantize_q4_0() {
        // Create a Q4_0 block: scale=1.0, values stored as nibbles
        let scale = f16::from_f32(1.0);
        let mut block = Vec::with_capacity(18);
        block.extend_from_slice(&scale.to_le_bytes());
        // 16 bytes for 32 4-bit values
        // Each byte: low nibble first, high nibble second
        // Values 0-15 map to -8 to 7 after subtracting 8
        for _ in 0..16 {
            block.push(0x88); // Both nibbles = 8, which maps to 0
        }

        let result = dequantize_q4_0_to_f16(&block, 32);
        assert_eq!(result.len(), 64); // 32 f16 values = 64 bytes

        // All values should be 0 (8 - 8 = 0)
        let val0 = f16::from_le_bytes([result[0], result[1]]);
        assert!((val0.to_f32() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}

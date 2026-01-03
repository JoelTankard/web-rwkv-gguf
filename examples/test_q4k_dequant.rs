//! Test to verify Q4_K dequantization correctness
//! Run with: cargo run --release --example test_q4k_dequant

use half::f16;

const QK_K: usize = 256;
const BLOCK_BYTES: usize = 144;

/// CPU reference implementation of get_scale_min_k4
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// CPU reference dequantization of a single Q4_K block
fn dequantize_q4k_block_cpu(block: &[u8]) -> [f32; QK_K] {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut output = [0.0f32; QK_K];
    let mut out_idx = 0usize;
    let mut is = 0usize;

    for j in (0..QK_K).step_by(64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, scales);

        let d1 = d * (sc0 as f32);
        let m1_val = dmin * (m0 as f32);
        let d2 = d * (sc1 as f32);
        let m2_val = dmin * (m1 as f32);

        let q_offset = j / 2;

        // First 32 elements: low nibbles
        for l in 0..32 {
            let q_byte = qs[q_offset + l];
            output[out_idx] = d1 * ((q_byte & 0xF) as f32) - m1_val;
            out_idx += 1;
        }
        // Next 32 elements: high nibbles
        for l in 0..32 {
            let q_byte = qs[q_offset + l];
            output[out_idx] = d2 * ((q_byte >> 4) as f32) - m2_val;
            out_idx += 1;
        }
        is += 2;
    }

    output
}

/// Simulated shader dequantization (matching matmul_vec_q4k_v2.wgsl logic)
fn dequantize_q4k_block_shader(block: &[u8]) -> [f32; QK_K] {
    // Read as u32 array (simulating GPU memory layout)
    let block_u32: Vec<u32> = block
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let d_dmin_packed = block_u32[0];
    let d = f16::from_bits((d_dmin_packed & 0xFFFF) as u16).to_f32();
    let dmin = f16::from_bits((d_dmin_packed >> 16) as u16).to_f32();

    let scales_u32 = [block_u32[1], block_u32[2], block_u32[3]];

    fn get_scale_byte(scales_u32: &[u32; 3], byte_idx: u32) -> u32 {
        (scales_u32[(byte_idx / 4) as usize] >> ((byte_idx % 4) * 8)) & 0xFF
    }

    fn get_scale_min_k4_shader(j: u32, scales_u32: &[u32; 3]) -> (f32, f32) {
        let sc: u32;
        let m: u32;
        if j < 4 {
            sc = get_scale_byte(scales_u32, j) & 63;
            m = get_scale_byte(scales_u32, j + 4) & 63;
        } else {
            let b1 = get_scale_byte(scales_u32, j + 4);
            let b2 = get_scale_byte(scales_u32, j - 4);
            let b3 = get_scale_byte(scales_u32, j);
            sc = (b1 & 0xF) | ((b2 >> 6) << 4);
            m = (b1 >> 4) | ((b3 >> 6) << 4);
        }
        (sc as f32, m as f32)
    }

    let mut output = [0.0f32; QK_K];
    let mut out_idx = 0usize;

    for j64 in 0..4u32 {
        let (sc0, m0) = get_scale_min_k4_shader(j64 * 2, &scales_u32);
        let (sc1, m1) = get_scale_min_k4_shader(j64 * 2 + 1, &scales_u32);
        let d1 = d * sc0;
        let m1_val = dmin * m0;
        let d2 = d * sc1;
        let m2_val = dmin * m1;

        let qs_base = 4 + (j64 * 8) as usize;

        for l in 0..8usize {
            let qs = block_u32[qs_base + l];

            // Low nibbles (elements 0-31 within this 64-element group)
            let q0 = (qs & 0xF) as f32;
            let q1 = ((qs >> 8) & 0xF) as f32;
            let q2 = ((qs >> 16) & 0xF) as f32;
            let q3 = ((qs >> 24) & 0xF) as f32;
            output[out_idx] = d1 * q0 - m1_val;
            output[out_idx + 1] = d1 * q1 - m1_val;
            output[out_idx + 2] = d1 * q2 - m1_val;
            output[out_idx + 3] = d1 * q3 - m1_val;
            out_idx += 4;
        }

        for l in 0..8usize {
            let qs = block_u32[qs_base + l];

            // High nibbles (elements 32-63 within this 64-element group)
            let q4 = ((qs >> 4) & 0xF) as f32;
            let q5 = ((qs >> 12) & 0xF) as f32;
            let q6 = ((qs >> 20) & 0xF) as f32;
            let q7 = ((qs >> 28) & 0xF) as f32;
            output[out_idx] = d2 * q4 - m2_val;
            output[out_idx + 1] = d2 * q5 - m2_val;
            output[out_idx + 2] = d2 * q6 - m2_val;
            output[out_idx + 3] = d2 * q7 - m2_val;
            out_idx += 4;
        }
    }

    output
}

fn main() {
    println!("Testing Q4_K dequantization...\n");

    // Create a test Q4_K block with known values
    let mut block = [0u8; BLOCK_BYTES];

    // Set d and dmin
    let d = f16::from_f32(0.5);
    let dmin = f16::from_f32(0.1);
    block[0..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());

    // Set scales (12 bytes, values 0-63)
    for i in 0..12 {
        block[4 + i] = (i as u8 * 5) % 64;
    }

    // Set qs (128 bytes, random-ish pattern)
    for i in 0..128 {
        block[16 + i] = ((i * 7 + 3) % 256) as u8;
    }

    let cpu_result = dequantize_q4k_block_cpu(&block);
    let shader_result = dequantize_q4k_block_shader(&block);

    let mut max_diff = 0.0f32;
    let mut mismatch_count = 0;

    for i in 0..QK_K {
        let diff = (cpu_result[i] - shader_result[i]).abs();
        if diff > 1e-5 {
            mismatch_count += 1;
            if mismatch_count <= 10 {
                println!(
                    "Mismatch at {}: CPU={:.6}, Shader={:.6}, diff={:.6}",
                    i, cpu_result[i], shader_result[i], diff
                );
            }
        }
        max_diff = max_diff.max(diff);
    }

    println!("\nMax diff: {:.6}", max_diff);
    println!("Mismatch count: {}/{}", mismatch_count, QK_K);

    if mismatch_count == 0 {
        println!("\n✅ CPU and shader dequantization match!");
    } else {
        println!(
            "\n❌ Found {} mismatches between CPU and shader dequantization",
            mismatch_count
        );
    }
}

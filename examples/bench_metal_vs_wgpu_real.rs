//! Real benchmark comparing Metal backend vs WebGPU for Q4K matmul.
//!
//! Run with: cargo run --example bench_metal_vs_wgpu_real --features metal-backend --release

use std::time::Instant;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
fn main() {
    use web_rwkv::metal::{MetalContext, MetalOps};

    println!("=== Metal Q4K Matmul Performance Analysis ===\n");

    let metal_ctx = MetalContext::new().expect("Failed to initialize Metal context");
    println!("Metal context initialized\n");

    // Test dimensions (2.9B model)
    let test_cases = [
        (2560, 2560, "Embedding/Attention"),
        (2560, 10240, "FFN up projection"),
        (10240, 2560, "FFN down projection"),
    ];

    println!("=== Individual Matmul (sync per op) ===");
    for (k, m, desc) in test_cases {
        let k = (k / 256) * 256;
        let num_blocks = k / 256;
        let weights_size = m * num_blocks * 144;

        let weights_data: Vec<u8> = (0..weights_size)
            .map(|i| ((i * 17 + 31) % 256) as u8)
            .collect();
        let input_data: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

        let metal_weights = metal_ctx.create_buffer(&weights_data);
        let metal_input = metal_ctx.create_buffer(bytemuck::cast_slice(&input_data));
        let metal_output = metal_ctx.create_buffer_empty((m * 4) as u64);

        // Warmup
        for _ in 0..10 {
            MetalOps::matmul_vec_q4k(
                &metal_ctx,
                &metal_weights,
                &metal_input,
                &metal_output,
                k as u32,
                m as u32,
            );
        }

        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            MetalOps::matmul_vec_q4k(
                &metal_ctx,
                &metal_weights,
                &metal_input,
                &metal_output,
                k as u32,
                m as u32,
            );
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * k as f64 * m as f64) / (avg_us * 1e-6) / 1e9;
        println!(
            "{}: {:>8.1} µs | {:>6.1} GFLOPS (sync)",
            desc, avg_us, gflops
        );
    }

    println!("\n=== Batched Matmuls (single sync at end) ===");

    // Simulate a full forward pass: batch multiple matmuls into one command buffer
    let k = 2560usize;
    let m = 2560usize;
    let num_blocks = k / 256;
    let weights_size = m * num_blocks * 144;

    let weights_data: Vec<u8> = (0..weights_size)
        .map(|i| ((i * 17 + 31) % 256) as u8)
        .collect();
    let input_data: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

    let metal_weights = metal_ctx.create_buffer(&weights_data);
    let metal_input = metal_ctx.create_buffer(bytemuck::cast_slice(&input_data));
    let metal_output = metal_ctx.create_buffer_empty((m * 4) as u64);

    // Simulate 192 matmuls per token (32 layers * 6 matmuls)
    let matmuls_per_token = 192;
    let tokens = 10;
    let total_matmuls = matmuls_per_token * tokens;

    // Warmup
    for _ in 0..5 {
        let cmd = metal_ctx.queue().new_command_buffer();
        for _ in 0..matmuls_per_token {
            MetalOps::encode_matmul_vec_q4k(
                &metal_ctx,
                &cmd,
                &metal_weights,
                &metal_input,
                &metal_output,
                k as u32,
                m as u32,
            );
        }
        cmd.commit();
        cmd.wait_until_completed();
    }

    let start = Instant::now();
    for _ in 0..tokens {
        let cmd = metal_ctx.queue().new_command_buffer();
        for _ in 0..matmuls_per_token {
            MetalOps::encode_matmul_vec_q4k(
                &metal_ctx,
                &cmd,
                &metal_weights,
                &metal_input,
                &metal_output,
                k as u32,
                m as u32,
            );
        }
        cmd.commit();
        cmd.wait_until_completed();
    }
    let elapsed = start.elapsed();

    let time_per_token_ms = elapsed.as_secs_f64() * 1000.0 / tokens as f64;
    let tokens_per_sec = tokens as f64 / elapsed.as_secs_f64();
    let avg_matmul_us = elapsed.as_micros() as f64 / total_matmuls as f64;
    let gflops = (2.0 * k as f64 * m as f64) / (avg_matmul_us * 1e-6) / 1e9;

    println!("Batched {} matmuls/token:", matmuls_per_token);
    println!(
        "  Average matmul: {:>8.1} µs | {:>6.1} GFLOPS",
        avg_matmul_us, gflops
    );
    println!("  Time per token: {:>8.2} ms", time_per_token_ms);
    println!("  Throughput:     {:>8.1} tok/s", tokens_per_sec);

    println!("\n--- Comparison ---");
    println!("WebGPU baseline: 44.2 tok/s");
    println!("Metal batched:   {:>5.1} tok/s", tokens_per_sec);

    if tokens_per_sec > 44.2 {
        println!(
            "\n✓ Metal is {:.2}x faster than WebGPU!",
            tokens_per_sec / 44.2
        );
    } else {
        println!(
            "\n✗ Metal is {:.2}x slower than WebGPU",
            44.2 / tokens_per_sec
        );
    }
}

#[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
fn main() {
    println!("This example requires macOS and the 'metal-backend' feature.");
}

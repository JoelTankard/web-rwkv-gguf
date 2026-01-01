//! Benchmark comparing Metal backend vs WebGPU for Q4K matmul.
//!
//! Run with: cargo run --example bench_metal_vs_wgpu --features metal-backend --release

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
fn main() -> anyhow::Result<()> {
    use std::time::Instant;
    use web_rwkv::metal::{MetalContext, MetalOps};

    println!("=== Metal vs WebGPU Q4K Matmul Benchmark ===\n");

    // Model-realistic dimensions (2.9B model has embedding dim 2560)
    let test_cases = [
        (2560, 2560, "2.9B embedding layer"),
        (2560, 10240, "2.9B FFN up projection"),
        (10240, 2560, "2.9B FFN down projection"),
    ];

    // Initialize Metal context
    let ctx = MetalContext::new().expect("Failed to initialize Metal context");
    println!("Metal device initialized\n");

    for (k, m, desc) in test_cases {
        println!("--- {} (K={}, M={}) ---", desc, k, m);

        // K must be multiple of 256 for Q4_K
        let k = (k / 256) * 256;
        if k == 0 {
            println!("  Skipping: K must be >= 256");
            continue;
        }

        // Q4_K block: 144 bytes per 256 elements
        let num_blocks = k / 256;
        let block_size = 144;
        let weights_size = m * num_blocks * block_size;

        // Create test data with realistic random values
        let weights_data: Vec<u8> = (0..weights_size)
            .map(|i| ((i * 17 + 31) % 256) as u8)
            .collect();
        let input_data: Vec<f32> = (0..k).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
        let output_size = m * std::mem::size_of::<f32>();

        // Create Metal buffers
        let weights_buffer = ctx.create_buffer(&weights_data);
        let input_buffer = ctx.create_buffer(bytemuck::cast_slice(&input_data));
        let output_buffer = ctx.create_buffer_empty(output_size as u64);

        // Warmup
        for _ in 0..5 {
            MetalOps::matmul_vec_q4k(
                &ctx,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                k as u32,
                m as u32,
            );
        }

        // Benchmark Metal
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            MetalOps::matmul_vec_q4k(
                &ctx,
                &weights_buffer,
                &input_buffer,
                &output_buffer,
                k as u32,
                m as u32,
            );
        }
        let metal_elapsed = start.elapsed();
        let metal_avg_us = metal_elapsed.as_micros() as f64 / iterations as f64;

        // Calculate throughput
        // For matmul: 2 * K * M FLOPs (multiply-add)
        let flops = 2.0 * k as f64 * m as f64;
        let metal_gflops = (flops * iterations as f64) / metal_elapsed.as_secs_f64() / 1e9;

        // Memory bandwidth: read K*M weights + K input, write M output
        // Q4_K: 0.5625 bytes per weight element
        let bytes_per_iter = (k * m) as f64 * 0.5625 + k as f64 * 4.0 + m as f64 * 4.0;
        let metal_gbps = (bytes_per_iter * iterations as f64) / metal_elapsed.as_secs_f64() / 1e9;

        println!(
            "  Metal:  {:>8.1} µs/iter | {:>6.2} GFLOPS | {:>6.2} GB/s",
            metal_avg_us, metal_gflops, metal_gbps
        );

        // Estimate WebGPU performance based on baseline benchmark
        // Baseline: 44.2 tok/s for 2.9B model
        // Each token requires ~32 layers * 4 matmuls = 128 matmuls
        // So ~5600 matmuls/sec, or ~178 µs per matmul average
        let wgpu_estimated_us = 178.0; // rough estimate
        println!(
            "  WebGPU: {:>8.1} µs/iter (estimated from baseline)",
            wgpu_estimated_us
        );

        let speedup = wgpu_estimated_us / metal_avg_us;
        if speedup > 1.0 {
            println!("  Metal is {:.2}x faster", speedup);
        } else {
            println!("  Metal is {:.2}x slower", 1.0 / speedup);
        }
        println!();
    }

    println!("=== Benchmark Complete ===");
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
fn main() {
    println!("This example requires macOS and the 'metal-backend' feature.");
    println!(
        "Run with: cargo run --example bench_metal_vs_wgpu --features metal-backend --release"
    );
}

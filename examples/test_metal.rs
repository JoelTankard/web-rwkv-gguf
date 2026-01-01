//! Test the Metal backend for Q4K matrix multiplication.
//!
//! Run with: cargo run --example test_metal --features metal-backend

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
fn main() {
    use web_rwkv::metal::{MetalContext, MetalOps};

    println!("Testing Metal backend for Q4K matmul...\n");

    // Initialize Metal context
    let ctx = match MetalContext::new() {
        Some(ctx) => {
            println!("✓ Metal context initialized successfully");
            ctx
        }
        None => {
            println!("✗ Failed to initialize Metal context");
            return;
        }
    };

    // Test parameters
    let k: u32 = 256; // Must be multiple of 256 (Q4_K block size)
    let m: u32 = 64; // Output dimension

    // Create test data
    // Q4_K block: 144 bytes per 256 elements
    // For m rows, we need m * (k / 256) * 144 bytes
    let num_blocks = k / 256;
    let block_size = 144;
    let weights_size = (m * num_blocks * block_size) as usize;

    // Create dummy Q4_K weights (in practice, these would come from the model)
    let weights_data: Vec<u8> = (0..weights_size).map(|i| (i % 256) as u8).collect();

    // Create input vector
    let input_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();

    // Create output buffer
    let output_size = (m * std::mem::size_of::<f32>() as u32) as usize;

    // Create Metal buffers
    let weights_buffer = ctx.create_buffer(&weights_data);
    let input_buffer = ctx.create_buffer(bytemuck::cast_slice(&input_data));
    let output_buffer = ctx.create_buffer_empty(output_size as u64);

    println!("✓ Created Metal buffers");
    println!("  - Weights: {} bytes", weights_size);
    println!("  - Input: {} floats", k);
    println!("  - Output: {} floats", m);

    // Run matmul
    println!("\nRunning matmul_vec_q4k...");
    let start = std::time::Instant::now();

    MetalOps::matmul_vec_q4k(&ctx, &weights_buffer, &input_buffer, &output_buffer, k, m);

    let elapsed = start.elapsed();
    println!("✓ Completed in {:?}", elapsed);

    // Read back results
    let output_ptr = output_buffer.contents() as *const f32;
    let output: Vec<f32> = unsafe { std::slice::from_raw_parts(output_ptr, m as usize).to_vec() };

    println!("\nFirst 8 output values:");
    for (i, val) in output.iter().take(8).enumerate() {
        println!("  output[{}] = {:.6}", i, val);
    }

    // Benchmark with multiple iterations
    println!("\nBenchmarking (100 iterations)...");
    let start = std::time::Instant::now();
    for _ in 0..100 {
        MetalOps::matmul_vec_q4k(&ctx, &weights_buffer, &input_buffer, &output_buffer, k, m);
    }
    let elapsed = start.elapsed();
    println!("✓ 100 iterations in {:?}", elapsed);
    println!("  Average: {:?} per iteration", elapsed / 100);

    println!("\n✓ Metal backend test completed successfully!");
}

#[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
fn main() {
    println!("This example requires macOS and the 'metal-backend' feature.");
    println!("Run with: cargo run --example test_metal --features metal-backend");
}

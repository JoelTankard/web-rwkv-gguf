# Option 8: Profiling and Measurement Guide

## Before Optimizing: Profile First

The current analysis is based on theory. We need data to confirm:

1. Is the bottleneck memory or compute?
2. Which operations take the most time?
3. Where is the GPU idle?

## Profiling Tools

### 1. GPU Timestamps (wgpu)

Add timestamp queries to measure kernel execution time:

```rust
use wgpu::Features;

// Enable timestamp queries
let features = Features::TIMESTAMP_QUERY | Features::TIMESTAMP_QUERY_INSIDE_PASSES;

// Create query set
let query_set = device.create_query_set(&QuerySetDescriptor {
    ty: QueryType::Timestamp,
    count: 2,
});

// In compute pass
compute_pass.write_timestamp(&query_set, 0);
// ... dispatch ...
compute_pass.write_timestamp(&query_set, 1);

// Resolve and read
encoder.resolve_query_set(&query_set, 0..2, &resolve_buffer, 0);
```

### 2. Metal GPU Profiler (macOS)

Use Xcode Instruments with Metal System Trace:

```bash
# Run with Metal validation
METAL_DEVICE_WRAPPER_TYPE=1 cargo run --release --example benchmark
```

Then profile with Instruments:

1. Open Instruments
2. Select "Metal System Trace"
3. Attach to process
4. Record and analyze

### 3. Custom Timing

Add timing around key operations:

```rust
use std::time::Instant;

let start = Instant::now();
context.queue.submit(commands);
device.poll(wgpu::Maintain::Wait);  // Force sync
let elapsed = start.elapsed();
println!("MatMul took: {:?}", elapsed);
```

## Key Metrics to Measure

### 1. Per-Layer Breakdown

```rust
struct LayerTiming {
    attention_linear: Duration,
    wkv: Duration,
    attention_output: Duration,
    ffn: Duration,
    total: Duration,
}
```

### 2. Memory Bandwidth Utilization

```rust
fn measure_bandwidth(bytes_read: usize, bytes_written: usize, duration: Duration) -> f64 {
    let total_bytes = bytes_read + bytes_written;
    let seconds = duration.as_secs_f64();
    (total_bytes as f64) / seconds / 1e9  // GB/s
}
```

### 3. Compute Utilization

Compare actual FLOPS to theoretical peak:

```rust
fn compute_utilization(flops: usize, duration: Duration, peak_tflops: f64) -> f64 {
    let actual_tflops = (flops as f64) / duration.as_secs_f64() / 1e12;
    actual_tflops / peak_tflops * 100.0  // Percentage
}
```

## Benchmark Experiments

### Experiment 1: Memory vs Compute

```rust
// Test A: Read-only (memory bound)
fn benchmark_read_only() {
    // Shader that just reads all weights and sums them
    // No actual matmul computation
}

// Test B: Compute-only (compute bound)
fn benchmark_compute_only() {
    // Shader that does matmul on data already in shared memory
    // No global memory reads
}

// Test C: Full matmul (current)
fn benchmark_full() {
    // Normal matmul
}
```

If A >> C: Memory-bound, optimize memory access
If B >> C: Compute-bound, optimize ALU usage
If A + B ≈ C: Balanced, optimize both

### Experiment 2: Quantization Overhead

```rust
// Test A: F16 matmul (no dequant)
fn benchmark_f16_matmul() {
    // Standard F16 weights
}

// Test B: Q4K matmul (with dequant)
fn benchmark_q4k_matmul() {
    // Q4K weights, inline dequant
}

// Overhead = (B - A) / A * 100%
```

### Experiment 3: Workgroup Size Sweep

```rust
for block_size in [32, 64, 128, 256, 512] {
    let timing = benchmark_with_block_size(block_size);
    println!("Block size {}: {:?}", block_size, timing);
}
```

### Experiment 4: Tile Size Sweep

```rust
for tile_size in [16, 32, 64, 128] {
    let timing = benchmark_with_tile_size(tile_size);
    println!("Tile size {}: {:?}", tile_size, timing);
}
```

## Expected Results

Based on M2 Max specs:

-   Memory bandwidth: 400 GB/s
-   FP32 compute: 13.6 TFLOPS
-   FP16 compute: 27.2 TFLOPS (theoretical)

For Q4K matmul (2560x2560, batch=1):

-   Weight reads: 2560 _ 2560 _ 0.5625 bytes = 3.7 MB
-   Compute: 2560 _ 2560 _ 2 FLOPs = 13.1 MFLOP

At peak bandwidth: 3.7 MB / 400 GB/s = 0.009 ms
At peak compute: 13.1 MFLOP / 13.6 TFLOP/s = 0.001 ms

**Theoretical minimum: ~0.01 ms per matmul**

Current: ~0.5 ms per matmul (estimated from 44 tok/s)

**We're ~50x slower than theoretical!**

This confirms significant optimization opportunity.

## Action Items

1. **Add timestamp profiling** to benchmark example
2. **Run memory vs compute experiment**
3. **Profile with Metal Instruments**
4. **Identify top 3 bottleneck operations**
5. **Target those for optimization**

## Sample Profiling Code

```rust
// Add to examples/benchmark.rs
fn profile_layer_timing(model: &Model, input: &TensorGpu<f16>) -> LayerTiming {
    let device = &model.context.device;
    let queue = &model.context.queue;

    // Create timestamp query set
    let query_set = device.create_query_set(&QuerySetDescriptor {
        label: Some("layer_timing"),
        ty: QueryType::Timestamp,
        count: 10,
    });

    let resolve_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("timestamp_resolve"),
        size: 10 * 8,  // 8 bytes per timestamp
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // ... run operations with timestamps ...

    // Read back and compute durations
    // ...
}
```

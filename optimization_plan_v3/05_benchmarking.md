# Benchmarking Guide

## Using benchmark.sh

The `benchmark.sh` script is the primary tool for measuring performance changes.

### Basic Usage

```bash
# Interactive mode - prompts for title and change description
./benchmark.sh

# With all options specified
./benchmark.sh -t "Metal Integration" -c "Added Metal backend for Q4K matmul" -f metal_perf

# Use a different model
./benchmark.sh -m assets/models/0.1b-Q4_K_M.gguf -t "Small Model Test" -c "Testing"
```

### Options

| Option         | Description                   | Default                          |
| -------------- | ----------------------------- | -------------------------------- |
| `-m, --model`  | Path to GGUF model file       | `assets/models/2.9b-Q4_K_M.gguf` |
| `-f, --file`   | Output filename (without .md) | `benchmarks`                     |
| `-t, --title`  | Benchmark title               | (prompted)                       |
| `-c, --change` | Change description            | (prompted)                       |
| `-h, --help`   | Show help                     | -                                |

### Output

Results are saved to `benchmarks/<filename>.md` with:

-   Load time (ms)
-   Prefill speed (tok/s) - processing input tokens
-   Generation speed (tok/s) - generating new tokens
-   Quality hash - deterministic hash for regression detection

## Benchmark Workflow

### Before Making Changes

```bash
# Record baseline
./benchmark.sh -t "Baseline" -c "Before optimization" -f my_optimization
```

### After Making Changes

```bash
# Record new performance
./benchmark.sh -t "After Metal" -c "Metal backend enabled" -f my_optimization
```

### Compare Results

Open `benchmarks/my_optimization.md` to see all runs side by side.

## What to Measure

### Key Metrics

| Metric           | What It Measures  | Target              |
| ---------------- | ----------------- | ------------------- |
| Generation speed | Token output rate | 90+ tok/s           |
| Prefill speed    | Input processing  | 150+ tok/s          |
| Load time        | Model loading     | <5s                 |
| Quality hash     | Correctness       | Must match baseline |

### Quality Hash

The quality hash ensures numerical correctness:

-   Same hash = identical outputs
-   Different hash = something changed numerically

If the hash changes after optimization:

1. Check for floating point precision issues
2. Verify dequantization logic matches
3. May be acceptable if within tolerance

## Metal-Specific Benchmarking

### Test Metal Backend Directly

```bash
# Run Metal kernel benchmark
cargo run --release --features metal-backend --example bench_metal_vs_wgpu_real
```

This shows:

-   Individual matmul timing (sync per op)
-   Batched matmul timing (realistic)
-   Comparison with WebGPU baseline

### Test With/Without Metal

```bash
# With Metal backend
cargo run --release --features metal-backend --example benchmark -- \
    -t "With Metal" -c "Metal enabled" --model assets/models/2.9b-Q4_K_M.gguf

# Without Metal backend (WebGPU only)
cargo run --release --example benchmark -- \
    -t "WebGPU Only" -c "No Metal" --model assets/models/2.9b-Q4_K_M.gguf
```

## Profiling

### GPU Profiling on macOS

Use Xcode Instruments for detailed GPU profiling:

```bash
# Build with debug symbols
cargo build --release --features metal-backend

# Profile with Instruments
xcrun xctrace record --template "Metal System Trace" \
    --launch target/release/examples/benchmark -- \
    --model assets/models/2.9b-Q4_K_M.gguf \
    --title "Profile" --change "Profiling"
```

### CPU Profiling

```bash
# Using samply (install: cargo install samply)
samply record cargo run --release --example benchmark -- \
    --model assets/models/2.9b-Q4_K_M.gguf \
    --title "CPU Profile" --change "Profiling"
```

## Regression Testing

### Automated Benchmark Script

Create a CI-friendly benchmark:

```bash
#!/bin/bash
# ci_benchmark.sh

BASELINE_GEN=44.0  # tok/s
BASELINE_PREFILL=150.0

# Run benchmark and capture output
OUTPUT=$(cargo run --release --example benchmark -- \
    --model assets/models/2.9b-Q4_K_M.gguf \
    --title "CI" --change "CI run" 2>&1)

# Extract generation speed
GEN_SPEED=$(echo "$OUTPUT" | grep "Generation:" | awk '{print $2}')

# Check for regression
if (( $(echo "$GEN_SPEED < $BASELINE_GEN * 0.95" | bc -l) )); then
    echo "REGRESSION: Generation speed $GEN_SPEED < $BASELINE_GEN"
    exit 1
fi

echo "PASS: Generation speed $GEN_SPEED tok/s"
```

## Benchmark History

Keep a log of significant benchmarks:

| Date       | Change                 | Gen (tok/s) | Prefill (tok/s) | Notes               |
| ---------- | ---------------------- | ----------- | --------------- | ------------------- |
| 2024-01-01 | Baseline               | 44.2        | 152.3           | WebGPU only         |
| 2024-01-01 | Metal kernel (batched) | 211.1       | -               | Synthetic benchmark |
| TBD        | Metal integrated       | TBD         | TBD             | Full integration    |

//! Benchmark comparing different Q4K matmul shader implementations.
//!
//! Usage:
//!   cargo run --release --example bench_q4k_shaders -- \
//!     --gguf /path/to/model.gguf

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use instant::{Duration, Instant};
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        loader::Loader,
        model::{ContextAutoLimits, ModelInfo},
    },
    tensor::{
        kind::ReadWrite,
        ops::{Activation, TensorOp},
        TensorGpu,
    },
};

const WARMUP_RUNS: usize = 10;
const BENCH_RUNS: usize = 100;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    gguf: PathBuf,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(long, default_value_t = 2560)]
    k: usize,
    #[arg(long, default_value_t = 2560)]
    m: usize,
}

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = wgpu::Instance::default();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
        let backends = wgpu::Backends::all();
        let adapters = instance.enumerate_adapters(backends);
        let names = adapters
            .iter()
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&names)
            .interact()?;
        adapters.into_iter().nth(selection).unwrap()
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

struct BenchResult {
    name: String,
    avg_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
    throughput_gflops: f64,
}

fn print_results(results: &[BenchResult], k: usize, m: usize) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║                         Q4K Shader Comparison Benchmark                              ║"
    );
    println!(
        "╠═══════════════════════════════════════════════════════════════════════════════════════╣"
    );
    println!(
        "║  Matrix: K={:<6} M={:<6}  (vec-mat multiply, single token)                         ║",
        k, m
    );
    println!(
        "╠═══════════════════════════════════╦═══════════════╦═══════════════╦═════════════════╣"
    );
    println!(
        "║             Shader                ║   Avg (μs)    ║  Min/Max (μs) ║   GFLOP/s       ║"
    );
    println!(
        "╠═══════════════════════════════════╬═══════════════╬═══════════════╬═════════════════╣"
    );

    for result in results {
        println!(
            "║ {:>33} ║ {:>13.1} ║ {:>5.0}/{:<7.0} ║ {:>15.1} ║",
            result.name,
            result.avg_time_us,
            result.min_time_us,
            result.max_time_us,
            result.throughput_gflops
        );
    }

    println!(
        "╚═══════════════════════════════════╩═══════════════╩═══════════════╩═════════════════╝"
    );

    if results.len() >= 2 {
        println!("\n📊 Relative Performance (vs first shader):");
        let baseline = results[0].avg_time_us;
        for result in results.iter().skip(1) {
            let speedup = baseline / result.avg_time_us;
            let indicator = if speedup > 1.0 { "faster" } else { "slower" };
            println!(
                "   {}: {:.2}x {} ({:.1}μs vs {:.1}μs)",
                result.name,
                speedup.max(1.0 / speedup),
                indicator,
                result.avg_time_us,
                baseline
            );
        }
    }
}

async fn bench_shader(
    context: &Context,
    matrix: &TensorGpu<u8, ReadWrite>,
    matrix_shape: &TensorGpu<u8, ReadWrite>,
    input: &TensorGpu<f16, ReadWrite>,
    output: &TensorGpu<f16, ReadWrite>,
    shader_name: &str,
    k: usize,
    m: usize,
) -> Result<BenchResult> {
    let op = match shader_name {
        "matmul_vec_q4k (original)" => {
            TensorOp::matmul_vec_q4k(matrix, matrix_shape, input, output, Activation::None)?
        }
        "matmul_vec_q4k_opt (v2/cached)" => {
            TensorOp::matmul_vec_q4k_opt(matrix, matrix_shape, input, output, Activation::None)?
        }
        _ => unreachable!(),
    };

    // Warmup
    for _ in 0..WARMUP_RUNS {
        context.queue.submit(context.encode(&op));
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        let start = Instant::now();
        context.queue.submit(context.encode(&op));
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let avg_time_us = total.as_secs_f64() * 1_000_000.0 / BENCH_RUNS as f64;
    let min_time_us = times.iter().min().unwrap().as_secs_f64() * 1_000_000.0;
    let max_time_us = times.iter().max().unwrap().as_secs_f64() * 1_000_000.0;

    // FLOPS: 2 * K * M (multiply-add per element)
    let flops = 2.0 * k as f64 * m as f64;
    let throughput_gflops = flops / (avg_time_us * 1000.0); // GFLOP/s

    Ok(BenchResult {
        name: shader_name.to_string(),
        avg_time_us,
        min_time_us,
        max_time_us,
        throughput_gflops,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();

    // Load GGUF to get model info for context creation
    let file = File::open(&cli.gguf).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data).expect("failed to parse GGUF file");
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?} ({})", info.version, cli.gguf.display());
    println!(
        "   Layers: {}, Embed: {}, Vocab: {}",
        info.num_layer, info.num_emb, info.num_vocab
    );

    let context = create_context(&info, cli.adapter).await?;
    println!("🖥️  Adapter: {}", context.adapter.get_info().name);

    // Use model dimensions or CLI overrides
    let k = cli.k;
    let m = cli.m;

    // Ensure K is multiple of 256 (Q4K block size) and M is multiple of 4
    let k = (k / 256) * 256;
    let m = (m / 4) * 4;

    println!("\n🔧 Creating test tensors: K={}, M={}", k, m);

    // Create Q4K matrix data
    // Q4K: 144 bytes per 256 elements = 0.5625 bytes per element
    let num_blocks = (k / 256) * m;
    let q4k_bytes = num_blocks * 144;

    // Create random Q4K data (for benchmarking, actual values don't matter)
    let q4k_data: Vec<u8> = (0..q4k_bytes).map(|i| (i % 256) as u8).collect();
    let matrix: TensorGpu<u8, ReadWrite> =
        context.tensor_from_data([q4k_bytes, 1, 1, 1], q4k_data)?;

    // Matrix shape tensor (stores logical shape [K, M, 1])
    let matrix_shape: TensorGpu<u8, ReadWrite> = context.tensor_init([k, m, 1, 1]);

    // Input tensor [K, 1, 1, 1] - single token
    let input_data: Vec<f16> = (0..k)
        .map(|i| f16::from_f32((i as f32 * 0.001).sin()))
        .collect();
    let input: TensorGpu<f16, ReadWrite> = context.tensor_from_data([k, 1, 1, 1], input_data)?;

    // Output tensor [M, 1, 1, 1]
    let output: TensorGpu<f16, ReadWrite> = context.tensor_init([m, 1, 1, 1]);

    println!(
        "🏃 Running benchmarks ({} warmup, {} runs each)...\n",
        WARMUP_RUNS, BENCH_RUNS
    );

    let mut results = Vec::new();

    // Benchmark original shader
    let result = bench_shader(
        &context,
        &matrix,
        &matrix_shape,
        &input,
        &output,
        "matmul_vec_q4k (original)",
        k,
        m,
    )
    .await?;
    println!("   ✓ {} - {:.1}μs avg", result.name, result.avg_time_us);
    results.push(result);

    // Benchmark v2/opt shader
    let result = bench_shader(
        &context,
        &matrix,
        &matrix_shape,
        &input,
        &output,
        "matmul_vec_q4k_opt (v2/cached)",
        k,
        m,
    )
    .await?;
    println!("   ✓ {} - {:.1}μs avg", result.name, result.avg_time_us);
    results.push(result);

    print_results(&results, k, m);

    Ok(())
}

//! Benchmark comparing SafeTensors (.st) vs GGUF (.gguf) model loading and inference.
//!
//! Usage:
//!   cargo run --release --example bench_format -- \
//!     --st /path/to/model.st \
//!     --gguf /path/to/model.gguf

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use instant::{Duration, Instant};
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use sysinfo::{Pid, System};
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
};

const WARMUP_RUNS: usize = 2;
const BENCH_RUNS: usize = 5;

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

fn sample(probs: &[f32]) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    st: Option<PathBuf>,
    #[arg(long, value_name = "FILE")]
    gguf: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(long, default_value_t = 256)]
    prefill_length: usize,
    #[arg(long, default_value_t = 64)]
    generation_length: usize,
}

struct BenchResult {
    format: String,
    file_size_mb: f64,
    load_time_ms: f64,
    peak_ram_mb: f64,
    prefill_tps: f64,
    generation_tps: f64,
}

fn get_process_memory_mb() -> f64 {
    let mut sys = System::new();
    let pid = Pid::from_u32(std::process::id());
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    if let Some(process) = sys.process(pid) {
        process.memory() as f64 / 1024.0 / 1024.0
    } else {
        0.0
    }
}

fn get_file_size_mb(path: &PathBuf) -> f64 {
    std::fs::metadata(path)
        .map(|m| m.len() as f64 / 1024.0 / 1024.0)
        .unwrap_or(0.0)
}

async fn bench_safetensors(
    path: &PathBuf,
    quant: HashMap<usize, Quant>,
    token_chunk_size: usize,
    prefill_length: usize,
    generation_length: usize,
    auto_adapter: bool,
) -> Result<BenchResult> {
    let file_size_mb = get_file_size_mb(path);
    let ram_before = get_process_memory_mb();
    let load_start = Instant::now();

    let file = File::open(path).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    let context = create_context(&info, auto_adapter).await?;

    let builder = ModelBuilder::new(&context, model).quant(quant);

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let ram_after = get_process_memory_mb();
    let peak_ram_mb = ram_after - ram_before;

    let (prefill_tps, generation_tps) = run_inference_bench(
        &runtime,
        &context,
        &info,
        token_chunk_size,
        prefill_length,
        generation_length,
    )
    .await?;

    Ok(BenchResult {
        format: "SafeTensors".to_string(),
        file_size_mb,
        load_time_ms,
        peak_ram_mb,
        prefill_tps,
        generation_tps,
    })
}

async fn bench_gguf(
    path: &PathBuf,
    quant: HashMap<usize, Quant>,
    token_chunk_size: usize,
    prefill_length: usize,
    generation_length: usize,
    auto_adapter: bool,
) -> Result<BenchResult> {
    let file_size_mb = get_file_size_mb(path);
    let ram_before = get_process_memory_mb();
    let load_start = Instant::now();

    let file = File::open(path).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data).expect("failed to parse GGUF file");
    let info = Loader::info(&model)?;
    let context = create_context(&info, auto_adapter).await?;

    let builder = ModelBuilder::new(&context, model).quant(quant);

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
    };

    let load_time_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let ram_after = get_process_memory_mb();
    let peak_ram_mb = ram_after - ram_before;

    let (prefill_tps, generation_tps) = run_inference_bench(
        &runtime,
        &context,
        &info,
        token_chunk_size,
        prefill_length,
        generation_length,
    )
    .await?;

    Ok(BenchResult {
        format: "GGUF".to_string(),
        file_size_mb,
        load_time_ms,
        peak_ram_mb,
        prefill_tps,
        generation_tps,
    })
}

async fn run_inference_bench(
    runtime: &Box<dyn Runtime<Rnn>>,
    context: &Context,
    info: &ModelInfo,
    token_chunk_size: usize,
    prefill_length: usize,
    generation_length: usize,
) -> Result<(f64, f64)> {
    fastrand::seed(42);

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let tokens: Vec<u16> = (0..16)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], token_chunk_size);
        for _ in 0..16 {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;
            if output[0].0.size() > 0 {
                break;
            }
        }
    }

    // Prefill benchmark
    let mut prefill_time = Duration::ZERO;
    for _ in 0..BENCH_RUNS {
        let tokens: Vec<u16> = (0..prefill_length)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], token_chunk_size);
        let instant = Instant::now();
        for _ in 0..prefill_length {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;
            if output[0].0.size() > 0 {
                prefill_time += instant.elapsed();
                break;
            }
        }
    }
    let prefill_tps = prefill_length as f64 / prefill_time.as_secs_f64() * BENCH_RUNS as f64;

    // Generation benchmark
    let mut generation_time = Duration::ZERO;
    for _ in 0..BENCH_RUNS {
        let tokens: Vec<u16> = vec![0];
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], token_chunk_size);
        let instant = Instant::now();
        for _ in 0..generation_length {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(context, output).await?;
                let output = output.to_vec();
                let token = sample(&output);
                prompt.batches[0].push(token);
            }
        }
        generation_time += instant.elapsed();
    }
    let generation_tps =
        generation_length as f64 / generation_time.as_secs_f64() * BENCH_RUNS as f64;

    Ok((prefill_tps, generation_tps))
}

fn print_results(results: &[BenchResult], prefill_len: usize, gen_len: usize) {
    println!("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              Format Comparison Benchmark                                        ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Prefill length: {:>4} tokens    Generation length: {:>4} tokens                                   ║",
        prefill_len, gen_len
    );
    println!("╠══════════════╦═══════════╦═══════════════╦═══════════════╦═══════════════╦═══════════════════════╣");
    println!("║    Format    ║ File (MB) ║  Load (ms)    ║  RAM Δ (MB)   ║  Prefill t/s  ║  Generation t/s       ║");
    println!("╠══════════════╬═══════════╬═══════════════╬═══════════════╬═══════════════╬═══════════════════════╣");

    for result in results {
        println!(
            "║ {:>12} ║ {:>9.1} ║ {:>13.1} ║ {:>13.1} ║ {:>13.1} ║ {:>21.1} ║",
            result.format,
            result.file_size_mb,
            result.load_time_ms,
            result.peak_ram_mb,
            result.prefill_tps,
            result.generation_tps
        );
    }

    println!("╚══════════════╩═══════════╩═══════════════╩═══════════════╩═══════════════╩═══════════════════════╝");

    if results.len() == 2 {
        let st = &results[0];
        let gguf = &results[1];

        println!("\n📊 Comparison (GGUF vs SafeTensors):");
        let file_diff = (gguf.file_size_mb - st.file_size_mb) / st.file_size_mb * 100.0;
        let load_diff = (gguf.load_time_ms - st.load_time_ms) / st.load_time_ms * 100.0;
        let ram_diff = if st.peak_ram_mb > 0.0 {
            (gguf.peak_ram_mb - st.peak_ram_mb) / st.peak_ram_mb * 100.0
        } else {
            0.0
        };
        let prefill_diff = (gguf.prefill_tps - st.prefill_tps) / st.prefill_tps * 100.0;
        let gen_diff = (gguf.generation_tps - st.generation_tps) / st.generation_tps * 100.0;

        println!(
            "   File size:   {:>+.1}% ({})",
            file_diff,
            if file_diff < 0.0 { "smaller" } else { "larger" }
        );
        println!(
            "   Load time:   {:>+.1}% ({})",
            load_diff,
            if load_diff < 0.0 { "faster" } else { "slower" }
        );
        println!(
            "   RAM usage:   {:>+.1}% ({})",
            ram_diff,
            if ram_diff < 0.0 { "less" } else { "more" }
        );
        println!(
            "   Prefill:     {:>+.1}% ({})",
            prefill_diff,
            if prefill_diff > 0.0 {
                "faster"
            } else {
                "slower"
            }
        );
        println!(
            "   Generation:  {:>+.1}% ({})",
            gen_diff,
            if gen_diff > 0.0 { "faster" } else { "slower" }
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("bench_format", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();

    if cli.st.is_none() && cli.gguf.is_none() {
        eprintln!("Error: At least one of --st or --gguf must be provided");
        std::process::exit(1);
    }

    let quant: HashMap<usize, Quant> = (0..cli.quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
        .collect();

    let mut results = Vec::new();

    if let Some(ref st_path) = cli.st {
        println!("\n🔄 Benchmarking SafeTensors: {}", st_path.display());
        let result = bench_safetensors(
            st_path,
            quant.clone(),
            cli.token_chunk_size,
            cli.prefill_length,
            cli.generation_length,
            cli.adapter,
        )
        .await?;
        results.push(result);
    }

    if let Some(ref gguf_path) = cli.gguf {
        println!("\n🔄 Benchmarking GGUF: {}", gguf_path.display());
        let result = bench_gguf(
            gguf_path,
            quant.clone(),
            cli.token_chunk_size,
            cli.prefill_length,
            cli.generation_length,
            cli.adapter,
        )
        .await?;
        results.push(result);
    }

    print_results(&results, cli.prefill_length, cli.generation_length);

    Ok(())
}

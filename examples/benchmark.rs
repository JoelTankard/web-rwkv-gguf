//! Comprehensive benchmark tool for tracking performance across changes.
//!
//! This tool measures:
//! - Model loading time
//! - Inference speed (tok/s for prefill and generation)
//! - Output quality (deterministic generation for comparison)
//!
//! Results are written to a markdown file for tracking across changes.
//!
//! Usage:
//!   cargo run --release --example benchmark -- \
//!     --model /path/to/model.gguf \
//!     --title "Baseline" \
//!     --change "Initial measurement before optimization" \
//!     --output benchmarks.md

use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::PathBuf,
};

use anyhow::Result;
use clap::Parser;
use half::f16;
use instant::Instant;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use tokio::fs::File as TokioFile;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        softmax::softmax_one,
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{shape::Shape, TensorCpu, TensorInit},
};

const WARMUP_RUNS: usize = 2;
const BENCH_RUNS: usize = 3;
const PREFILL_LENGTH: usize = 128;
const GENERATION_LENGTH: usize = 64;
const TOKEN_CHUNK_SIZE: usize = 32;
const QUALITY_SEED: u64 = 12345;
const QUALITY_PROMPT_LENGTH: usize = 16;
const QUALITY_GEN_LENGTH: usize = 32;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Comprehensive benchmark for tracking performance"
)]
struct Args {
    /// Path to model file (.gguf)
    #[arg(
        short,
        long,
        default_value = "assets/models/rwkv7-g1b-2.9b-20251205-ctx8192-Q4_K_M.gguf"
    )]
    model: PathBuf,

    /// Title for this benchmark group (tests with same title are grouped)
    #[arg(short, long)]
    title: String,

    /// Description of what changed since last benchmark
    #[arg(short, long)]
    change: String,

    /// Output markdown file path
    #[arg(short, long, default_value = "optimization_plan/benchmarks.md")]
    output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    timestamp: String,
    title: String,
    change: String,
    model_name: String,
    model_version: String,
    num_layers: usize,
    num_emb: usize,
    gpu_name: String,
    load_time_ms: f64,
    prefill_tps: f64,
    generation_tps: f64,
    quality_hash: String,
    quality_tokens: Vec<u16>,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

fn sample_argmax(probs: &[f32]) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u16)
        .unwrap_or(0)
}

fn format_timestamp(secs: u64) -> String {
    // Simple timestamp formatting without external dependencies
    // Returns ISO-like format: YYYY-MM-DD HH:MM:SS
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Approximate date calculation (not accounting for leap years perfectly)
    let mut year = 1970;
    let mut remaining_days = days_since_epoch;

    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
            366
        } else {
            365
        };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let is_leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_months: [u64; 12] = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= *days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        year, month, day, hours, minutes, seconds
    )
}

async fn run_benchmark(args: &Args) -> Result<BenchmarkResult> {
    let model_name = args
        .model
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Comprehensive Benchmark Tool                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("📦 Model: {}", model_name);
    println!("📝 Title: {}", args.title);
    println!("🔄 Change: {}", args.change);
    println!();

    // Load and parse model info
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("   Version: {:?}", info.version);
    println!("   Layers: {}", info.num_layer);
    println!("   Embedding: {}", info.num_emb);
    println!("   Vocab: {}", info.num_vocab);
    println!();

    // Create context
    let context = create_context(&info).await?;
    let gpu_name = context.adapter.get_info().name.clone();
    println!("🖥️  GPU: {}", gpu_name);
    println!();

    // === LOADING BENCHMARK ===
    println!("⏳ Benchmarking model loading...");
    let mut load_times = Vec::with_capacity(BENCH_RUNS);

    for i in 0..BENCH_RUNS {
        // Re-open file for each run to avoid caching effects
        let file = TokioFile::open(&args.model).await?;
        let data = unsafe { Mmap::map(&file)? };
        let model = GgufReader::new(&data)?;

        let start = Instant::now();
        let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
        let _runtime: Box<dyn Runtime<Rnn>> = match info.version {
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
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        load_times.push(elapsed);
        println!("   Run {}: {:.2} ms", i + 1, elapsed);
    }

    let load_time_ms = load_times.iter().sum::<f64>() / load_times.len() as f64;
    println!("   Average: {:.2} ms", load_time_ms);
    println!();

    // === INFERENCE BENCHMARK ===
    println!("⏳ Benchmarking inference speed...");

    // Create runtime for inference benchmarks
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());

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

    // Warmup
    println!("   Warming up ({} runs)...", WARMUP_RUNS);
    fastrand::seed(42);
    for _ in 0..WARMUP_RUNS {
        let tokens: Vec<u16> = (0..16)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        for _ in 0..16 {
            let (new_input, output) = runtime.infer(input).await?;
            input = new_input;
            if output[0].0.size() > 0 {
                break;
            }
        }
    }

    // Prefill benchmark
    println!("   Prefill benchmark ({} tokens)...", PREFILL_LENGTH);
    let mut prefill_times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        fastrand::seed(42);
        let tokens: Vec<u16> = (0..PREFILL_LENGTH)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);

        let start = Instant::now();
        loop {
            let (new_input, output) = runtime.infer(input).await?;
            input = new_input;
            if output[0].0.size() > 0 {
                break;
            }
        }
        prefill_times.push(start.elapsed());
    }

    let avg_prefill =
        prefill_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / prefill_times.len() as f64;
    let prefill_tps = PREFILL_LENGTH as f64 / avg_prefill;
    println!("   Prefill: {:.2} tok/s", prefill_tps);

    // Generation benchmark
    println!("   Generation benchmark ({} tokens)...", GENERATION_LENGTH);
    let mut gen_times = Vec::with_capacity(BENCH_RUNS);
    for _ in 0..BENCH_RUNS {
        fastrand::seed(42);
        // First do prefill
        let tokens: Vec<u16> = (0..32)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        loop {
            let (new_input, output) = runtime.infer(input).await?;
            input = new_input;
            if output[0].0.size() > 0 {
                break;
            }
        }

        // Then measure generation
        let start = Instant::now();
        for _ in 0..GENERATION_LENGTH {
            let token = fastrand::u16(0..((info.num_vocab - 1) as u16));
            let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
            input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
            let (new_input, _) = runtime.infer(input).await?;
            input = new_input;
        }
        gen_times.push(start.elapsed());
    }

    let avg_gen = gen_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / gen_times.len() as f64;
    let generation_tps = GENERATION_LENGTH as f64 / avg_gen;
    println!("   Generation: {:.2} tok/s", generation_tps);
    println!();

    // === QUALITY CHECK ===
    println!("⏳ Running quality check (deterministic generation)...");
    fastrand::seed(QUALITY_SEED);

    // Generate deterministic prompt
    let prompt_tokens: Vec<u16> = (0..QUALITY_PROMPT_LENGTH)
        .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
        .collect();

    // Prefill
    let prompt = RnnInputBatch::new(prompt_tokens.clone(), RnnOption::Last);
    let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
    let mut last_output: TensorCpu<f32> =
        TensorCpu::from_data(Shape::new(1, 1, 1, 1), vec![0.0f32])?;
    loop {
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        if output[0].0.size() > 0 {
            last_output = output[0].0.clone();
            break;
        }
    }

    // Generate tokens deterministically using argmax on logits
    let mut quality_tokens = Vec::with_capacity(QUALITY_GEN_LENGTH);
    for _ in 0..QUALITY_GEN_LENGTH {
        let probs = softmax_one(&context, last_output.clone()).await?;
        let probs_vec: Vec<f32> = probs.to_vec();
        let token = sample_argmax(&probs_vec);
        quality_tokens.push(token);

        let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
    }

    // Create hash of quality tokens for easy comparison
    let quality_hash = format!(
        "{:x}",
        quality_tokens
            .iter()
            .fold(0u64, |acc, &t| acc.wrapping_mul(31).wrapping_add(t as u64))
    );
    println!("   Quality hash: {}", quality_hash);
    println!(
        "   First 8 tokens: {:?}",
        &quality_tokens[..8.min(quality_tokens.len())]
    );
    println!();

    // Get current timestamp
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let timestamp = format_timestamp(now);

    let result = BenchmarkResult {
        timestamp,
        title: args.title.clone(),
        change: args.change.clone(),
        model_name,
        model_version: format!("{:?}", info.version),
        num_layers: info.num_layer,
        num_emb: info.num_emb,
        gpu_name,
        load_time_ms,
        prefill_tps,
        generation_tps,
        quality_hash,
        quality_tokens,
    };

    Ok(result)
}

fn write_result_to_markdown(result: &BenchmarkResult, output_path: &PathBuf) -> Result<()> {
    let mut existing_content = String::new();
    let mut has_title_section = false;

    // Read existing file if it exists
    if output_path.exists() {
        let mut file = File::open(output_path)?;
        file.read_to_string(&mut existing_content)?;
        has_title_section = existing_content.contains(&format!("## {}", result.title));
    }

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    if existing_content.is_empty() {
        // Write header for new file
        writeln!(file, "# Benchmark Results")?;
        writeln!(file)?;
        writeln!(
            file,
            "This file tracks performance benchmarks across changes to the codebase."
        )?;
        writeln!(file)?;
        writeln!(file, "**How to use:**")?;
        writeln!(
            file,
            "1. Run benchmark BEFORE making changes with a descriptive title"
        )?;
        writeln!(file, "2. Make your changes")?;
        writeln!(
            file,
            "3. Run benchmark AFTER with the SAME title to group results"
        )?;
        writeln!(file)?;
        writeln!(file, "```bash")?;
        writeln!(
            file,
            "cargo run --release --example benchmark -- --model <path> --title \"<title>\" --change \"<description>\""
        )?;
        writeln!(file, "```")?;
        writeln!(file)?;
        writeln!(file, "---")?;
        writeln!(file)?;
    }

    if has_title_section {
        // Find the title section and append to it
        let title_marker = format!("## {}", result.title);
        let parts: Vec<&str> = existing_content.splitn(2, &title_marker).collect();

        if parts.len() == 2 {
            // Write everything before the title section
            write!(file, "{}", parts[0])?;
            write!(file, "{}", title_marker)?;

            // Find where the next section starts (next ## or end of file)
            let rest = parts[1];
            let next_section = rest.find("\n## ");

            let (section_content, after_section) = if let Some(pos) = next_section {
                (&rest[..pos], &rest[pos..])
            } else {
                (rest, "")
            };

            // Write existing section content
            write!(file, "{}", section_content)?;

            // Append new result
            writeln!(file)?;
            write_single_result(&mut file, result)?;

            // Write rest of file
            write!(file, "{}", after_section)?;
        }
    } else {
        // Write existing content
        write!(file, "{}", existing_content)?;

        // Add new section
        writeln!(file, "## {}", result.title)?;
        writeln!(file)?;
        write_single_result(&mut file, result)?;
    }

    Ok(())
}

fn write_single_result(file: &mut File, result: &BenchmarkResult) -> Result<()> {
    writeln!(file, "### {} - {}", result.timestamp, result.change)?;
    writeln!(file)?;
    writeln!(file, "| Metric | Value |")?;
    writeln!(file, "|--------|-------|")?;
    writeln!(file, "| Model | {} |", result.model_name)?;
    writeln!(file, "| Version | {} |", result.model_version)?;
    writeln!(file, "| Layers | {} |", result.num_layers)?;
    writeln!(file, "| Embedding | {} |", result.num_emb)?;
    writeln!(file, "| GPU | {} |", result.gpu_name)?;
    writeln!(
        file,
        "| **Load Time** | **{:.2} ms** |",
        result.load_time_ms
    )?;
    writeln!(
        file,
        "| **Prefill** | **{:.2} tok/s** |",
        result.prefill_tps
    )?;
    writeln!(
        file,
        "| **Generation** | **{:.2} tok/s** |",
        result.generation_tps
    )?;
    writeln!(file, "| Quality Hash | `{}` |", result.quality_hash)?;
    writeln!(file)?;
    writeln!(
        file,
        "<details><summary>Quality tokens (first 16)</summary>"
    )?;
    writeln!(file)?;
    writeln!(file, "```")?;
    writeln!(
        file,
        "{:?}",
        &result.quality_tokens[..16.min(result.quality_tokens.len())]
    )?;
    writeln!(file, "```")?;
    writeln!(file)?;
    writeln!(file, "</details>")?;
    writeln!(file)?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let result = run_benchmark(&args).await?;

    // Write to markdown
    write_result_to_markdown(&result, &args.output)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Load Time:    {:>8.2} ms                                  ║",
        result.load_time_ms
    );
    println!(
        "║  Prefill:      {:>8.2} tok/s                               ║",
        result.prefill_tps
    );
    println!(
        "║  Generation:   {:>8.2} tok/s                               ║",
        result.generation_tps
    );
    println!(
        "║  Quality Hash: {}                              ║",
        &result.quality_hash[..16.min(result.quality_hash.len())]
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("📄 Results written to: {}", args.output.display());

    Ok(())
}

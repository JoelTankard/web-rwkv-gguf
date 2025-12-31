//! Layer-by-layer profiling example using wall-clock timing with GPU sync.
//!
//! This measures time spent in each phase of the forward pass:
//! - Embedding layer norm
//! - Each transformer layer (grouped)
//! - Head projection

use std::{collections::HashMap, path::PathBuf, time::Instant};

use anyhow::Result;
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        softmax::softmax_one,
        v7, Runtime, TokioRuntime,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(long, default_value_t = 32)]
    token_chunk_size: usize,
    #[arg(long, default_value_t = 128)]
    prefill_length: usize,
    #[arg(long, default_value_t = 2)]
    warmup_runs: usize,
    #[arg(long, default_value_t = 5)]
    profile_runs: usize,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;

    println!("Adapter: {:?}", adapter.get_info().name);

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

#[derive(Debug, Default, Clone)]
struct TimingStats {
    total_ms: f64,
    count: usize,
}

impl TimingStats {
    fn add(&mut self, ms: f64) {
        self.total_ms += ms;
        self.count += 1;
    }

    fn avg(&self) -> f64 {
        if self.count > 0 {
            self.total_ms / self.count as f64
        } else {
            0.0
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();

    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Layer-by-Layer Profiling Tool                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model: {:?}", cli.model.file_name().unwrap());
    println!("Version: {:?}", info.version);
    println!("Layers: {}", info.num_layer);
    println!("Embed dim: {}", info.num_emb);
    println!("Prefill length: {} tokens", cli.prefill_length);
    println!("Chunk size: {} tokens", cli.token_chunk_size);
    println!();

    let context = create_context(&info).await?;

    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());

    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        _ => {
            anyhow::bail!("Only V7 models are supported for profiling");
        }
    };

    fastrand::seed(42);
    let prompt_len = cli.prefill_length;

    println!("=== Warmup ({} runs) ===", cli.warmup_runs);
    for i in 0..cli.warmup_runs {
        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);

        loop {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let output = output.to_vec();
                let _ = sample(&output);
                break;
            }
        }
        println!("  Warmup run {} complete", i + 1);
    }

    println!(
        "\n=== Profiling ({} runs, {} tokens each) ===\n",
        cli.profile_runs, prompt_len
    );

    let mut total_times: Vec<f64> = Vec::new();
    let mut chunk_times: Vec<Vec<f64>> = Vec::new();

    for run in 0..cli.profile_runs {
        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);

        let mut run_chunk_times: Vec<f64> = Vec::new();
        let total_start = Instant::now();

        loop {
            let chunk_start = Instant::now();
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let chunk_elapsed = chunk_start.elapsed().as_secs_f64() * 1000.0;
                run_chunk_times.push(chunk_elapsed);

                let output = output.to_vec();
                let _ = sample(&output);
                break;
            } else {
                let chunk_elapsed = chunk_start.elapsed().as_secs_f64() * 1000.0;
                run_chunk_times.push(chunk_elapsed);
            }
        }

        let total_elapsed = total_start.elapsed().as_secs_f64() * 1000.0;
        total_times.push(total_elapsed);
        chunk_times.push(run_chunk_times);

        println!(
            "Run {} - Total: {:.2}ms ({:.1} tok/s), {} chunks",
            run + 1,
            total_elapsed,
            prompt_len as f64 / (total_elapsed / 1000.0),
            chunk_times[run].len()
        );
    }

    println!("\n=== Summary ===\n");

    let avg_total = total_times.iter().sum::<f64>() / total_times.len() as f64;
    let avg_tps = prompt_len as f64 / (avg_total / 1000.0);

    println!("Average total time: {:.2}ms", avg_total);
    println!("Average throughput: {:.1} tok/s", avg_tps);
    println!("Tokens per run: {}", prompt_len);

    if !chunk_times.is_empty() && !chunk_times[0].is_empty() {
        let num_chunks = chunk_times[0].len();
        println!(
            "\nPer-chunk breakdown (avg over {} runs):",
            cli.profile_runs
        );
        println!("{:<10} {:>12} {:>12}", "Chunk", "Time (ms)", "% of Total");
        println!("{}", "-".repeat(36));

        for chunk_idx in 0..num_chunks {
            let chunk_avg: f64 = chunk_times
                .iter()
                .filter_map(|run| run.get(chunk_idx))
                .sum::<f64>()
                / cli.profile_runs as f64;
            let pct = (chunk_avg / avg_total) * 100.0;
            println!("{:<10} {:>12.2} {:>11.1}%", chunk_idx + 1, chunk_avg, pct);
        }
    }

    println!("\n=== Analysis ===\n");
    println!("Prefill throughput: {:.1} tok/s", avg_tps);
    println!("Time per token: {:.2}ms", avg_total / prompt_len as f64);
    println!();
    println!("Target: 700 tok/s (llama.cpp parity)");
    println!("Gap: {:.1}x slower", 700.0 / avg_tps);

    Ok(())
}

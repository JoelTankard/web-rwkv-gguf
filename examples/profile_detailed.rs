//! Detailed profiling to identify the exact bottleneck in the forward pass.
//!
//! Measures:
//! - Forward pass (without output)
//! - Head projection
//! - Softmax
//! - GPU readback

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
    #[arg(long, default_value_t = 128)]
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
    println!("║              Detailed Profiling Tool                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model: {:?}", cli.model.file_name().unwrap());
    println!("Version: {:?}", info.version);
    println!("Layers: {}", info.num_layer);
    println!("Embed dim: {}", info.num_emb);
    println!("Vocab size: {}", info.num_vocab);
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

    println!("\n=== Detailed Profiling ({} runs) ===\n", cli.profile_runs);

    let mut infer_times: Vec<f64> = Vec::new();
    let mut softmax_times: Vec<f64> = Vec::new();
    let mut to_vec_times: Vec<f64> = Vec::new();
    let mut total_times: Vec<f64> = Vec::new();

    for run in 0..cli.profile_runs {
        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);

        let total_start = Instant::now();
        let mut run_infer_time = 0.0;
        let mut run_softmax_time = 0.0;
        let mut run_to_vec_time = 0.0;

        loop {
            let input = prompt.clone();

            let infer_start = Instant::now();
            let (input, output) = runtime.infer(input).await?;
            let infer_elapsed = infer_start.elapsed().as_secs_f64() * 1000.0;
            run_infer_time += infer_elapsed;

            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let softmax_start = Instant::now();
                let output = softmax_one(&context, output).await?;
                let softmax_elapsed = softmax_start.elapsed().as_secs_f64() * 1000.0;
                run_softmax_time += softmax_elapsed;

                let to_vec_start = Instant::now();
                let output = output.to_vec();
                let to_vec_elapsed = to_vec_start.elapsed().as_secs_f64() * 1000.0;
                run_to_vec_time += to_vec_elapsed;

                let _ = sample(&output);
                break;
            }
        }

        let total_elapsed = total_start.elapsed().as_secs_f64() * 1000.0;

        infer_times.push(run_infer_time);
        softmax_times.push(run_softmax_time);
        to_vec_times.push(run_to_vec_time);
        total_times.push(total_elapsed);

        println!(
            "Run {} - Infer: {:.2}ms, Softmax: {:.2}ms, ToVec: {:.2}ms, Total: {:.2}ms",
            run + 1,
            run_infer_time,
            run_softmax_time,
            run_to_vec_time,
            total_elapsed
        );
    }

    println!("\n=== Summary ===\n");

    let avg_infer = infer_times.iter().sum::<f64>() / infer_times.len() as f64;
    let avg_softmax = softmax_times.iter().sum::<f64>() / softmax_times.len() as f64;
    let avg_to_vec = to_vec_times.iter().sum::<f64>() / to_vec_times.len() as f64;
    let avg_total = total_times.iter().sum::<f64>() / total_times.len() as f64;

    println!("{:<20} {:>12} {:>10}", "Operation", "Time (ms)", "% Total");
    println!("{}", "-".repeat(44));
    println!(
        "{:<20} {:>12.2} {:>9.1}%",
        "Inference",
        avg_infer,
        (avg_infer / avg_total) * 100.0
    );
    println!(
        "{:<20} {:>12.2} {:>9.1}%",
        "Softmax",
        avg_softmax,
        (avg_softmax / avg_total) * 100.0
    );
    println!(
        "{:<20} {:>12.2} {:>9.1}%",
        "GPU→CPU (to_vec)",
        avg_to_vec,
        (avg_to_vec / avg_total) * 100.0
    );
    println!("{}", "-".repeat(44));
    println!("{:<20} {:>12.2}", "Total", avg_total);

    println!("\n=== Performance ===\n");
    let tps = prompt_len as f64 / (avg_total / 1000.0);
    let infer_tps = prompt_len as f64 / (avg_infer / 1000.0);
    println!("Total throughput: {:.1} tok/s", tps);
    println!("Inference-only throughput: {:.1} tok/s", infer_tps);
    println!();
    println!("Target: 700 tok/s");
    println!("Gap (total): {:.1}x slower", 700.0 / tps);
    println!("Gap (inference): {:.1}x slower", 700.0 / infer_tps);

    Ok(())
}

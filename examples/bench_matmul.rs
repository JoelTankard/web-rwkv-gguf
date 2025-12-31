//! Benchmark matrix multiplication with different configurations.

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
    #[arg(long, default_value_t = 2)]
    warmup_runs: usize,
    #[arg(long, default_value_t = 5)]
    bench_runs: usize,
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

fn sample(probs: &[f32]) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

async fn bench_prefill(
    runtime: &dyn Runtime<Rnn>,
    context: &Context,
    num_vocab: usize,
    prefill_len: usize,
    chunk_size: usize,
    runs: usize,
) -> f64 {
    let mut total_time = 0.0;

    for _ in 0..runs {
        let tokens: Vec<u16> = (0..prefill_len)
            .map(|_| fastrand::u16(0..((num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], chunk_size);

        let start = Instant::now();
        loop {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await.unwrap();
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(context, output).await.unwrap();
                let _ = output.to_vec();
                break;
            }
        }
        total_time += start.elapsed().as_secs_f64() * 1000.0;
    }

    total_time / runs as f64
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .init()?;

    let cli = Cli::parse();

    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Matrix Multiplication Benchmark                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model: {:?}", cli.model.file_name().unwrap());
    println!("Layers: {}, Embed: {}", info.num_layer, info.num_emb);
    println!();

    let context = create_context(&info).await?;
    println!("GPU: {}", context.adapter.get_info().name);
    println!();

    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            Box::new(TokioRuntime::new(bundle).await)
        }
        _ => anyhow::bail!("Only V7 models supported"),
    };

    fastrand::seed(42);

    // Warmup
    println!("Warming up ({} runs)...", cli.warmup_runs);
    bench_prefill(&*runtime, &context, info.num_vocab, 64, 64, cli.warmup_runs).await;

    // Test different prefill lengths and chunk sizes
    println!("\n=== Prefill Performance ===\n");
    println!(
        "{:<12} {:>12} {:>12} {:>12}",
        "Tokens", "Chunk", "Time (ms)", "Tok/s"
    );
    println!("{}", "-".repeat(52));

    let test_configs = [
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 512),
        (1024, 1024),
    ];

    for (prefill_len, chunk_size) in test_configs {
        let time_ms = bench_prefill(
            &*runtime,
            &context,
            info.num_vocab,
            prefill_len,
            chunk_size,
            cli.bench_runs,
        )
        .await;
        let tps = prefill_len as f64 / (time_ms / 1000.0);
        println!(
            "{:<12} {:>12} {:>12.2} {:>12.1}",
            prefill_len, chunk_size, time_ms, tps
        );
    }

    println!("\n=== Analysis ===\n");

    // Calculate theoretical peak
    let best_tps = 466.0; // From our earlier profiling
    let target_tps = 700.0;
    println!("Current best: {:.0} tok/s (chunk=512)", best_tps);
    println!("Target: {:.0} tok/s", target_tps);
    println!("Gap: {:.1}x", target_tps / best_tps);

    Ok(())
}

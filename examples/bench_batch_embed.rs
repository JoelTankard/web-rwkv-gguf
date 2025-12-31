//! Benchmark batch embedding vs sequential embedding.

use std::{path::PathBuf, time::Instant};

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
        v7, Runtime, TokioRuntime,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(long, default_value_t = 6)]
    num_sequences: usize,
    #[arg(long, default_value_t = 50)]
    tokens_per_sequence: usize,
    #[arg(long, default_value_t = 512)]
    token_chunk_size: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
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
    println!("║              Batch Embedding Benchmark                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model: {:?}", cli.model.file_name().unwrap());
    println!(
        "Sequences: {}, Tokens/seq: {}",
        cli.num_sequences, cli.tokens_per_sequence
    );
    println!();

    let context = create_context(&info).await?;
    println!("GPU: {}", context.adapter.get_info().name);
    println!();

    let builder = ModelBuilder::new(&context, model).quant(std::collections::HashMap::new());
    let runtime: Box<dyn Runtime<Rnn>> = match info.version {
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, cli.num_sequences);
            Box::new(TokioRuntime::new(bundle).await)
        }
        _ => anyhow::bail!("Only V7 models supported"),
    };

    fastrand::seed(42);

    // Generate test sequences
    let sequences: Vec<Vec<u16>> = (0..cli.num_sequences)
        .map(|_| {
            (0..cli.tokens_per_sequence)
                .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
                .collect()
        })
        .collect();

    let total_tokens = cli.num_sequences * cli.tokens_per_sequence;

    println!("=== Sequential Embedding (one at a time) ===\n");

    let mut seq_times = vec![];
    for run in 0..cli.runs {
        let start = Instant::now();

        for seq in &sequences {
            let prompt = RnnInputBatch::new(seq.clone(), RnnOption::EmbedLast);
            let mut input = RnnInput::new(vec![prompt], cli.token_chunk_size);

            loop {
                let (new_input, output) = runtime.infer(input).await?;
                input = new_input;

                if output[0].0.size() > 0 {
                    let _ = output[0].0.clone().to_vec();
                    break;
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        seq_times.push(elapsed);
        println!(
            "Run {}: {:.2}ms ({:.1} tok/s)",
            run + 1,
            elapsed,
            total_tokens as f64 / (elapsed / 1000.0)
        );
    }

    let avg_seq = seq_times.iter().sum::<f64>() / seq_times.len() as f64;
    println!(
        "\nAverage: {:.2}ms ({:.1} tok/s)\n",
        avg_seq,
        total_tokens as f64 / (avg_seq / 1000.0)
    );

    println!("=== Batched Embedding (all at once) ===\n");

    let mut batch_times = vec![];
    for run in 0..cli.runs {
        let start = Instant::now();

        let batches: Vec<RnnInputBatch> = sequences
            .iter()
            .map(|seq| RnnInputBatch::new(seq.clone(), RnnOption::EmbedLast))
            .collect();

        let mut input = RnnInput::new(batches, cli.token_chunk_size);

        loop {
            let (new_input, output) = runtime.infer(input).await?;
            input = new_input;

            let all_done = output.0.iter().all(|o| o.0.size() > 0);
            if all_done {
                for o in &output.0 {
                    let _ = o.0.clone().to_vec();
                }
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        batch_times.push(elapsed);
        println!(
            "Run {}: {:.2}ms ({:.1} tok/s)",
            run + 1,
            elapsed,
            total_tokens as f64 / (elapsed / 1000.0)
        );
    }

    let avg_batch = batch_times.iter().sum::<f64>() / batch_times.len() as f64;
    println!(
        "\nAverage: {:.2}ms ({:.1} tok/s)\n",
        avg_batch,
        total_tokens as f64 / (avg_batch / 1000.0)
    );

    println!("=== Summary ===\n");
    println!("Sequential: {:.2}ms", avg_seq);
    println!("Batched:    {:.2}ms", avg_batch);
    println!("Speedup:    {:.2}x", avg_seq / avg_batch);

    Ok(())
}

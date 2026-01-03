//! Simple generation-only benchmark - no prefill, just single-token generation speed.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use half::f16;
use instant::Instant;
use memmap2::Mmap;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, Quant},
        softmax::softmax_one,
        v7, Runtime, TokioRuntime,
    },
};

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    println!("GPU: {}", context.adapter.get_info().name);
    Ok(context)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(short, long, default_value_t = 32)]
    tokens: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();
    println!("Model: {:?}", cli.model);
    println!("Generating {} tokens...", cli.tokens);

    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = GgufReader::new(&data).expect("failed to parse GGUF file");
    let info = Loader::info(&model)?;
    println!(
        "Version: {:?}, Layers: {}, Embed: {}",
        info.version, info.num_layer, info.num_emb
    );

    let context = create_context(&info).await?;

    let quant: std::collections::HashMap<usize, Quant> = Default::default();
    let builder = ModelBuilder::new(&context, model).quant(quant);

    let model = builder.build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);
    let runtime: Box<dyn Runtime<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    // Start with a single token (no prefill)
    let start_token = 33u32; // newline
    let mut tokens = vec![start_token];

    const TOKEN_CHUNK_SIZE: usize = 32;

    println!("\nWarming up (3 tokens)...");
    for _ in 0..3 {
        let token = fastrand::u16(0..65535);
        let input = RnnInputBatch::new(vec![token], RnnOption::Last);
        let input = RnnInput::new(vec![input], TOKEN_CHUNK_SIZE);
        let (_, _) = runtime.infer(input).await?;
    }

    println!("Benchmarking {} tokens...", cli.tokens);
    let instant = Instant::now();

    for i in 0..cli.tokens {
        let token = fastrand::u16(0..65535);
        let input = RnnInputBatch::new(vec![token], RnnOption::Last);
        let input = RnnInput::new(vec![input], TOKEN_CHUNK_SIZE);
        let (_, _) = runtime.infer(input).await?;

        if (i + 1) % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }

    let duration = instant.elapsed();
    let tps = cli.tokens as f64 / duration.as_secs_f64();

    println!("\n\n=== Results ===");
    println!("Tokens: {}", cli.tokens);
    println!("Time: {:.2}s", duration.as_secs_f64());
    println!("Speed: {:.2} tok/s", tps);

    Ok(())
}

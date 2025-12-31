//! GPU profiling example to measure per-operation timing during inference.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    profiler::GpuProfiler,
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
    #[arg(long, default_value_t = 64)]
    prefill_length: usize,
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,
    #[arg(long, default_value_t = 3)]
    profile_runs: usize,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;

    println!("Adapter: {:?}", adapter.get_info());

    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;

    if context.profiler.is_available() {
        println!("GPU timestamp queries: AVAILABLE");
    } else {
        println!("GPU timestamp queries: NOT AVAILABLE");
        println!("Note: Profiling requires TIMESTAMP_QUERY feature support");
    }

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
        .with_module_level("profile", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();

    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;
    println!("Model: {:?}", cli.model.file_name().unwrap());
    println!("Version: {:?}", info.version);
    println!("Layers: {}", info.num_layer);
    println!("Embed dim: {}", info.num_emb);

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

    println!("\n=== Warmup ({} runs) ===", cli.warmup_runs);
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
        "\n=== Profiling ({} runs, {} tokens each) ===",
        cli.profile_runs, prompt_len
    );

    context.profiler.set_enabled(true);

    for run in 0..cli.profile_runs {
        context.profiler.reset();

        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);

        let start = std::time::Instant::now();

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

        let elapsed = start.elapsed();
        println!(
            "\nRun {} - Wall time: {:.2}ms ({:.1} tok/s)",
            run + 1,
            elapsed.as_secs_f64() * 1000.0,
            prompt_len as f64 / elapsed.as_secs_f64()
        );

        let results = context.profiler.read_results(&context.device).await;
        if !results.is_empty() {
            GpuProfiler::print_results(&results);
        } else {
            println!("No GPU timing results available");
        }
    }

    context.profiler.set_enabled(false);

    Ok(())
}

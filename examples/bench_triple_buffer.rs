//! Benchmark comparing TokioRuntime vs TripleBufferRuntime.
//!
//! TripleBufferRuntime overlaps job building with GPU execution.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use half::f16;
use instant::Instant;
use memmap2::Mmap;
use tokio::fs::File as TokioFile;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        triple_buffer::TripleBufferRuntime,
        v7, Runtime, TokioRuntime,
    },
    tensor::{shape::Shape, TensorCpu, TensorInit},
};

const WARMUP_RUNS: usize = 10;
const BENCH_TOKENS: usize = 64;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark TokioRuntime vs TripleBufferRuntime")]
struct Args {
    #[arg(short, long, default_value = "assets/models/2.9b-Q8_0.gguf")]
    model: PathBuf,
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

async fn run_generation<R: Runtime<Rnn> + ?Sized>(
    runtime: &R,
    context: &Context,
    num_tokens: usize,
    seed: u64,
) -> Result<(std::time::Duration, Vec<u16>)> {
    use web_rwkv::runtime::softmax::softmax_one;
    
    fastrand::seed(seed);
    
    // Initial prompt
    let initial_token: u16 = fastrand::u16(1000..2000);
    let batch = RnnInputBatch::new(vec![initial_token], RnnOption::Last);
    let mut input = RnnInput::new(vec![batch], 32);
    
    // Get first output
    let (new_input, output) = runtime.infer(input).await?;
    input = new_input;
    let mut last_output = output[0].0.clone();
    
    let mut generated_tokens = Vec::with_capacity(num_tokens);
    
    let start = Instant::now();
    
    for _ in 0..num_tokens {
        // Sample next token
        let probs = softmax_one(context, last_output.clone()).await?;
        let probs_vec: Vec<f32> = probs.to_vec();
        let token = sample_argmax(&probs_vec);
        generated_tokens.push(token);
        
        // Generate next
        let batch = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![batch], 32);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
    }
    
    let elapsed = start.elapsed();
    Ok((elapsed, generated_tokens))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     TokioRuntime vs TripleBufferRuntime Benchmark            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?}", args.model.file_name().unwrap());
    println!("   Version: {:?}", info.version);
    println!();

    let context = create_context(&info).await?;
    println!("🖥️  GPU: {}", context.adapter.get_info().name);
    println!();

    if info.version != ModelVersion::V7 {
        println!("❌ This benchmark only supports V7 models");
        return Ok(());
    }

    // Build model for TokioRuntime
    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
    let model = builder.build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);
    let tokio_runtime: Box<dyn Runtime<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    // Reload model for TripleBufferRuntime (need fresh bundle)
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
    let model = builder.build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);
    let triple_runtime: Box<dyn Runtime<Rnn>> = Box::new(TripleBufferRuntime::new(bundle).await);

    // Warmup TokioRuntime
    println!("⏳ Warming up TokioRuntime ({} runs)...", WARMUP_RUNS);
    for _ in 0..WARMUP_RUNS {
        let _ = run_generation(&*tokio_runtime, &context, 1, 42).await?;
    }

    // Warmup TripleBufferRuntime
    println!("⏳ Warming up TripleBufferRuntime ({} runs)...", WARMUP_RUNS);
    for _ in 0..WARMUP_RUNS {
        let _ = run_generation(&*triple_runtime, &context, 1, 42).await?;
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking TokioRuntime ({} tokens)", BENCH_TOKENS);
    println!("═══════════════════════════════════════════════════════════════");
    
    let (tokio_time, tokio_tokens) = run_generation(&*tokio_runtime, &context, BENCH_TOKENS, 12345).await?;
    let tokio_tps = BENCH_TOKENS as f64 / tokio_time.as_secs_f64();
    println!("   Time: {:.2} ms", tokio_time.as_secs_f64() * 1000.0);
    println!("   Speed: {:.2} tok/s", tokio_tps);
    println!("   First 5 tokens: {:?}", &tokio_tokens[..5.min(tokio_tokens.len())]);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking TripleBufferRuntime ({} tokens)", BENCH_TOKENS);
    println!("═══════════════════════════════════════════════════════════════");
    
    let (triple_time, triple_tokens) = run_generation(&*triple_runtime, &context, BENCH_TOKENS, 12345).await?;
    let triple_tps = BENCH_TOKENS as f64 / triple_time.as_secs_f64();
    println!("   Time: {:.2} ms", triple_time.as_secs_f64() * 1000.0);
    println!("   Speed: {:.2} tok/s", triple_tps);
    println!("   First 5 tokens: {:?}", &triple_tokens[..5.min(triple_tokens.len())]);

    // Verify correctness
    let matching = tokio_tokens.iter().zip(triple_tokens.iter()).filter(|(a, b)| a == b).count();

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  TokioRuntime:        {:>8.2} tok/s                         ║", tokio_tps);
    println!("║  TripleBufferRuntime: {:>8.2} tok/s                         ║", triple_tps);
    println!("║  Speedup:             {:>8.2}x                              ║", triple_tps / tokio_tps);
    println!("║  Token match:         {}/{} ({:.1}%)                       ║", 
             matching, BENCH_TOKENS, matching as f64 / BENCH_TOKENS as f64 * 100.0);
    println!("╚══════════════════════════════════════════════════════════════╝");

    if matching != BENCH_TOKENS {
        println!();
        println!("⚠️  Warning: Token mismatch detected - outputs differ!");
    }

    Ok(())
}

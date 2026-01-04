//! Benchmark comparing GPU-side sampling vs CPU-side sampling.
//!
//! GPU-side sampling avoids reading back 260KB of logits per token.
//! Instead, it performs argmax on GPU and only reads back 4 bytes (token ID).

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
        softmax::{sample_argmax_one, softmax_one},
        v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{shape::Shape, TensorCpu, TensorInit},
};

const WARMUP_RUNS: usize = 5;
const BENCH_RUNS: usize = 20;
const TOKEN_CHUNK_SIZE: usize = 32;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark GPU vs CPU sampling")]
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

fn sample_argmax_cpu(probs: &[f32]) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u16)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          GPU vs CPU Sampling Benchmark                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?}", args.model.file_name().unwrap());
    println!("   Version: {:?}", info.version);
    println!("   Vocab size: {} ({:.2} KB per readback)", 
             info.num_vocab, 
             info.num_vocab as f64 * 4.0 / 1024.0);
    println!();

    let context = create_context(&info).await?;
    println!("🖥️  GPU: {}", context.adapter.get_info().name);
    println!();

    // Build runtime
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
    println!("⏳ Warming up ({} runs)...", WARMUP_RUNS);
    fastrand::seed(42);
    let mut last_output: TensorCpu<f32> = TensorCpu::from_data(Shape::new(1, 1, 1, 1), vec![0.0f32])?;
    
    // Initial prefill
    let tokens: Vec<u16> = (0..32)
        .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
        .collect();
    let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
    let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
    loop {
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        if output[0].0.size() > 0 {
            last_output = output[0].0.clone();
            break;
        }
    }

    // Warmup generation
    for _ in 0..WARMUP_RUNS {
        let token = fastrand::u16(0..((info.num_vocab - 1) as u16));
        let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking CPU-side sampling (softmax + argmax on CPU)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut cpu_times = Vec::with_capacity(BENCH_RUNS);
    let mut cpu_tokens = Vec::with_capacity(BENCH_RUNS);
    
    for i in 0..BENCH_RUNS {
        // Generate a token using CPU sampling
        let start = Instant::now();
        
        // This is the current approach: read back all logits, softmax on CPU, argmax on CPU
        let probs = softmax_one(&context, last_output.clone()).await?;
        let probs_vec: Vec<f32> = probs.to_vec();
        let token = sample_argmax_cpu(&probs_vec);
        
        let elapsed = start.elapsed();
        cpu_times.push(elapsed);
        cpu_tokens.push(token);
        
        // Generate next token
        let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
        
        if i < 5 {
            println!("   Run {}: {:.3} ms, token: {}", i + 1, elapsed.as_secs_f64() * 1000.0, token);
        }
    }
    
    let cpu_avg = cpu_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / cpu_times.len() as f64;
    println!("   ...");
    println!("   Average: {:.3} ms per sample", cpu_avg * 1000.0);

    // Reset for GPU sampling test
    fastrand::seed(42);
    let tokens: Vec<u16> = (0..32)
        .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
        .collect();
    let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
    let mut input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
    loop {
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        if output[0].0.size() > 0 {
            last_output = output[0].0.clone();
            break;
        }
    }
    
    // Warmup GPU sampling
    for _ in 0..WARMUP_RUNS {
        let token = fastrand::u16(0..((info.num_vocab - 1) as u16));
        let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking GPU-side sampling (argmax on GPU, 4 bytes back)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut gpu_times = Vec::with_capacity(BENCH_RUNS);
    let mut gpu_tokens = Vec::with_capacity(BENCH_RUNS);
    
    for i in 0..BENCH_RUNS {
        // Generate a token using GPU sampling
        let start = Instant::now();
        
        // New approach: argmax directly on GPU, only read back 4 bytes
        let token = sample_argmax_one(&context, last_output.clone()).await? as u16;
        
        let elapsed = start.elapsed();
        gpu_times.push(elapsed);
        gpu_tokens.push(token);
        
        // Generate next token
        let prompt = RnnInputBatch::new(vec![token], RnnOption::Last);
        input = RnnInput::new(vec![prompt], TOKEN_CHUNK_SIZE);
        let (new_input, output) = runtime.infer(input).await?;
        input = new_input;
        last_output = output[0].0.clone();
        
        if i < 5 {
            println!("   Run {}: {:.3} ms, token: {}", i + 1, elapsed.as_secs_f64() * 1000.0, token);
        }
    }
    
    let gpu_avg = gpu_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / gpu_times.len() as f64;
    println!("   ...");
    println!("   Average: {:.3} ms per sample", gpu_avg * 1000.0);

    // Verify correctness - tokens should match (both use argmax)
    let matching = cpu_tokens.iter().zip(gpu_tokens.iter()).filter(|(a, b)| a == b).count();
    
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  CPU sampling:  {:>8.3} ms/sample                          ║", cpu_avg * 1000.0);
    println!("║  GPU sampling:  {:>8.3} ms/sample                          ║", gpu_avg * 1000.0);
    println!("║  Speedup:       {:>8.2}x                                   ║", cpu_avg / gpu_avg);
    println!("║  Token match:   {}/{} ({:.1}%)                             ║", 
             matching, BENCH_RUNS, matching as f64 / BENCH_RUNS as f64 * 100.0);
    println!("╚══════════════════════════════════════════════════════════════╝");

    if matching != BENCH_RUNS {
        println!();
        println!("⚠️  Warning: Token mismatch detected!");
        println!("   CPU tokens: {:?}", &cpu_tokens[..5.min(cpu_tokens.len())]);
        println!("   GPU tokens: {:?}", &gpu_tokens[..5.min(gpu_tokens.len())]);
    }

    Ok(())
}

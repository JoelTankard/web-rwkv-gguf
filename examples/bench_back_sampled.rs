//! Benchmark comparing back() vs back_sampled() on RnnJob.
//!
//! back() reads back 260KB of logits per token.
//! back_sampled() performs argmax on GPU and reads back only 4 bytes.

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
        infer::{RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v7, Dispatcher, Job, JobInput,
    },
};

const WARMUP_RUNS: usize = 10;
const BENCH_RUNS: usize = 50;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark back() vs back_sampled()")]
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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       back() vs back_sampled() Benchmark                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?}", args.model.file_name().unwrap());
    println!("   Version: {:?}", info.version);
    println!("   Vocab size: {} ({:.2} KB per full readback)", 
             info.num_vocab, 
             info.num_vocab as f64 * 4.0 / 1024.0);
    println!();

    let context = create_context(&info).await?;
    println!("🖥️  GPU: {}", context.adapter.get_info().name);
    println!();

    // Build model (only V7 supported for this test)
    if info.version != ModelVersion::V7 {
        println!("❌ This benchmark only supports V7 models");
        return Ok(());
    }

    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
    let model = builder.build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);

    // Create a simple input
    fastrand::seed(42);
    let token: u16 = fastrand::u16(0..((info.num_vocab - 1) as u16));
    let batch = RnnInputBatch::new(vec![token], RnnOption::Last);
    let rnn_input = RnnInput::new(vec![batch], 32);
    
    // Get RnnInfo and chunk from the input
    let rnn_info = rnn_input.iter().next().unwrap();
    let chunk = rnn_input.chunk();

    println!("⏳ Warming up ({} runs)...", WARMUP_RUNS);
    
    // Warmup both paths
    for _ in 0..WARMUP_RUNS {
        let mut job = bundle.dispatch(rnn_info.clone())?;
        job.load(&chunk)?;
        job.submit();
        let _ = job.back().await?;
    }
    
    for _ in 0..WARMUP_RUNS {
        let mut job = bundle.dispatch(rnn_info.clone())?;
        job.load(&chunk)?;
        job.submit();
        let _ = job.back_sampled().await?;
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking back() - reads full logits ({:.0} KB)", info.num_vocab as f64 * 4.0 / 1024.0);
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut back_times = Vec::with_capacity(BENCH_RUNS);
    
    for i in 0..BENCH_RUNS {
        let mut job = bundle.dispatch(rnn_info.clone())?;
        job.load(&chunk)?;
        job.submit();
        
        let start = Instant::now();
        let output = job.back().await?;
        let elapsed = start.elapsed();
        
        back_times.push(elapsed);
        
        if i < 5 {
            println!("   Run {}: {:.3} ms, output size: {} floats", 
                     i + 1, 
                     elapsed.as_secs_f64() * 1000.0,
                     output.0[0].0.size());
        }
    }
    
    let back_avg = back_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / back_times.len() as f64;
    println!("   ...");
    println!("   Average: {:.3} ms", back_avg * 1000.0);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking back_sampled() - reads only token ID (4 bytes)");
    println!("═══════════════════════════════════════════════════════════════");
    
    let mut sampled_times = Vec::with_capacity(BENCH_RUNS);
    let mut sampled_tokens = Vec::with_capacity(BENCH_RUNS);
    
    for i in 0..BENCH_RUNS {
        let mut job = bundle.dispatch(rnn_info.clone())?;
        job.load(&chunk)?;
        job.submit();
        
        let start = Instant::now();
        let tokens = job.back_sampled().await?;
        let elapsed = start.elapsed();
        
        sampled_times.push(elapsed);
        sampled_tokens.push(tokens[0]);
        
        if i < 5 {
            println!("   Run {}: {:.3} ms, token: {}", 
                     i + 1, 
                     elapsed.as_secs_f64() * 1000.0,
                     tokens[0]);
        }
    }
    
    let sampled_avg = sampled_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / sampled_times.len() as f64;
    println!("   ...");
    println!("   Average: {:.3} ms", sampled_avg * 1000.0);

    // Calculate data transfer savings
    let full_readback_bytes = info.num_vocab * 4; // f32 per vocab
    let sampled_readback_bytes = 4; // single u32
    let savings_ratio = full_readback_bytes as f64 / sampled_readback_bytes as f64;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  back() avg:          {:>8.3} ms                           ║", back_avg * 1000.0);
    println!("║  back_sampled() avg:  {:>8.3} ms                           ║", sampled_avg * 1000.0);
    println!("║  Speedup:             {:>8.2}x                              ║", back_avg / sampled_avg);
    println!("║  Data transfer saved: {:>8.0}x ({} KB → 4 bytes)        ║", 
             savings_ratio, info.num_vocab * 4 / 1024);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Estimate impact on generation speed
    let baseline_gen_tps = 54.11; // From baseline benchmark
    let time_per_token_ms = 1000.0 / baseline_gen_tps;
    let readback_savings_ms = (back_avg - sampled_avg) * 1000.0;
    let new_time_per_token_ms = time_per_token_ms - readback_savings_ms;
    let estimated_new_tps = 1000.0 / new_time_per_token_ms;
    
    println!();
    println!("📊 Estimated impact on generation speed:");
    println!("   Current: {:.2} tok/s ({:.2} ms/token)", baseline_gen_tps, time_per_token_ms);
    println!("   Readback savings: {:.3} ms/token", readback_savings_ms);
    println!("   Estimated new: {:.2} tok/s ({:.2} ms/token)", estimated_new_tps, new_time_per_token_ms);
    println!("   Potential improvement: {:.1}%", (estimated_new_tps / baseline_gen_tps - 1.0) * 100.0);

    Ok(())
}

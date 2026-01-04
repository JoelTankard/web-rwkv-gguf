//! Benchmark individual phases of the inference pipeline.
//!
//! Measures: dispatch (job building), load (embedding), submit, back (readback)

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
const BENCH_RUNS: usize = 100;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark pipeline phases")]
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
    println!("║           Pipeline Phase Benchmark                           ║");
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

    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());
    let model = builder.build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);

    // Create input
    fastrand::seed(42);
    let token: u16 = fastrand::u16(0..((info.num_vocab - 1) as u16));
    let batch = RnnInputBatch::new(vec![token], RnnOption::Last);
    let rnn_input = RnnInput::new(vec![batch], 32);
    let rnn_info = rnn_input.iter().next().unwrap();
    let chunk = rnn_input.chunk();

    println!("⏳ Warming up ({} runs)...", WARMUP_RUNS);
    for _ in 0..WARMUP_RUNS {
        let mut job = bundle.dispatch(rnn_info.clone())?;
        job.load(&chunk)?;
        job.submit();
        let _ = job.back().await?;
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Benchmarking individual phases ({} runs)", BENCH_RUNS);
    println!("═══════════════════════════════════════════════════════════════");

    let mut dispatch_times = Vec::with_capacity(BENCH_RUNS);
    let mut load_times = Vec::with_capacity(BENCH_RUNS);
    let mut submit_times = Vec::with_capacity(BENCH_RUNS);
    let mut back_times = Vec::with_capacity(BENCH_RUNS);
    let mut total_times = Vec::with_capacity(BENCH_RUNS);

    for _ in 0..BENCH_RUNS {
        let total_start = Instant::now();

        // Phase 1: Dispatch (build job/command buffers)
        let dispatch_start = Instant::now();
        let mut job = bundle.dispatch(rnn_info.clone())?;
        let dispatch_elapsed = dispatch_start.elapsed();
        dispatch_times.push(dispatch_elapsed);

        // Phase 2: Load (embedding lookup, CPU→GPU)
        let load_start = Instant::now();
        job.load(&chunk)?;
        let load_elapsed = load_start.elapsed();
        load_times.push(load_elapsed);

        // Phase 3: Submit (queue GPU commands)
        let submit_start = Instant::now();
        job.submit();
        let submit_elapsed = submit_start.elapsed();
        submit_times.push(submit_elapsed);

        // Phase 4: Back (wait for GPU, read results)
        let back_start = Instant::now();
        let _ = job.back().await?;
        let back_elapsed = back_start.elapsed();
        back_times.push(back_elapsed);

        let total_elapsed = total_start.elapsed();
        total_times.push(total_elapsed);
    }

    let dispatch_avg = dispatch_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let load_avg = load_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let submit_avg = submit_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let back_avg = back_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let total_avg = total_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    PHASE BREAKDOWN                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  dispatch():  {:>8.3} ms  ({:>5.1}%)  - build commands       ║", 
             dispatch_avg * 1000.0, dispatch_avg / total_avg * 100.0);
    println!("║  load():      {:>8.3} ms  ({:>5.1}%)  - embed lookup         ║", 
             load_avg * 1000.0, load_avg / total_avg * 100.0);
    println!("║  submit():    {:>8.3} ms  ({:>5.1}%)  - queue to GPU         ║", 
             submit_avg * 1000.0, submit_avg / total_avg * 100.0);
    println!("║  back():      {:>8.3} ms  ({:>5.1}%)  - GPU exec + readback  ║", 
             back_avg * 1000.0, back_avg / total_avg * 100.0);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL:       {:>8.3} ms  (100.0%)                          ║", total_avg * 1000.0);
    println!("║  Throughput:  {:>8.2} tok/s                                 ║", 1.0 / total_avg);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Identify optimization opportunities
    println!();
    println!("📊 Optimization opportunities:");
    
    let cpu_time = dispatch_avg + load_avg + submit_avg;
    let gpu_time = back_avg;
    
    println!("   CPU phases (dispatch+load+submit): {:.3} ms ({:.1}%)", 
             cpu_time * 1000.0, cpu_time / total_avg * 100.0);
    println!("   GPU phase (back): {:.3} ms ({:.1}%)", 
             gpu_time * 1000.0, gpu_time / total_avg * 100.0);
    
    if cpu_time > gpu_time * 0.1 {
        println!();
        println!("   💡 CPU phases are significant ({:.1}% of total).", cpu_time / total_avg * 100.0);
        println!("      Triple buffering could hide {:.3} ms/token.", cpu_time * 1000.0);
        println!("      Potential speedup: {:.1}%", cpu_time / total_avg * 100.0);
    }

    Ok(())
}

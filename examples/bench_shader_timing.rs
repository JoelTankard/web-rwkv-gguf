//! Benchmark individual shader execution times.
//!
//! Profiles the time spent in each type of operation during inference.

use std::{collections::HashMap, path::PathBuf, time::Duration};

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

const WARMUP_RUNS: usize = 5;
const BENCH_RUNS: usize = 20;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark shader timing")]
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
    println!("║              Shader Timing Benchmark                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?}", args.model.file_name().unwrap());
    println!("   Version: {:?}", info.version);
    println!("   Layers: {}", info.num_layer);
    println!("   Embedding: {}", info.num_emb);
    println!("   Heads: {}", info.num_head);
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
    println!("  Profiling dispatch + submit + back ({} runs)", BENCH_RUNS);
    println!("═══════════════════════════════════════════════════════════════");

    let mut dispatch_times = Vec::with_capacity(BENCH_RUNS);
    let mut submit_times = Vec::with_capacity(BENCH_RUNS);
    let mut gpu_times = Vec::with_capacity(BENCH_RUNS);

    for _ in 0..BENCH_RUNS {
        // Measure dispatch (command buffer building)
        let dispatch_start = Instant::now();
        let mut job = bundle.dispatch(rnn_info.clone())?;
        let dispatch_time = dispatch_start.elapsed();
        dispatch_times.push(dispatch_time);

        // Load embeddings
        job.load(&chunk)?;

        // Measure submit (queue to GPU)
        let submit_start = Instant::now();
        job.submit();
        let submit_time = submit_start.elapsed();
        submit_times.push(submit_time);

        // Measure GPU execution + readback
        let gpu_start = Instant::now();
        let _ = job.back().await?;
        let gpu_time = gpu_start.elapsed();
        gpu_times.push(gpu_time);
    }

    let dispatch_avg = dispatch_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let submit_avg = submit_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let gpu_avg = gpu_times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / BENCH_RUNS as f64;
    let total_avg = dispatch_avg + submit_avg + gpu_avg;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    TIMING BREAKDOWN                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  dispatch():  {:>8.3} ms  ({:>5.1}%)  - build {} commands    ║", 
             dispatch_avg * 1000.0, dispatch_avg / total_avg * 100.0, info.num_layer);
    println!("║  submit():    {:>8.3} ms  ({:>5.1}%)  - queue to GPU         ║", 
             submit_avg * 1000.0, submit_avg / total_avg * 100.0);
    println!("║  back():      {:>8.3} ms  ({:>5.1}%)  - GPU exec + readback  ║", 
             gpu_avg * 1000.0, gpu_avg / total_avg * 100.0);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  TOTAL:       {:>8.3} ms  (100.0%)                          ║", total_avg * 1000.0);
    println!("║  Throughput:  {:>8.2} tok/s                                 ║", 1.0 / total_avg);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Estimate per-layer time
    let per_layer_time = gpu_avg / info.num_layer as f64;
    println!();
    println!("📊 Per-layer estimate: {:.3} ms", per_layer_time * 1000.0);
    println!();

    // Estimate operation breakdown based on typical V7 layer structure
    // Per layer: ~14 matmuls, 2 layer_norm, 1 group_norm, 1 time_mix, 6 token_shift, misc ops
    let matmuls_per_layer = 14;
    let total_matmuls = matmuls_per_layer * info.num_layer;
    
    println!("📈 Estimated operation counts:");
    println!("   Total matmuls: ~{} ({} per layer × {} layers)", 
             total_matmuls, matmuls_per_layer, info.num_layer);
    println!("   Layer norms: ~{} (2 per layer)", info.num_layer * 2);
    println!("   Token shifts: ~{} (7 per layer)", info.num_layer * 7);
    println!("   Time mix ops: ~{} (1 per layer)", info.num_layer);

    // If matmuls dominate (typically 80-90% of compute)
    let estimated_matmul_time = gpu_avg * 0.85;
    let per_matmul_time = estimated_matmul_time / total_matmuls as f64;
    println!();
    println!("💡 If matmuls are 85% of GPU time:");
    println!("   Estimated per-matmul: {:.3} µs", per_matmul_time * 1_000_000.0);
    println!("   Total matmul time: {:.3} ms", estimated_matmul_time * 1000.0);

    // Suggest optimizations
    println!();
    println!("🔧 Optimization opportunities:");
    if dispatch_avg > gpu_avg * 0.1 {
        println!("   ⚠️  Dispatch time is significant ({:.1}% of total)", dispatch_avg / total_avg * 100.0);
        println!("      Consider: command buffer caching, fewer dispatches");
    }
    
    println!("   📌 GPU time dominates ({:.1}% of total)", gpu_avg / total_avg * 100.0);
    println!("      Focus on: matmul optimization, fused operations");
    println!();
    println!("   Potential improvements:");
    println!("   1. Fused attention (combine 6 token_shift + matmuls)");
    println!("   2. Subgroup operations for reductions");
    println!("   3. Workgroup size tuning for Apple Silicon");
    println!("   4. Memory coalescing in matmul shaders");

    Ok(())
}

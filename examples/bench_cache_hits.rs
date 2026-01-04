//! Benchmark to measure bind group cache effectiveness.
//!
//! Runs multiple inferences and measures if dispatch time decreases
//! as the cache warms up.

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

const BENCH_RUNS: usize = 50;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark cache hit rates")]
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
    println!("║              Cache Hit Rate Benchmark                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    let file = TokioFile::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    println!("📦 Model: {:?}", args.model.file_name().unwrap());
    println!("   Layers: {}", info.num_layer);
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

    // Create input - same token each time for consistent cache behavior
    let token: u16 = 1000;
    let batch = RnnInputBatch::new(vec![token], RnnOption::Last);
    let rnn_input = RnnInput::new(vec![batch], 32);
    let rnn_info = rnn_input.iter().next().unwrap();
    let chunk = rnn_input.chunk();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Measuring dispatch time over {} runs (same input)", BENCH_RUNS);
    println!("  If cache is working, dispatch time should decrease after run 1");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut dispatch_times = Vec::with_capacity(BENCH_RUNS);
    let mut total_times = Vec::with_capacity(BENCH_RUNS);

    for i in 0..BENCH_RUNS {
        let total_start = Instant::now();

        // Measure dispatch
        let dispatch_start = Instant::now();
        let mut job = bundle.dispatch(rnn_info.clone())?;
        let dispatch_time = dispatch_start.elapsed();
        dispatch_times.push(dispatch_time);

        // Run full inference
        job.load(&chunk)?;
        job.submit();
        let _ = job.back().await?;

        let total_time = total_start.elapsed();
        total_times.push(total_time);

        if i < 10 || i == BENCH_RUNS - 1 {
            println!("   Run {:>2}: dispatch {:.3} ms, total {:.3} ms", 
                     i + 1, 
                     dispatch_time.as_secs_f64() * 1000.0,
                     total_time.as_secs_f64() * 1000.0);
        } else if i == 10 {
            println!("   ...");
        }
    }

    // Calculate statistics
    let first_dispatch = dispatch_times[0].as_secs_f64() * 1000.0;
    let last_10_dispatch: f64 = dispatch_times[BENCH_RUNS-10..].iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>() / 10.0;
    
    let first_total = total_times[0].as_secs_f64() * 1000.0;
    let last_10_total: f64 = total_times[BENCH_RUNS-10..].iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>() / 10.0;

    let dispatch_improvement = (first_dispatch - last_10_dispatch) / first_dispatch * 100.0;
    let total_improvement = (first_total - last_10_total) / first_total * 100.0;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    CACHE ANALYSIS                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  First run dispatch:    {:>8.3} ms                         ║", first_dispatch);
    println!("║  Last 10 avg dispatch:  {:>8.3} ms                         ║", last_10_dispatch);
    println!("║  Dispatch improvement:  {:>8.1}%                           ║", dispatch_improvement);
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  First run total:       {:>8.3} ms                         ║", first_total);
    println!("║  Last 10 avg total:     {:>8.3} ms                         ║", last_10_total);
    println!("║  Total improvement:     {:>8.1}%                           ║", total_improvement);
    println!("╚══════════════════════════════════════════════════════════════╝");

    if dispatch_improvement > 10.0 {
        println!();
        println!("✅ Cache is working! Dispatch time reduced by {:.1}%", dispatch_improvement);
    } else if dispatch_improvement > 0.0 {
        println!();
        println!("⚠️  Minimal cache benefit ({:.1}% improvement)", dispatch_improvement);
        println!("   Possible reasons:");
        println!("   - Bind groups already cached from previous runs");
        println!("   - Cache key includes dynamic data preventing hits");
    } else {
        println!();
        println!("❌ No cache benefit detected");
        println!("   Cache may not be working as expected");
    }

    Ok(())
}

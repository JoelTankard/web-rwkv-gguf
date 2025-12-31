//! Benchmark comparing model loading time with and without per-layer GPU sync.
//!
//! This test measures the impact of removing per-layer `device.poll(Wait)` calls
//! during model loading. The optimization allows GPU buffer uploads to be batched
//! and pipelined instead of blocking after each layer.
//!
//! Usage:
//!   cargo run --release --example bench_loading -- --model /path/to/model.gguf
//!
//! To test the OLD (slow) behavior, use --sync-per-layer flag:
//!   cargo run --release --example bench_loading -- --model /path/to/model.gguf --sync-per-layer
//!
//! Note: On Apple Silicon with unified memory, the difference may be minimal or
//! even favor the per-layer sync due to better memory pressure management.
//! The optimization is more impactful on discrete GPUs with separate VRAM.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use instant::Instant;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        loader::{Loader, Reader},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
    },
};

const WARMUP_RUNS: usize = 1;
const BENCH_RUNS: usize = 3;

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark model loading time")]
struct Args {
    /// Path to model file (.gguf or .st)
    #[arg(short, long)]
    model: PathBuf,

    /// Use per-layer GPU sync (old slow behavior)
    #[arg(long, default_value = "false")]
    sync_per_layer: bool,

    /// Auto-select GPU adapter
    #[arg(long, default_value = "true")]
    auto: bool,
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

/// Load model with the OPTIMIZED path (no per-layer sync)
async fn load_model_optimized<R: Reader>(
    context: &Context,
    model: R,
    info: &ModelInfo,
) -> Result<()> {
    let builder = ModelBuilder::new(context, model).quant(HashMap::new());

    match info.version {
        ModelVersion::V4 => {
            let _ = builder.build_v4().await?;
        }
        ModelVersion::V5 => {
            let _ = builder.build_v5().await?;
        }
        ModelVersion::V6 => {
            let _ = builder.build_v6().await?;
        }
        ModelVersion::V7 => {
            let _ = builder.build_v7().await?;
        }
    }
    Ok(())
}

/// Load model with the OLD path (per-layer sync) - simulates old behavior
async fn load_model_with_sync<R: Reader>(
    context: &Context,
    model: R,
    info: &ModelInfo,
) -> Result<()> {
    // This function manually implements the old loading behavior with per-layer sync
    // to compare against the optimized version

    let loader = Loader {
        context: context.clone(),
        model,
        lora: vec![],
    };

    match info.version {
        ModelVersion::V7 => {
            load_v7_with_sync(context, &loader, info).await?;
        }
        _ => {
            // For other versions, just use the standard loader
            // The comparison will still be valid for V7 models
            println!("Note: Per-layer sync simulation only implemented for V7");
            let builder = ModelBuilder::new(&context, loader.model).quant(HashMap::new());
            match info.version {
                ModelVersion::V4 => {
                    let _ = builder.build_v4().await?;
                }
                ModelVersion::V5 => {
                    let _ = builder.build_v5().await?;
                }
                ModelVersion::V6 => {
                    let _ = builder.build_v6().await?;
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// V7 loading with per-layer GPU sync (old slow behavior)
async fn load_v7_with_sync<R: Reader>(
    context: &Context,
    loader: &Loader<R>,
    info: &ModelInfo,
) -> Result<()> {
    use web_rwkv::runtime::model::Quant;
    use web_rwkv::tensor::matrix::Matrix;

    // Load embed
    let _embed_ln_w = loader.load_vector_f16("blocks.0.ln0.weight")?;
    let _embed_ln_b = loader.load_vector_f16("blocks.0.ln0.bias")?;
    let _embed_w = loader.load_matrix_f16_padded_cpu("emb.weight")?;

    // Load head
    let _head_ln_w = loader.load_vector_f16("ln_out.weight")?;
    let _head_ln_b = loader.load_vector_f16("ln_out.bias")?;
    let _head_w = Matrix::Fp16(loader.load_matrix_f16_padded("head.weight")?);

    // Initial sync after embed/head (this is kept in both versions)
    let submission_index = Some(context.queue.submit(None));
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index,
        timeout: None,
    });

    // Load layers WITH per-layer sync (old behavior)
    for layer in 0..info.num_layer {
        let att = format!("blocks.{layer}.att");
        let ffn = format!("blocks.{layer}.ffn");

        // Load attention layer norm
        let _att_ln_w = loader.load_vector_f16(format!("blocks.{layer}.ln1.weight"))?;
        let _att_ln_b = loader.load_vector_f16(format!("blocks.{layer}.ln1.bias"))?;

        // Load attention vectors
        let _x_r = loader.load_vector_f16(format!("{att}.x_r"))?;
        let _x_w = loader.load_vector_f16(format!("{att}.x_w"))?;
        let _x_k = loader.load_vector_f16(format!("{att}.x_k"))?;
        let _x_v = loader.load_vector_f16(format!("{att}.x_v"))?;
        let _x_a = loader.load_vector_f16(format!("{att}.x_a"))?;
        let _x_g = loader.load_vector_f16(format!("{att}.x_g"))?;

        let _w0 = loader.load_vector_f16(format!("{att}.w0"))?;
        let _a0 = loader.load_vector_f16(format!("{att}.a0"))?;

        // Load attention matrices
        let _w1 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.w1"))?);
        let _w2 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.w2"))?);
        let _a1 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.a1"))?);
        let _a2 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.a2"))?);
        let _g1 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.g1"))?);
        let _g2 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.g2"))?);

        if layer > 0 {
            let _v0 = loader.load_vector_f16(format!("{att}.v0"))?;
            let _v1 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.v1"))?);
            let _v2 = Matrix::Fp16(loader.load_matrix_f16(format!("{att}.v2"))?);
        }

        let _r_k = loader.load_matrix_f16(format!("{att}.r_k"))?;
        let _k_k = loader.load_vector_f16(format!("{att}.k_k"))?;
        let _k_a = loader.load_vector_f16(format!("{att}.k_a"))?;

        let _gn_w = loader.load_vector_f16(format!("{att}.ln_x.weight"))?;
        let _gn_b = loader.load_vector_f16(format!("{att}.ln_x.bias"))?;

        // Load main attention weights
        let _w_k = loader.load_matrix(format!("{att}.key.weight"), Quant::None)?;
        let _w_v = loader.load_matrix(format!("{att}.value.weight"), Quant::None)?;
        let _w_r = loader.load_matrix(format!("{att}.receptance.weight"), Quant::None)?;
        let _w_o = loader.load_matrix(format!("{att}.output.weight"), Quant::None)?;

        // Load FFN layer norm
        let _ffn_ln_w = loader.load_vector_f16(format!("blocks.{layer}.ln2.weight"))?;
        let _ffn_ln_b = loader.load_vector_f16(format!("blocks.{layer}.ln2.bias"))?;

        // Load FFN
        let _ffn_x_k = loader.load_vector_f16(format!("{ffn}.x_k"))?;
        let _ffn_w_k = loader.load_matrix(format!("{ffn}.key.weight"), Quant::None)?;
        let _ffn_w_v = loader.load_matrix(format!("{ffn}.value.weight"), Quant::None)?;

        // OLD BEHAVIOR: Sync after EVERY layer - this is what we removed!
        let submission_index = Some(context.queue.submit(None));
        let _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });
    }

    // Final sync
    let submission_index = Some(context.queue.submit(None));
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index,
        timeout: None,
    });

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Model Loading Benchmark - Sync Optimization          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let is_gguf = args.model.extension().map_or(false, |ext| ext == "gguf");

    // Load file data
    let file = File::open(&args.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    // Get model info
    let info = if is_gguf {
        let model = GgufReader::new(&data)?;
        Loader::info(&model)?
    } else {
        let model = SafeTensors::deserialize(&data)?;
        Loader::info(&model)?
    };

    println!("📦 Model: {:?}", args.model.file_name().unwrap_or_default());
    println!("   Version: {:?}", info.version);
    println!("   Layers: {}", info.num_layer);
    println!("   Embedding: {}", info.num_emb);
    println!("   Vocab: {}", info.num_vocab);
    println!();

    // Create context
    let context = create_context(&info).await?;
    println!("🖥️  GPU: {}", context.adapter.get_info().name);
    println!();

    if args.sync_per_layer {
        println!("🔴 Testing OLD behavior (with per-layer GPU sync)");
        println!("   This simulates the slow loading path before optimization.");
    } else {
        println!("🟢 Testing OPTIMIZED behavior (batched GPU uploads)");
        println!("   This is the new fast loading path.");
    }
    println!();

    // Warmup
    println!("⏳ Warming up ({} runs)...", WARMUP_RUNS);
    for _ in 0..WARMUP_RUNS {
        if is_gguf {
            let model = GgufReader::new(&data)?;
            if args.sync_per_layer {
                load_model_with_sync(&context, model, &info).await?;
            } else {
                load_model_optimized(&context, model, &info).await?;
            }
        } else {
            let model = SafeTensors::deserialize(&data)?;
            if args.sync_per_layer {
                load_model_with_sync(&context, model, &info).await?;
            } else {
                load_model_optimized(&context, model, &info).await?;
            }
        }
    }

    // Benchmark
    println!("📊 Benchmarking ({} runs)...", BENCH_RUNS);
    let mut times = Vec::with_capacity(BENCH_RUNS);

    for i in 0..BENCH_RUNS {
        let start = Instant::now();

        if is_gguf {
            let model = GgufReader::new(&data)?;
            if args.sync_per_layer {
                load_model_with_sync(&context, model, &info).await?;
            } else {
                load_model_optimized(&context, model, &info).await?;
            }
        } else {
            let model = SafeTensors::deserialize(&data)?;
            if args.sync_per_layer {
                load_model_with_sync(&context, model, &info).await?;
            } else {
                load_model_optimized(&context, model, &info).await?;
            }
        }

        let elapsed = start.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to ms
        println!("   Run {}: {:.2} ms", i + 1, times[i]);
    }

    // Calculate statistics
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Mode: {:<54}║",
        if args.sync_per_layer {
            "Per-layer sync (OLD)"
        } else {
            "Batched (OPTIMIZED)"
        }
    );
    println!(
        "║  Average: {:>8.2} ms                                       ║",
        avg
    );
    println!(
        "║  Min:     {:>8.2} ms                                       ║",
        min
    );
    println!(
        "║  Max:     {:>8.2} ms                                       ║",
        max
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    if !args.sync_per_layer {
        println!("💡 To compare with the old (slow) behavior, run with --sync-per-layer");
    } else {
        println!("💡 To see the optimized (fast) behavior, run without --sync-per-layer");
    }

    Ok(())
}

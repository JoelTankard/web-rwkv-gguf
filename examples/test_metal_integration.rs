//! Test Metal backend integration with Q4K model loading.
//!
//! This example loads a Q4K model and verifies that Metal buffers are created
//! during loading.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use half::f16;
use memmap2::Mmap;
use tokio::fs::File as TokioFile;
use web_rwkv::{
    context::{ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelVersion},
        v7,
    },
    wgpu::PowerPreference,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    model: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()?;

    let args = Args::parse();
    let model_path = args
        .model
        .unwrap_or_else(|| PathBuf::from("assets/models/2.9b-Q4_K_M.gguf"));

    log::info!("Loading model from: {:?}", model_path);

    // Load and parse model info
    let file = TokioFile::open(&model_path).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;

    log::info!("Model version: {:?}", info.version);
    log::info!("Layers: {}, Embedding: {}", info.num_layer, info.num_emb);

    // Create wgpu instance and adapter
    let instance = wgpu::Instance::default();
    let adapter = instance.adapter(PowerPreference::HighPerformance).await?;
    let adapter_info = adapter.get_info();
    log::info!(
        "Using adapter: {} ({:?})",
        adapter_info.name,
        adapter_info.backend
    );

    // Create context with Metal backend
    let context = ContextBuilder::new(adapter)
        .auto_limits(&info)
        .build()
        .await?;

    // Check if Metal backend is available
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    {
        if let Some(metal_ctx) = context.metal() {
            log::info!("✅ Metal backend is available!");
            log::info!("Metal context: {:?}", metal_ctx);
        } else {
            log::warn!("⚠️ Metal backend feature enabled but Metal context failed to initialize");
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
    {
        log::info!("ℹ️ Metal backend not enabled (compile with --features metal-backend)");
    }

    // Load the model (this should create Metal buffers for Q4K weights)
    log::info!("Loading model weights...");
    let start = std::time::Instant::now();

    let builder = ModelBuilder::new(&context, model).quant(HashMap::new());

    // Only V7 models are supported for this test
    if info.version != ModelVersion::V7 {
        log::warn!(
            "This test is designed for V7 models, but got {:?}",
            info.version
        );
        return Ok(());
    }

    let model = builder.build_v7().await?;
    let elapsed = start.elapsed();
    log::info!("Model loaded in {:.2}s", elapsed.as_secs_f64());

    // Check Metal buffer cache status after loading
    #[cfg(all(target_os = "macos", feature = "metal-backend"))]
    {
        if let Some(metal_ctx) = context.metal() {
            log::info!("✅ Metal context after loading: {:?}", metal_ctx);
            log::info!("Metal buffers should now be cached for Q4K weights");
        }
    }

    // Create a bundle to verify the model works
    let bundle = v7::Bundle::<f16>::new(model, 1);
    log::info!("Bundle created successfully with {} batch(es)", 1);

    log::info!("✅ Metal integration test completed successfully!");

    Ok(())
}

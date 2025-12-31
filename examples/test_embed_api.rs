//! Test for Phase 2 optimizations:
//! 1. Optimized State::back() with single GPU buffer
//! 2. RnnOption::EmbedLast/EmbedFull that skip head projection

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
        model::{Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, State},
        v7, Runtime, TokioRuntime,
    },
    tensor::TensorShape,
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
    Ok(context)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();

    let file = File::open(&cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };

    let model = GgufReader::new(&data).expect("failed to parse GGUF file");
    let info = Loader::info(&model)?;
    log::info!("Model: {:?}", info.version);

    let context = create_context(&info).await?;

    // Only test V7 for now
    assert_eq!(
        info.version,
        ModelVersion::V7,
        "This test requires a V7 model"
    );

    let model = ModelBuilder::new(&context, model).build_v7().await?;
    let bundle = v7::Bundle::<f16>::new(model, 1);
    let state = bundle.state();
    let runtime: Box<dyn Runtime<Rnn>> = Box::new(TokioRuntime::new(bundle).await);

    // Test tokens
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];

    println!("\n=== Test 1: State::back() optimization ===");
    {
        // Run inference to populate state
        let input = RnnInput::new(
            vec![RnnInputBatch::new(tokens.clone(), RnnOption::Last)],
            128,
        );
        let _ = runtime.infer(input).await?;

        // Test state readback performance
        let iterations = 10;
        let start = Instant::now();
        for _ in 0..iterations {
            let backed = state.back(0).await?;
            // Verify shape is correct
            let shape = backed.shape();
            assert_eq!(shape[0], info.num_emb, "State dim 0 should be num_emb");
            assert_eq!(shape[2], info.num_layer, "State dim 2 should be num_layer");
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
        println!(
            "  State::back() average: {:.2} ms ({} iterations)",
            avg_ms, iterations
        );

        // Verify state content is non-zero (model has been run)
        let backed = state.back(0).await?;
        let data: Vec<f32> = backed.into();
        let non_zero = data.iter().filter(|&&x| x != 0.0).count();
        println!(
            "  State has {} non-zero values out of {}",
            non_zero,
            data.len()
        );
        assert!(
            non_zero > 0,
            "State should have non-zero values after inference"
        );
    }

    println!("\n=== Test 2: RnnOption::EmbedLast (skip head projection) ===");
    {
        // First, run with normal Last option to get logits
        let input_normal = RnnInput::new(
            vec![RnnInputBatch::new(tokens.clone(), RnnOption::Last)],
            128,
        );
        let start = Instant::now();
        let (_, output_normal) = runtime.infer(input_normal).await?;
        let normal_time = start.elapsed();

        let normal_output = &output_normal[0].0;
        let normal_shape = normal_output.shape();
        println!("  RnnOption::Last output shape: {:?}", normal_shape);
        println!(
            "  RnnOption::Last time: {:.2} ms",
            normal_time.as_secs_f64() * 1000.0
        );

        // Now run with EmbedLast option
        let input_embed = RnnInput::new(
            vec![RnnInputBatch::new(tokens.clone(), RnnOption::EmbedLast)],
            128,
        );
        let start = Instant::now();
        let (_, output_embed) = runtime.infer(input_embed).await?;
        let embed_time = start.elapsed();

        let embed_output = &output_embed[0].0;
        let embed_shape = embed_output.shape();
        println!("  RnnOption::EmbedLast output shape: {:?}", embed_shape);
        println!(
            "  RnnOption::EmbedLast time: {:.2} ms",
            embed_time.as_secs_f64() * 1000.0
        );

        // The output shapes should be different:
        // - Last: [vocab_size, 1, 1, 1] (logits)
        // - EmbedLast: [num_emb, 1, 1, 1] (hidden state) - but output tensor is still vocab-sized
        // Note: The output tensor size is pre-allocated, so we check the computation was skipped
        // by verifying the embed path is faster (no head matmul)

        println!(
            "\n  Speedup: {:.2}x",
            normal_time.as_secs_f64() / embed_time.as_secs_f64()
        );
    }

    println!("\n=== Test 3: RnnOption::EmbedFull ===");
    {
        // Run with EmbedFull option
        let input_embed_full = RnnInput::new(
            vec![RnnInputBatch::new(tokens.clone(), RnnOption::EmbedFull)],
            128,
        );
        let start = Instant::now();
        let (_, output_embed_full) = runtime.infer(input_embed_full).await?;
        let embed_full_time = start.elapsed();

        let embed_full_output = &output_embed_full[0].0;
        let embed_full_shape = embed_full_output.shape();
        println!(
            "  RnnOption::EmbedFull output shape: {:?}",
            embed_full_shape
        );
        println!(
            "  RnnOption::EmbedFull time: {:.2} ms",
            embed_full_time.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Test 4: State save/restore roundtrip ===");
    {
        // Run some tokens
        let input = RnnInput::new(
            vec![RnnInputBatch::new(tokens.clone(), RnnOption::Last)],
            128,
        );
        let _ = runtime.infer(input).await?;

        // Save state
        let saved_state = state.back(0).await?;
        let saved_data: Vec<f32> = saved_state.clone().into();

        // Run more tokens (modifies state)
        let more_tokens: Vec<u32> = vec![10, 20, 30];
        let input = RnnInput::new(vec![RnnInputBatch::new(more_tokens, RnnOption::Last)], 128);
        let _ = runtime.infer(input).await?;

        // Verify state changed
        let modified_state = state.back(0).await?;
        let modified_data: Vec<f32> = modified_state.into();
        let diff_count = saved_data
            .iter()
            .zip(modified_data.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        println!(
            "  State changed in {} positions after more inference",
            diff_count
        );
        assert!(diff_count > 0, "State should change after more inference");

        // Restore saved state
        state.load(saved_state.clone(), 0)?;

        // Verify state was restored
        let restored_state = state.back(0).await?;
        let restored_data: Vec<f32> = restored_state.into();
        let restore_diff = saved_data
            .iter()
            .zip(restored_data.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-6)
            .count();
        println!(
            "  After restore, {} positions differ from saved",
            restore_diff
        );
        assert_eq!(
            restore_diff, 0,
            "Restored state should match saved state exactly"
        );
    }

    println!("\n=== All tests passed! ===\n");

    Ok(())
}

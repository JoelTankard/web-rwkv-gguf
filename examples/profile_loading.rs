//! Profile model loading to identify exact bottlenecks.
//!
//! Run with: cargo run --release --example profile_loading -- -m path/to/model.gguf

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use memmap2::Mmap;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        loader::{Loader, Reader},
        model::{ContextAutoLimits, ModelInfo, ModelVersion, Quant},
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
}

#[derive(Default)]
struct LoadingProfile {
    file_open: std::time::Duration,
    mmap: std::time::Duration,
    parse_header: std::time::Duration,
    context_create: std::time::Duration,
    embed_ln: std::time::Duration,
    embed_weight: std::time::Duration,
    head_ln: std::time::Duration,
    head_weight: std::time::Duration,
    gpu_sync_1: std::time::Duration,
    layers: Vec<LayerProfile>,
    total: std::time::Duration,
}

#[derive(Default, Clone)]
struct LayerProfile {
    att_ln: std::time::Duration,
    att_vectors: std::time::Duration,
    att_small_matrices: std::time::Duration,
    att_large_matrices: std::time::Duration,
    ffn_ln: std::time::Duration,
    ffn_vectors: std::time::Duration,
    ffn_matrices: std::time::Duration,
}

impl LoadingProfile {
    fn print_summary(&self) {
        let total_ms = self.total.as_secs_f64() * 1000.0;

        println!("\n=== Model Loading Profile ===\n");
        println!("Phase                          Time (ms)    % of Total");
        println!("─────────────────────────────────────────────────────────");

        self.print_phase("File open", self.file_open, total_ms);
        self.print_phase("Memory map", self.mmap, total_ms);
        self.print_phase("Parse header", self.parse_header, total_ms);
        self.print_phase("Context create", self.context_create, total_ms);
        self.print_phase("Embed layer norm", self.embed_ln, total_ms);
        self.print_phase("Embed weight (CPU)", self.embed_weight, total_ms);
        self.print_phase("Head layer norm", self.head_ln, total_ms);
        self.print_phase("Head weight (GPU)", self.head_weight, total_ms);
        self.print_phase("GPU sync #1", self.gpu_sync_1, total_ms);

        // Aggregate layer stats
        let num_layers = self.layers.len();
        let mut layer_totals = LayerProfile::default();
        for layer in &self.layers {
            layer_totals.att_ln += layer.att_ln;
            layer_totals.att_vectors += layer.att_vectors;
            layer_totals.att_small_matrices += layer.att_small_matrices;
            layer_totals.att_large_matrices += layer.att_large_matrices;
            layer_totals.ffn_ln += layer.ffn_ln;
            layer_totals.ffn_vectors += layer.ffn_vectors;
            layer_totals.ffn_matrices += layer.ffn_matrices;
        }

        let layer_total = layer_totals.att_ln
            + layer_totals.att_vectors
            + layer_totals.att_small_matrices
            + layer_totals.att_large_matrices
            + layer_totals.ffn_ln
            + layer_totals.ffn_vectors
            + layer_totals.ffn_matrices;

        println!("─────────────────────────────────────────────────────────");
        println!("Layers ({} total):", num_layers);
        self.print_phase("  ATT layer norms", layer_totals.att_ln, total_ms);
        self.print_phase(
            "  ATT vectors (x_r, x_w, etc)",
            layer_totals.att_vectors,
            total_ms,
        );
        self.print_phase(
            "  ATT small matrices (w1,w2,a1,a2,g1,g2,v1,v2)",
            layer_totals.att_small_matrices,
            total_ms,
        );
        self.print_phase(
            "  ATT large matrices (key,value,receptance,output)",
            layer_totals.att_large_matrices,
            total_ms,
        );
        self.print_phase("  FFN layer norms", layer_totals.ffn_ln, total_ms);
        self.print_phase("  FFN vectors (x_k)", layer_totals.ffn_vectors, total_ms);
        self.print_phase(
            "  FFN matrices (key,value)",
            layer_totals.ffn_matrices,
            total_ms,
        );
        self.print_phase("  Layer subtotal", layer_total, total_ms);

        println!("─────────────────────────────────────────────────────────");
        self.print_phase("TOTAL", self.total, total_ms);

        // Per-layer breakdown for first few layers
        println!("\n=== Per-Layer Breakdown (first 3 layers) ===\n");
        for (i, layer) in self.layers.iter().take(3).enumerate() {
            let layer_total = layer.att_ln
                + layer.att_vectors
                + layer.att_small_matrices
                + layer.att_large_matrices
                + layer.ffn_ln
                + layer.ffn_vectors
                + layer.ffn_matrices;
            println!("Layer {}: {:.1}ms", i, layer_total.as_secs_f64() * 1000.0);
            println!("  ATT LN: {:.1}ms, ATT vec: {:.1}ms, ATT small mat: {:.1}ms, ATT large mat: {:.1}ms",
                layer.att_ln.as_secs_f64() * 1000.0,
                layer.att_vectors.as_secs_f64() * 1000.0,
                layer.att_small_matrices.as_secs_f64() * 1000.0,
                layer.att_large_matrices.as_secs_f64() * 1000.0);
            println!(
                "  FFN LN: {:.1}ms, FFN vec: {:.1}ms, FFN mat: {:.1}ms",
                layer.ffn_ln.as_secs_f64() * 1000.0,
                layer.ffn_vectors.as_secs_f64() * 1000.0,
                layer.ffn_matrices.as_secs_f64() * 1000.0
            );
        }

        // Bottleneck analysis
        println!("\n=== Bottleneck Analysis ===\n");

        let embed_total = self.embed_ln + self.embed_weight;
        let head_total = self.head_ln + self.head_weight;
        let overhead =
            self.file_open + self.mmap + self.parse_header + self.context_create + self.gpu_sync_1;

        let mut phases: Vec<(&str, std::time::Duration)> = vec![
            ("Overhead (file, mmap, parse, context, sync)", overhead),
            ("Embed (LN + weight)", embed_total),
            ("Head (LN + weight)", head_total),
            ("ATT large matrices", layer_totals.att_large_matrices),
            ("ATT small matrices", layer_totals.att_small_matrices),
            ("ATT vectors", layer_totals.att_vectors),
            ("ATT layer norms", layer_totals.att_ln),
            ("FFN matrices", layer_totals.ffn_matrices),
            (
                "FFN vectors + LN",
                layer_totals.ffn_vectors + layer_totals.ffn_ln,
            ),
        ];
        phases.sort_by(|a, b| b.1.cmp(&a.1));

        println!("Top bottlenecks (sorted by time):");
        for (name, duration) in phases.iter().take(5) {
            let ms = duration.as_secs_f64() * 1000.0;
            let pct = ms / total_ms * 100.0;
            println!("  {}: {:.1}ms ({:.1}%)", name, ms, pct);
        }
    }

    fn print_phase(&self, name: &str, duration: std::time::Duration, total_ms: f64) {
        let ms = duration.as_secs_f64() * 1000.0;
        let pct = ms / total_ms * 100.0;
        println!("{:<35} {:>8.1}    {:>6.1}%", name, ms, pct);
    }
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
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .init()?;

    let cli = Cli::parse();
    let mut profile = LoadingProfile::default();

    let total_start = Instant::now();

    // Phase 1: File open
    let start = Instant::now();
    let file = File::open(&cli.model).await?;
    profile.file_open = start.elapsed();

    // Phase 2: Memory map
    let start = Instant::now();
    let data = unsafe { Mmap::map(&file)? };
    profile.mmap = start.elapsed();

    // Phase 3: Parse header
    let start = Instant::now();
    let model = GgufReader::new(&data)?;
    let info = Loader::info(&model)?;
    profile.parse_header = start.elapsed();

    println!("Model: {:?}", cli.model);
    println!("Version: {:?}", info.version);
    println!("Layers: {}", info.num_layer);
    println!("Embed: {}", info.num_emb);
    println!("Vocab: {}", info.num_vocab);
    println!("Heads: {}", info.num_head);

    // Phase 4: Context creation
    let start = Instant::now();
    let context = create_context(&info).await?;
    profile.context_create = start.elapsed();

    println!("Adapter: {}", context.adapter.get_info().name);

    // Now do the actual model loading with instrumentation
    // We need to replicate build_v7 logic with timing

    if info.version != ModelVersion::V7 {
        println!("This profiler currently only supports V7 models");
        return Ok(());
    }

    let quant_map: std::collections::HashMap<usize, Quant> =
        (0..cli.quant).map(|layer| (layer, Quant::Int8)).collect();

    // Check tensor types before moving model into loader
    let emb_quant = model.quantized_tensor("emb.weight");
    let head_quant = model.quantized_tensor("head.weight");
    if let Some((gguf_type, raw_data)) = emb_quant {
        println!(
            "emb.weight: quantized type={}, size={} bytes",
            gguf_type,
            raw_data.len()
        );
    } else {
        println!("emb.weight: NOT quantized (F16/F32)");
    }
    if let Some((gguf_type, raw_data)) = head_quant {
        println!(
            "head.weight: quantized type={}, size={} bytes",
            gguf_type,
            raw_data.len()
        );
    } else {
        println!("head.weight: NOT quantized (F16/F32)");
    }

    let loader = Loader {
        context: context.clone(),
        model,
        lora: vec![],
    };

    // Embed loading
    let start = Instant::now();
    let _embed_ln_w = loader.load_vector_f16("blocks.0.ln0.weight")?;
    let _embed_ln_b = loader.load_vector_f16("blocks.0.ln0.bias")?;
    profile.embed_ln = start.elapsed();

    let start = Instant::now();
    let _embed_w = loader.load_matrix_f16_padded_cpu("emb.weight")?;
    profile.embed_weight = start.elapsed();

    // Head loading
    let start = Instant::now();
    let _head_ln_w = loader.load_vector_f16("ln_out.weight")?;
    let _head_ln_b = loader.load_vector_f16("ln_out.bias")?;
    profile.head_ln = start.elapsed();

    let start = Instant::now();
    let _head_w = loader.load_matrix("head.weight".to_string(), Quant::None)?;
    profile.head_weight = start.elapsed();

    // GPU sync
    let start = Instant::now();
    let submission_index = Some(context.queue.submit(None));
    _ = context.device.poll(wgpu::PollType::Wait {
        submission_index,
        timeout: None,
    });
    profile.gpu_sync_1 = start.elapsed();

    // Layer loading
    for layer_idx in 0..info.num_layer {
        let mut layer_profile = LayerProfile::default();
        let quant = quant_map.get(&layer_idx).copied().unwrap_or_default();

        // ATT layer norm
        let start = Instant::now();
        let _att_ln_w = loader.load_vector_f16(format!("blocks.{layer_idx}.ln1.weight"))?;
        let _att_ln_b = loader.load_vector_f16(format!("blocks.{layer_idx}.ln1.bias"))?;
        layer_profile.att_ln = start.elapsed();

        // ATT vectors
        let att = format!("blocks.{layer_idx}.att");
        let start = Instant::now();
        let _x_r = loader.load_vector_f16(format!("{att}.x_r"))?;
        let _x_w = loader.load_vector_f16(format!("{att}.x_w"))?;
        let _x_k = loader.load_vector_f16(format!("{att}.x_k"))?;
        let _x_v = loader.load_vector_f16(format!("{att}.x_v"))?;
        let _x_a = loader.load_vector_f16(format!("{att}.x_a"))?;
        let _x_g = loader.load_vector_f16(format!("{att}.x_g"))?;
        let _w0 = loader.load_vector_f16(format!("{att}.w0"))?;
        let _a0 = loader.load_vector_f16(format!("{att}.a0"))?;
        let _k_k = loader.load_vector_f16(format!("{att}.k_k"))?;
        let _k_a = loader.load_vector_f16(format!("{att}.k_a"))?;
        if layer_idx > 0 {
            let _v0 = loader.load_vector_f16(format!("{att}.v0"))?;
        }
        layer_profile.att_vectors = start.elapsed();

        // ATT small matrices (w1, w2, a1, a2, g1, g2, v1, v2, r_k, ln_x)
        let start = Instant::now();
        let _w1 = loader.load_matrix_f16(format!("{att}.w1"))?;
        let _w2 = loader.load_matrix_f16(format!("{att}.w2"))?;
        let _a1 = loader.load_matrix_f16(format!("{att}.a1"))?;
        let _a2 = loader.load_matrix_f16(format!("{att}.a2"))?;
        let _g1 = loader.load_matrix_f16(format!("{att}.g1"))?;
        let _g2 = loader.load_matrix_f16(format!("{att}.g2"))?;
        if layer_idx > 0 {
            let _v1 = loader.load_matrix_f16(format!("{att}.v1"))?;
            let _v2 = loader.load_matrix_f16(format!("{att}.v2"))?;
        }
        let _r_k = loader.load_matrix_f16(format!("{att}.r_k"))?;
        let _ln_x_w = loader.load_vector_f16(format!("{att}.ln_x.weight"))?;
        let _ln_x_b = loader.load_vector_f16(format!("{att}.ln_x.bias"))?;
        layer_profile.att_small_matrices = start.elapsed();

        // ATT large matrices (key, value, receptance, output)
        let start = Instant::now();
        let _w_k = loader.load_matrix(format!("{att}.key.weight"), quant)?;
        let _w_v = loader.load_matrix(format!("{att}.value.weight"), quant)?;
        let _w_r = loader.load_matrix(format!("{att}.receptance.weight"), quant)?;
        let _w_o = loader.load_matrix(format!("{att}.output.weight"), quant)?;
        layer_profile.att_large_matrices = start.elapsed();

        // FFN layer norm
        let start = Instant::now();
        let _ffn_ln_w = loader.load_vector_f16(format!("blocks.{layer_idx}.ln2.weight"))?;
        let _ffn_ln_b = loader.load_vector_f16(format!("blocks.{layer_idx}.ln2.bias"))?;
        layer_profile.ffn_ln = start.elapsed();

        // FFN vectors
        let ffn = format!("blocks.{layer_idx}.ffn");
        let start = Instant::now();
        let _ffn_x_k = loader.load_vector_f16(format!("{ffn}.x_k"))?;
        layer_profile.ffn_vectors = start.elapsed();

        // FFN matrices
        let start = Instant::now();
        let _ffn_w_k = loader.load_matrix(format!("{ffn}.key.weight"), quant)?;
        let _ffn_w_v = loader.load_matrix(format!("{ffn}.value.weight"), quant)?;
        layer_profile.ffn_matrices = start.elapsed();

        profile.layers.push(layer_profile);

        if layer_idx % 8 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush()?;
        }
    }
    println!();

    profile.total = total_start.elapsed();
    profile.print_summary();

    Ok(())
}

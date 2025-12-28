use memmap2::Mmap;
use std::fs::File;
use web_rwkv::runtime::gguf::GgufReader;
use web_rwkv::runtime::loader::Reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("/Users/joel/Dev/Experimental/web-rwkv-gguf/assets/models/rwkv7-g1a-0.1b-20250728-ctx4096.f16.gguf")?;
    let data = unsafe { Mmap::map(&file)? };

    let reader = GgufReader::new(&data)?;

    println!("GGUF Version: {}", reader.version);
    println!("Tensor count: {}", reader.tensor_count);
    println!("\nAll Metadata:");
    for (key, value) in &reader.metadata {
        println!("  {}: {:?}", key, value);
    }

    println!("\nTensors:");
    for (name, info) in &reader.tensors {
        println!(
            "  {}: {:?} {:?} offset={}",
            name, info.dimensions, info.tensor_type, info.offset
        );
    }

    println!("Available tensors:");
    for name in reader.names() {
        println!("  {}", name);
    }

    println!("\nChecking V7 detection tensors:");
    let v7_tensors = [
        "blocks.0.att.x_r",
        "blocks.0.att.x_w",
        "blocks.0.att.x_k",
        "blocks.0.att.x_v",
        "blocks.0.att.x_a",
        "blocks.0.att.x_g",
        "blocks.0.att.w0",
        "blocks.0.att.w1",
        "blocks.0.att.w2",
        "blocks.0.att.a0",
        "blocks.0.att.a1",
        "blocks.0.att.a2",
        "blocks.0.att.g1",
        "blocks.0.att.g2",
        "blocks.0.att.r_k",
        "blocks.0.att.k_k",
        "blocks.0.att.k_a",
    ];

    for tensor in v7_tensors {
        println!("  {}: {}", tensor, reader.contains(tensor));
    }

    Ok(())
}

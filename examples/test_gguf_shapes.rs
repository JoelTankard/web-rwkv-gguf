use half::f16;
use safetensors::SafeTensors;
use std::path::PathBuf;
use web_rwkv::runtime::gguf::GgufReader;
use web_rwkv::runtime::loader::Reader;

fn main() -> anyhow::Result<()> {
    // Test GGUF
    let gguf_path = PathBuf::from("assets/models/rwkv7-g1a-0.1b-20250728-ctx4096.f16.gguf");
    let st_path = PathBuf::from("assets/models/rwkv7-g1a-0.1b-20250728-ctx4096.st");

    let test_tensors = [
        "emb.weight",
        "blocks.0.att.a1",
        "blocks.0.att.r_k",
        "blocks.0.ffn.key.weight",
    ];

    // SafeTensors shapes
    if st_path.exists() {
        println!("=== SafeTensors ===");
        let st_data = std::fs::read(&st_path)?;
        let st = SafeTensors::deserialize(&st_data)?;
        for name in &test_tensors {
            if let Ok(tensor) = st.tensor(name) {
                println!("{}: shape={:?}", name, tensor.shape());
            }
        }
    }

    // GGUF shapes
    if gguf_path.exists() {
        println!("\n=== GGUF (via Reader) ===");
        let data = std::fs::read(&gguf_path)?;
        let reader = GgufReader::new(&data)?;
        for name in &test_tensors {
            if reader.contains(name) {
                let shape = reader.shape(name)?;
                let (dtype, tensor_shape, _) = reader.tensor(name)?;
                println!("{}: shape()={:?}, tensor()={:?}", name, shape, tensor_shape);
            }
        }
    }

    // Compare tensors including fused time_maa slices
    let compare_tensors = [
        "blocks.0.att.a1",
        "blocks.0.att.x_r", // Virtual slice from fused tensor
        "blocks.0.att.x_w",
        "blocks.0.att.x_k",
    ];

    if st_path.exists() && gguf_path.exists() {
        let st_data = std::fs::read(&st_path)?;
        let st = SafeTensors::deserialize(&st_data)?;

        let gguf_data = std::fs::read(&gguf_path)?;
        let reader = GgufReader::new(&gguf_data)?;

        for test_tensor in &compare_tensors {
            println!("\n=== {} ===", test_tensor);

            if let Ok(st_tensor) = st.tensor(test_tensor) {
                let st_bytes = st_tensor.data();
                println!(
                    "SafeTensors shape: {:?}, size: {} bytes",
                    st_tensor.shape(),
                    st_bytes.len()
                );
                println!(
                    "SafeTensors first 16 bytes: {:02x?}",
                    &st_bytes[..16.min(st_bytes.len())]
                );

                if reader.contains(test_tensor) {
                    let (dtype, gguf_shape, gguf_bytes) = reader.tensor(test_tensor)?;
                    println!(
                        "GGUF shape: {:?}, dtype: {:?}, size: {} bytes",
                        gguf_shape,
                        dtype,
                        gguf_bytes.len()
                    );

                    // Compare actual float values
                    let st_f16: &[f16] = bytemuck::cast_slice(st_bytes);
                    let st_vals: Vec<f32> = st_f16.iter().take(4).map(|x| x.to_f32()).collect();
                    println!("SafeTensors first 4 values: {:?}", st_vals);

                    // GGUF might be F32
                    if gguf_bytes.len() == st_bytes.len() * 2 {
                        // F32 data
                        let gguf_f32: &[f32] = bytemuck::cast_slice(&gguf_bytes);
                        let gguf_vals: Vec<f32> = gguf_f32.iter().take(4).copied().collect();
                        println!("GGUF first 4 values (F32): {:?}", gguf_vals);
                    } else {
                        let gguf_f16: &[f16] = bytemuck::cast_slice(&gguf_bytes);
                        let gguf_vals: Vec<f32> =
                            gguf_f16.iter().take(4).map(|x| x.to_f32()).collect();
                        println!("GGUF first 4 values (F16): {:?}", gguf_vals);
                    }
                } else {
                    println!("GGUF: NOT FOUND");
                }
            } else {
                println!("SafeTensors: NOT FOUND");
            }
        }
    }

    Ok(())
}

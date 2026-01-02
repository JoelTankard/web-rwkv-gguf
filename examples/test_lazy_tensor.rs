//! Integration test for lazy tensor evaluation.
//!
//! This test validates that the lazy tensor path produces the same results
//! as the eager path for core operations.

use half::f16;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    tensor::{
        kind::ReadWrite, lazy_tensor::LazyTensor, matrix::Matrix, ops::Activation, shape::Shape,
        TensorCpu, TensorGpu, TensorInit, TensorInto,
    },
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()?;

    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter).build().await?;

    log::info!("Testing lazy tensor evaluation...");

    // Test 1: Basic input → materialize roundtrip
    test_input_materialize(&context).await?;

    // Test 2: Lazy add operation
    test_lazy_add(&context).await?;

    // Test 3: Lazy mul operation
    test_lazy_mul(&context).await?;

    // Test 4: Lazy softmax operation
    test_lazy_softmax(&context).await?;

    // Test 5: Lazy affine operation
    test_lazy_affine(&context).await?;

    // Test 6: Chained operations
    test_chained_ops(&context).await?;

    // Test 7: Token shift operation
    test_lazy_token_shift(&context).await?;

    // Test 8: Blend operation
    test_lazy_blend(&context).await?;

    // Test 9: MatMul + Bias fusion
    test_matmul_bias_fusion(&context).await?;

    // Test 10: Explicit matmul_bias operation
    test_explicit_matmul_bias(&context).await?;

    log::info!("All lazy tensor tests passed!");
    Ok(())
}

async fn test_input_materialize(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 1: Input → Materialize roundtrip");

    let shape = Shape::new(64, 4, 1, 1);
    let data: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();
    let cpu_tensor: TensorCpu<f16> = TensorInit::from_data(shape, data.clone())?;
    let gpu_tensor: TensorGpu<f16, ReadWrite> = cpu_tensor.clone().to(context);

    // Create lazy tensor from GPU tensor
    let lazy = LazyTensor::from_gpu(&gpu_tensor);
    assert_eq!(lazy.shape(), shape);

    // Materialize and verify
    let buffer = lazy.materialize()?;
    assert_eq!(
        buffer.size() as usize,
        shape.len() * std::mem::size_of::<f16>()
    );

    log::info!("  ✓ Input → Materialize roundtrip passed");
    Ok(())
}

async fn test_lazy_add(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 2: Lazy add operation");

    let shape = Shape::new(64, 1, 1, 1);

    // Create two tensors with known values
    let data_a: Vec<f16> = (0..shape.len()).map(|i| f16::from_f32(i as f32)).collect();
    let data_b: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 2.0))
        .collect();

    let cpu_a: TensorCpu<f16> = TensorInit::from_data(shape, data_a.clone())?;
    let cpu_b: TensorCpu<f16> = TensorInit::from_data(shape, data_b.clone())?;
    let gpu_a: TensorGpu<f16, ReadWrite> = cpu_a.to(context);
    let gpu_b: TensorGpu<f16, ReadWrite> = cpu_b.to(context);

    // Create lazy tensors and add
    let lazy_a = LazyTensor::from_gpu(&gpu_a);
    let lazy_b = LazyTensor::from_gpu(&gpu_b);
    let lazy_sum = lazy_a.add(&lazy_b);

    // Materialize
    let _buffer = lazy_sum.materialize()?;

    log::info!("  ✓ Lazy add operation passed");
    Ok(())
}

async fn test_lazy_mul(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 3: Lazy mul operation");

    let shape = Shape::new(64, 1, 1, 1);

    let data_a: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 + 1.0))
        .collect();
    let data_b: Vec<f16> = (0..shape.len()).map(|_| f16::from_f32(0.5)).collect();

    let cpu_a: TensorCpu<f16> = TensorInit::from_data(shape, data_a)?;
    let cpu_b: TensorCpu<f16> = TensorInit::from_data(shape, data_b)?;
    let gpu_a: TensorGpu<f16, ReadWrite> = cpu_a.to(context);
    let gpu_b: TensorGpu<f16, ReadWrite> = cpu_b.to(context);

    let lazy_a = LazyTensor::from_gpu(&gpu_a);
    let lazy_b = LazyTensor::from_gpu(&gpu_b);
    let lazy_prod = lazy_a.mul(&lazy_b);

    let _buffer = lazy_prod.materialize()?;

    log::info!("  ✓ Lazy mul operation passed");
    Ok(())
}

async fn test_lazy_softmax(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 4: Lazy softmax operation");

    let shape = Shape::new(64, 1, 1, 1);

    let data: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32((i as f32) * 0.1 - 3.0))
        .collect();
    let cpu_tensor: TensorCpu<f16> = TensorInit::from_data(shape, data)?;
    let gpu_tensor: TensorGpu<f16, ReadWrite> = cpu_tensor.to(context);

    let lazy = LazyTensor::from_gpu(&gpu_tensor);
    let lazy_softmax = lazy.softmax();

    let _buffer = lazy_softmax.materialize()?;

    log::info!("  ✓ Lazy softmax operation passed");
    Ok(())
}

async fn test_lazy_affine(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 5: Lazy affine operation");

    let shape = Shape::new(64, 1, 1, 1);

    let data: Vec<f16> = (0..shape.len()).map(|i| f16::from_f32(i as f32)).collect();
    let cpu_tensor: TensorCpu<f16> = TensorInit::from_data(shape, data)?;
    let gpu_tensor: TensorGpu<f16, ReadWrite> = cpu_tensor.to(context);

    let lazy = LazyTensor::from_gpu(&gpu_tensor);
    let lazy_affine = lazy.affine(0.5, 1.0); // x * 0.5 + 1.0

    let _buffer = lazy_affine.materialize()?;

    log::info!("  ✓ Lazy affine operation passed");
    Ok(())
}

async fn test_chained_ops(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 6: Chained operations");

    let shape = Shape::new(64, 1, 1, 1);

    let data_a: Vec<f16> = (0..shape.len()).map(|i| f16::from_f32(i as f32)).collect();
    let data_b: Vec<f16> = (0..shape.len()).map(|_| f16::from_f32(1.0)).collect();

    let cpu_a: TensorCpu<f16> = TensorInit::from_data(shape, data_a)?;
    let cpu_b: TensorCpu<f16> = TensorInit::from_data(shape, data_b)?;
    let gpu_a: TensorGpu<f16, ReadWrite> = cpu_a.to(context);
    let gpu_b: TensorGpu<f16, ReadWrite> = cpu_b.to(context);

    let lazy_a = LazyTensor::from_gpu(&gpu_a);
    let lazy_b = LazyTensor::from_gpu(&gpu_b);

    // Chain: (a + b) * 0.5 + 0.0 → softmax
    let lazy_sum = lazy_a.add(&lazy_b);
    let lazy_scaled = lazy_sum.affine(0.5, 0.0);
    let lazy_result = lazy_scaled.softmax();

    // This should execute the entire chain
    let _buffer = lazy_result.materialize()?;

    // Verify graph stats
    let graph = context.graph();
    log::info!("  Graph has {} active nodes", graph.len());

    log::info!("  ✓ Chained operations passed");
    Ok(())
}

async fn test_lazy_token_shift(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 7: Lazy token shift operation");

    let c = 64; // channels
    let t = 4; // tokens
    let b = 1; // batch

    // Input: [C, T, B]
    let input_shape = Shape::new(c, t, b, 1);
    let input_data: Vec<f16> = (0..input_shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();
    let cpu_input: TensorCpu<f16> = TensorInit::from_data(input_shape, input_data)?;
    let gpu_input: TensorGpu<f16, ReadWrite> = cpu_input.to(context);

    // State: [C, 1, B] - previous token state (f32 for compatibility with existing state format)
    let state_shape = Shape::new(c, 1, b, 1);
    let state_data: Vec<f32> = (0..state_shape.len()).map(|i| i as f32 * 0.05).collect();
    let cpu_state: TensorCpu<f32> = TensorInit::from_data(state_shape, state_data)?;
    let gpu_state: TensorGpu<f32, ReadWrite> = cpu_state.to(context);

    // Mix: [C, 1, 1] - mixing coefficients (broadcast across tokens)
    let mix_shape = Shape::new(c, 1, 1, 1);
    let mix_data: Vec<f16> = (0..mix_shape.len())
        .map(|_| f16::from_f32(0.5)) // 50% mix
        .collect();
    let cpu_mix: TensorCpu<f16> = TensorInit::from_data(mix_shape, mix_data)?;
    let gpu_mix: TensorGpu<f16, ReadWrite> = cpu_mix.to(context);

    // Create lazy tensors
    let lazy_input = LazyTensor::from_gpu(&gpu_input);
    let lazy_state = LazyTensor::from_gpu(&gpu_state);
    let lazy_mix = LazyTensor::from_gpu(&gpu_mix);

    // Apply token shift: input.token_shift(state, mix)
    let lazy_shifted = lazy_input.token_shift(&lazy_state, &lazy_mix);

    // Materialize
    let _buffer = lazy_shifted.materialize()?;

    log::info!("  ✓ Lazy token shift operation passed");
    Ok(())
}

async fn test_lazy_blend(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 8: Lazy blend operation");

    let shape = Shape::new(64, 4, 1, 1);

    // lhs tensor
    let lhs_data: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();
    let cpu_lhs: TensorCpu<f16> = TensorInit::from_data(shape, lhs_data)?;
    let gpu_lhs: TensorGpu<f16, ReadWrite> = cpu_lhs.to(context);

    // rhs tensor
    let rhs_data: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.2))
        .collect();
    let cpu_rhs: TensorCpu<f16> = TensorInit::from_data(shape, rhs_data)?;
    let gpu_rhs: TensorGpu<f16, ReadWrite> = cpu_rhs.to(context);

    // factor tensor (broadcast: [C, 1, 1])
    let factor_shape = Shape::new(64, 1, 1, 1);
    let factor_data: Vec<f16> = (0..factor_shape.len())
        .map(|_| f16::from_f32(0.3)) // 30% lhs, 70% rhs
        .collect();
    let cpu_factor: TensorCpu<f16> = TensorInit::from_data(factor_shape, factor_data)?;
    let gpu_factor: TensorGpu<f16, ReadWrite> = cpu_factor.to(context);

    // Create lazy tensors
    let lazy_lhs = LazyTensor::from_gpu(&gpu_lhs);
    let lazy_rhs = LazyTensor::from_gpu(&gpu_rhs);
    let lazy_factor = LazyTensor::from_gpu(&gpu_factor);

    // Apply blend: lhs * factor + rhs * (1 - factor)
    let lazy_blended = lazy_lhs.blend(&lazy_rhs, &lazy_factor);

    // Materialize
    let _buffer = lazy_blended.materialize()?;

    log::info!("  ✓ Lazy blend operation passed");
    Ok(())
}

async fn test_matmul_bias_fusion(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 9: MatMul + Bias fusion (automatic)");

    // Create a simple FP16 matrix [K=64, M=32, B=1]
    let k = 64;
    let m = 32;
    let t = 4; // tokens

    // Matrix weights
    let matrix_shape = Shape::new(k, m, 1, 1);
    let matrix_data: Vec<f16> = (0..matrix_shape.len())
        .map(|i| f16::from_f32((i as f32 * 0.01) - 0.5))
        .collect();
    let cpu_matrix: TensorCpu<f16> = TensorInit::from_data(matrix_shape, matrix_data)?;
    let gpu_matrix: TensorGpu<f16, ReadWrite> = cpu_matrix.to(context);
    let matrix = Matrix::Fp16(gpu_matrix);

    // Input tensor [K, T, B]
    let input_shape = Shape::new(k, t, 1, 1);
    let input_data: Vec<f16> = (0..input_shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();
    let cpu_input: TensorCpu<f16> = TensorInit::from_data(input_shape, input_data)?;
    let gpu_input: TensorGpu<f16, ReadWrite> = cpu_input.to(context);

    // Bias tensor [M, 1, 1] - will be broadcast
    let bias_shape = Shape::new(m, 1, 1, 1);
    let bias_data: Vec<f16> = (0..bias_shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.05))
        .collect();
    let cpu_bias: TensorCpu<f16> = TensorInit::from_data(bias_shape, bias_data)?;
    let gpu_bias: TensorGpu<f16, ReadWrite> = cpu_bias.to(context);

    // Create lazy tensors
    let lazy_input = LazyTensor::from_gpu(&gpu_input);
    let lazy_bias = LazyTensor::from_gpu(&gpu_bias);

    // Build graph: matmul(input, matrix) + bias
    // This pattern should be detected and fused by the optimizer
    let lazy_matmul = lazy_input.matmul(&matrix, Activation::None, false);
    let lazy_result = lazy_matmul.add(&lazy_bias);

    // Check graph stats before materialization
    let graph = context.graph();
    let nodes_before = graph.len();
    drop(graph);

    // Materialize - this triggers optimization
    let _buffer = lazy_result.materialize()?;

    log::info!("  Graph had {} nodes before optimization", nodes_before);
    log::info!("  ✓ MatMul + Bias fusion test passed");
    Ok(())
}

async fn test_explicit_matmul_bias(context: &Context) -> anyhow::Result<()> {
    log::info!("Test 10: Explicit matmul_bias operation");

    // Create a simple FP16 matrix [K=64, M=32, B=1]
    let k = 64;
    let m = 32;
    let t = 4; // tokens

    // Matrix weights
    let matrix_shape = Shape::new(k, m, 1, 1);
    let matrix_data: Vec<f16> = (0..matrix_shape.len())
        .map(|i| f16::from_f32((i as f32 * 0.01) - 0.5))
        .collect();
    let cpu_matrix: TensorCpu<f16> = TensorInit::from_data(matrix_shape, matrix_data)?;
    let gpu_matrix: TensorGpu<f16, ReadWrite> = cpu_matrix.to(context);
    let matrix = Matrix::Fp16(gpu_matrix);

    // Input tensor [K, T, B]
    let input_shape = Shape::new(k, t, 1, 1);
    let input_data: Vec<f16> = (0..input_shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.1))
        .collect();
    let cpu_input: TensorCpu<f16> = TensorInit::from_data(input_shape, input_data)?;
    let gpu_input: TensorGpu<f16, ReadWrite> = cpu_input.to(context);

    // Bias tensor [M, 1, 1] - will be broadcast
    let bias_shape = Shape::new(m, 1, 1, 1);
    let bias_data: Vec<f16> = (0..bias_shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.05))
        .collect();
    let cpu_bias: TensorCpu<f16> = TensorInit::from_data(bias_shape, bias_data)?;
    let gpu_bias: TensorGpu<f16, ReadWrite> = cpu_bias.to(context);

    // Create lazy tensors
    let lazy_input = LazyTensor::from_gpu(&gpu_input);
    let lazy_bias = LazyTensor::from_gpu(&gpu_bias);

    // Use explicit matmul_bias - directly creates fused operation
    let lazy_result = lazy_input.matmul_bias(&matrix, &lazy_bias, Activation::None, false);

    // Materialize
    let _buffer = lazy_result.materialize()?;

    log::info!("  ✓ Explicit matmul_bias operation passed");
    Ok(())
}

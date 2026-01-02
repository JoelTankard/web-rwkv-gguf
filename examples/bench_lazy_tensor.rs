//! Benchmark comparing lazy vs eager tensor evaluation.
//!
//! This benchmark measures the overhead of lazy evaluation vs direct eager execution.

use std::time::Instant;

use half::f16;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    tensor::{
        kind::ReadWrite, lazy_tensor::LazyTensor, ops::TensorOp, shape::Shape, TensorCpu,
        TensorGpu, TensorInit, TensorInto,
    },
};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 100;

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

    log::info!("Benchmarking lazy vs eager tensor evaluation...\n");

    // Benchmark different operation chains
    bench_single_add(&context).await?;
    bench_chained_ops(&context).await?;

    log::info!("Benchmark complete!");
    Ok(())
}

async fn bench_single_add(context: &Context) -> anyhow::Result<()> {
    log::info!("=== Single Add Operation ===");

    let shape = Shape::new(2048, 16, 1, 1); // Realistic size
    let data_a: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.001))
        .collect();
    let data_b: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.002))
        .collect();

    let cpu_a: TensorCpu<f16> = TensorInit::from_data(shape, data_a)?;
    let cpu_b: TensorCpu<f16> = TensorInit::from_data(shape, data_b)?;
    let gpu_a: TensorGpu<f16, ReadWrite> = cpu_a.to(context);
    let gpu_b: TensorGpu<f16, ReadWrite> = cpu_b.to(context);

    // Warmup eager
    for _ in 0..WARMUP_ITERS {
        let op = TensorOp::add(&gpu_a, &gpu_b)?;
        context.queue.submit(context.encode(&op));
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Benchmark eager
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let op = TensorOp::add(&gpu_a, &gpu_b)?;
        context.queue.submit(context.encode(&op));
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let eager_time = start.elapsed();

    // Warmup lazy
    for _ in 0..WARMUP_ITERS {
        let lazy_a = LazyTensor::from_gpu(&gpu_a);
        let lazy_b = LazyTensor::from_gpu(&gpu_b);
        let lazy_sum = lazy_a.add(&lazy_b);
        let _ = lazy_sum.materialize()?;
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Benchmark lazy
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let lazy_a = LazyTensor::from_gpu(&gpu_a);
        let lazy_b = LazyTensor::from_gpu(&gpu_b);
        let lazy_sum = lazy_a.add(&lazy_b);
        let _ = lazy_sum.materialize()?;
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let lazy_time = start.elapsed();

    log::info!(
        "  Eager: {:>8.2}ms ({:.2}µs/iter)",
        eager_time.as_secs_f64() * 1000.0,
        eager_time.as_micros() as f64 / BENCH_ITERS as f64
    );
    log::info!(
        "  Lazy:  {:>8.2}ms ({:.2}µs/iter)",
        lazy_time.as_secs_f64() * 1000.0,
        lazy_time.as_micros() as f64 / BENCH_ITERS as f64
    );
    log::info!(
        "  Overhead: {:.1}%\n",
        (lazy_time.as_secs_f64() / eager_time.as_secs_f64() - 1.0) * 100.0
    );

    Ok(())
}

async fn bench_chained_ops(context: &Context) -> anyhow::Result<()> {
    log::info!("=== Chained Operations (add → affine → softmax) ===");

    let shape = Shape::new(2048, 16, 1, 1);
    let data_a: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.001))
        .collect();
    let data_b: Vec<f16> = (0..shape.len())
        .map(|i| f16::from_f32(i as f32 * 0.002))
        .collect();

    let cpu_a: TensorCpu<f16> = TensorInit::from_data(shape, data_a)?;
    let cpu_b: TensorCpu<f16> = TensorInit::from_data(shape, data_b)?;
    let gpu_a: TensorGpu<f16, ReadWrite> = cpu_a.to(context);
    let gpu_b: TensorGpu<f16, ReadWrite> = cpu_b.to(context);

    // Warmup eager
    for _ in 0..WARMUP_ITERS {
        let op1 = TensorOp::add(&gpu_a, &gpu_b)?;
        context.queue.submit(context.encode(&op1));
        let op2 = TensorOp::affine(&gpu_b, 0.5, 0.0)?;
        context.queue.submit(context.encode(&op2));
        let op3 = TensorOp::softmax(&gpu_b)?;
        context.queue.submit(context.encode(&op3));
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Benchmark eager
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let op1 = TensorOp::add(&gpu_a, &gpu_b)?;
        context.queue.submit(context.encode(&op1));
        let op2 = TensorOp::affine(&gpu_b, 0.5, 0.0)?;
        context.queue.submit(context.encode(&op2));
        let op3 = TensorOp::softmax(&gpu_b)?;
        context.queue.submit(context.encode(&op3));
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let eager_time = start.elapsed();

    // Warmup lazy
    for _ in 0..WARMUP_ITERS {
        let lazy_a = LazyTensor::from_gpu(&gpu_a);
        let lazy_b = LazyTensor::from_gpu(&gpu_b);
        let lazy_sum = lazy_a.add(&lazy_b);
        let lazy_scaled = lazy_sum.affine(0.5, 0.0);
        let lazy_result = lazy_scaled.softmax();
        let _ = lazy_result.materialize()?;
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Benchmark lazy
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let lazy_a = LazyTensor::from_gpu(&gpu_a);
        let lazy_b = LazyTensor::from_gpu(&gpu_b);
        let lazy_sum = lazy_a.add(&lazy_b);
        let lazy_scaled = lazy_sum.affine(0.5, 0.0);
        let lazy_result = lazy_scaled.softmax();
        let _ = lazy_result.materialize()?;
    }
    let _ = context.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    let lazy_time = start.elapsed();

    log::info!(
        "  Eager: {:>8.2}ms ({:.2}µs/iter)",
        eager_time.as_secs_f64() * 1000.0,
        eager_time.as_micros() as f64 / BENCH_ITERS as f64
    );
    log::info!(
        "  Lazy:  {:>8.2}ms ({:.2}µs/iter)",
        lazy_time.as_secs_f64() * 1000.0,
        lazy_time.as_micros() as f64 / BENCH_ITERS as f64
    );
    log::info!(
        "  Overhead: {:.1}%\n",
        (lazy_time.as_secs_f64() / eager_time.as_secs_f64() - 1.0) * 100.0
    );

    Ok(())
}

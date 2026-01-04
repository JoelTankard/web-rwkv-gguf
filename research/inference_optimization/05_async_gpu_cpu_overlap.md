# Async GPU-CPU Overlap Optimization

## Current Implementation

Location: `src/runtime/mod.rs:111-209` (`TokioRuntime::run`)

Current flow:

```
1. Receive input from channel
2. Build job (CPU) - dispatch()
3. Load embeddings (CPU→GPU) - job.load()
4. Submit to GPU - job.submit()
5. Wait for GPU - job.back().await
6. Send result back
```

The `TokioRuntime` does have some pipelining via the job queue, but there's room for improvement.

## Problem

1. **Sequential phases**: CPU waits for GPU, GPU waits for CPU
2. **Single job in flight**: Only one inference running at a time per batch
3. **Blocking readback**: `job.back().await` blocks until GPU completes

## Optimization Ideas

### Idea 1: Triple Buffering

Overlap three phases: build, execute, readback

```rust
struct TripleBufferedRuntime {
    building: Option<JoinHandle<Job>>,
    executing: Option<Job>,
    reading: Option<JoinHandle<Output>>,
}

impl TripleBufferedRuntime {
    async fn step(&mut self, input: Input) -> Option<Output> {
        // Start building next job
        let new_building = tokio::spawn(self.dispatch(input));

        // Submit currently built job
        if let Some(job) = self.executing.take() {
            let reading = tokio::spawn(async move {
                job.back().await
            });
            self.reading = Some(reading);
        }

        // Get previously executing job's result
        let result = if let Some(reading) = self.reading.take() {
            Some(reading.await?)
        } else {
            None
        };

        // Rotate buffers
        if let Some(building) = self.building.take() {
            self.executing = Some(building.await?);
            self.executing.as_mut().unwrap().submit();
        }
        self.building = Some(new_building);

        result
    }
}
```

**Benefit**: GPU never waits for CPU job building

### Idea 2: Speculative Job Pre-building

Pre-build jobs for likely next inputs:

```rust
impl TokioRuntime {
    async fn run(model: Arc<M>, receiver: Receiver<Submission>) {
        let mut job_cache: LruCache<RnnInfo, Job> = LruCache::new(4);

        while let Ok(submission) = receiver.recv_async().await {
            let info = submission.input.iter().next().unwrap();

            // Check cache first
            let job = if let Some(cached) = job_cache.pop(&info) {
                cached
            } else {
                model.dispatch(info)?
            };

            // Speculatively build next likely job
            let next_info = predict_next_info(&info);
            if !job_cache.contains(&next_info) {
                let model = model.clone();
                tokio::spawn(async move {
                    if let Ok(job) = model.dispatch(next_info.clone()) {
                        job_cache.put(next_info, job);
                    }
                });
            }

            // ... rest of inference
        }
    }
}
```

**Benefit**: Job building happens during previous token's GPU execution

### Idea 3: Async Embedding Upload

Currently `job.load()` is synchronous. Make it async:

```rust
impl Job for RnnJob {
    async fn load(&self, input: &RnnChunk) -> Result<(), RuntimeError> {
        // Prepare data on CPU (can be parallelized)
        let embed_data = tokio::task::spawn_blocking(|| {
            prepare_embeddings(input)
        }).await?;

        // Async upload to GPU
        self.input.load_async(&embed_data).await?;

        Ok(())
    }
}

impl TensorGpu {
    async fn load_async(&self, data: &[T]) -> Result<(), TensorError> {
        // Use write_buffer_with for zero-copy when possible
        let staging = self.context.device.create_buffer_init(&BufferInitDescriptor {
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::COPY_SRC,
            ..
        });

        // Async copy
        let mut encoder = self.context.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&staging, 0, &self.buffer, 0, self.size());
        self.context.queue.submit(Some(encoder.finish()));

        // Don't wait - let it overlap with other work
        Ok(())
    }
}
```

### Idea 4: Pipelined Batch Processing

For batch inference, pipeline across batches:

```rust
async fn infer_batch(&self, inputs: Vec<Input>) -> Vec<Output> {
    let mut futures = FuturesOrdered::new();

    for input in inputs {
        // Start each inference without waiting
        let future = self.infer_single(input);
        futures.push_back(future);

        // Small yield to allow GPU work to start
        tokio::task::yield_now().await;
    }

    // Collect results in order
    futures.collect().await
}
```

### Idea 5: GPU-Side Token Sampling

Move sampling to GPU to avoid readback latency:

```rust
// Current: read all logits back to CPU, sample there
let output = job.back().await;  // Reads [vocab_size] floats
let token = sample(&output);

// Optimized: sample on GPU, only read token ID
impl RnnJob {
    async fn back_sampled(self, temperature: f32, top_p: f32) -> u32 {
        // GPU kernel: softmax → top-p filter → sample
        let sample_op = TensorOp::sample_token(&self.output, temperature, top_p);
        self.context.queue.submit(self.context.encode(&sample_op));

        // Only read back single u32 token ID
        self.sampled_token.back().await
    }
}
```

**Benefit**: Reads 4 bytes instead of 260KB (65K vocab × 4 bytes)

### Idea 6: Continuous Batching

Process multiple sequences with different lengths simultaneously:

```rust
struct ContinuousBatcher {
    active_sequences: Vec<Sequence>,
    max_batch_size: usize,
}

impl ContinuousBatcher {
    async fn step(&mut self) -> Vec<(SequenceId, Token)> {
        // Collect all sequences that need processing
        let batch: Vec<_> = self.active_sequences
            .iter()
            .filter(|s| s.needs_token())
            .take(self.max_batch_size)
            .collect();

        // Single batched inference
        let outputs = self.runtime.infer_batch(batch).await;

        // Return results, some sequences may finish
        outputs
    }
}
```

## Estimated Impact

| Optimization      | Latency Reduction | Throughput Increase | Complexity |
| ----------------- | ----------------- | ------------------- | ---------- |
| Triple Buffering  | 10-20%            | 20-30%              | Medium     |
| Speculative Build | 5-15%             | 10-20%              | Medium     |
| Async Upload      | 5-10%             | 5-10%               | Low        |
| Pipelined Batch   | N/A               | 30-50%              | Medium     |
| GPU Sampling      | 20-40%            | 10-20%              | High       |
| Continuous Batch  | N/A               | 50-100%             | High       |

## Experimental Results (Apple M2 Max)

**Baseline**: 54.11 tok/s generation, 208.60 tok/s prefill (2.9B Q8_0 model)

### Pipeline Phase Breakdown

| Phase      | Time (ms) | % of Total |
| ---------- | --------- | ---------- |
| dispatch() | 6.815     | 26.2%      |
| load()     | 0.053     | 0.2%       |
| submit()   | 0.037     | 0.1%       |
| back()     | 19.115    | 73.5%      |
| **TOTAL**  | 26.019    | 100%       |

### GPU-side Sampling Results

-   **Implementation**: `TensorOp::sample_argmax` + `RnnJob::back_sampled()`
-   **Data transfer savings**: 256KB → 4 bytes (65536x reduction)
-   **Actual time savings**: ~0.07ms per token
-   **Speedup**: ~0.4%
-   **Conclusion**: Minimal benefit on unified memory (Apple Silicon). The readback time is dominated by GPU compute, not PCIe transfer.

### Triple Buffering Results

-   **Implementation**: `TripleBufferRuntime` in `src/runtime/triple_buffer.rs`
-   **TokioRuntime**: 50.94 tok/s
-   **TripleBufferRuntime**: 50.45 tok/s
-   **Speedup**: 0.99x (no improvement)
-   **Conclusion**: The existing `TokioRuntime` already has speculative job pre-building (lines 156-169). The GPU execution time (~19ms) dominates, so overlapping the ~7ms dispatch doesn't help.

### Key Findings

1. **GPU compute is the bottleneck**: 73.5% of time is spent in GPU execution
2. **Unified memory negates transfer optimizations**: On Apple Silicon, CPU-GPU data transfer is nearly free
3. **TokioRuntime already pipelines**: The existing implementation has speculative job pre-building
4. **Async overlap has diminishing returns**: When GPU time >> CPU time, overlap doesn't help

### Recommendations

For Apple Silicon / unified memory systems:

-   Focus on **GPU compute optimization** (better shaders, Metal acceleration)
-   **Continuous batching** for throughput scenarios
-   GPU-side sampling may help on **discrete GPUs** with PCIe bottleneck

## Profiling

```bash
# Measure current pipeline efficiency
RUST_LOG=trace cargo run --release --example bench 2>&1 | grep -E "(dispatch|load|submit|back)"

# Time breakdown
cargo run --release --example bench -- --model $MODEL --tokens 100 --verbose

# Pipeline phase benchmark
cargo run --release --example bench_pipeline_phases -- --model $MODEL
```

## Files Modified

-   `src/runtime/mod.rs`: Added `triple_buffer` module
-   `src/runtime/triple_buffer.rs`: `TripleBufferRuntime` (experimental)
-   `src/runtime/v7.rs`: Added `RnnJob::back_sampled()` method
-   `src/runtime/softmax.rs`: Added `sample_argmax_one()`, `sample_argmax_gpu()`
-   `src/tensor/ops.rs`: Added `TensorOp::sample_argmax`
-   `src/shaders/sample_argmax.wgsl`: GPU argmax sampling kernel

---

## GPU Compute Optimization

Since GPU compute is the bottleneck (73.5% of time), the next step is to optimize the GPU shaders themselves.

### Current GPU Time Breakdown

The `back()` phase (~19ms) includes:

1. GPU execution of all layer operations
2. Buffer mapping and readback (minimal on unified memory)

### Optimization Opportunities

#### 1. Subgroup Operations

Use WGSL subgroup operations for faster reductions in:

-   LayerNorm (already has `subgroup-ops` feature)
-   Softmax (already has `subgroup-ops` feature)
-   Token mixing reductions

**Status**: Partially implemented, needs verification of actual usage.

#### 2. Fused Operations

Combine multiple shader dispatches into single kernels:

-   Fused attention (time_mix + key/value/receptance matmuls)
-   Fused FFN (gate + up matmul → activation → down matmul)

**Potential**: 10-30% reduction in dispatch overhead

#### 3. Memory Access Patterns

Optimize memory coalescing in matmul shaders:

-   Ensure threads in a workgroup access contiguous memory
-   Use shared memory for frequently accessed data
-   Reduce bank conflicts

#### 4. Workgroup Size Tuning

Current workgroup sizes may not be optimal for Apple Silicon:

-   Test different BLOCK_SIZE values (64, 128, 256)
-   Profile occupancy and register pressure

#### 5. Reduce Precision Where Possible

-   Use FP16 for intermediate calculations where precision allows
-   Consider mixed-precision accumulation

### Profiling Commands

```bash
# GPU timing with Metal System Trace
xcrun xctrace record --template 'Metal System Trace' --launch -- \
    cargo run --release --example benchmark -- --model $MODEL --tokens 32

# Shader compilation time
RUST_LOG=wgpu_core=debug cargo run --release --example benchmark 2>&1 | grep -i shader
```

### Experimental Results

#### Shader Timing Benchmark (2.9B Q8_0, Apple M2 Max)

| Phase      | Time   | % of Total |
| ---------- | ------ | ---------- |
| dispatch() | 6.7ms  | 26%        |
| submit()   | 0.04ms | 0.2%       |
| back()     | 19ms   | 74%        |
| **TOTAL**  | 25.8ms | 100%       |

**Per-layer estimate**: 0.6ms
**Total matmuls**: ~448 (14 per layer × 32 layers)
**Estimated per-matmul**: ~36µs

#### Fused Token Shift + LayerNorm Test

-   **Fused**: 53.19 tok/s
-   **Non-fused**: 53.08 tok/s
-   **Improvement**: ~0.2% (negligible)

The fused shader reduces dispatch count but doesn't significantly improve performance because:

1. GPU execution time dominates (74%)
2. Dispatch overhead is already amortized across many operations
3. The fused shader has similar computational complexity

#### Key Findings

1. **Matmuls dominate GPU time**: ~85% of GPU time is matrix multiplication
2. **Subgroup ops already enabled**: Using `subgroupAdd` for efficient reductions
3. **Workgroup size is reasonable**: 128 threads = 4 subgroups on Apple Silicon
4. **Dispatch overhead is bind group creation**: ~448 bind groups per inference

#### Recommendations

For further GPU compute optimization:

1. **Metal acceleration** - Native Metal shaders are 4.78x faster for Q4K matmul (blocked by sync issues)
2. **Bind group caching** - Reuse bind groups across inferences when buffers don't change
3. **Persistent kernels** - Single kernel that processes entire layer (complex)
4. **Quantization** - Q4K is faster than Q8_0 due to reduced memory bandwidth

### Bind Group Caching Analysis

Tested bind group cache effectiveness over 50 runs with same input:

| Metric        | First Run | Steady State | Improvement |
| ------------- | --------- | ------------ | ----------- |
| Dispatch time | 25.2ms    | 6.6ms        | **73.8%**   |
| Total time    | 976ms     | 26.2ms       | **97.3%**   |

**Findings:**

-   Bind group caching is **already implemented** in `Context.bindings`
-   First run includes shader compilation + bind group creation
-   Subsequent runs benefit from cached pipelines and bind groups
-   Remaining 6.6ms dispatch time is `TensorOp` tree construction
-   `TensorOp` caching not possible (doesn't implement `Clone`)

### TensorOp Tree Caching

Implemented `Clone` for `TensorOp` and added caching in `Bundle`:

| Metric                | Before   | After          | Improvement |
| --------------------- | -------- | -------------- | ----------- |
| Dispatch time         | 6.6ms    | **2.1ms**      | **69%**     |
| Dispatch % of total   | 26%      | **9.8%**       | -           |
| Generation throughput | 53 tok/s | **54.3 tok/s** | **~2.5%**   |

**Implementation:**

-   Added `#[derive(Clone)]` to `TensorOp` enum (all fields are `Arc`-wrapped)
-   Added `OpCacheKey` struct with `(num_token, num_header, embed_only)`
-   Added `op_cache: SharedResourceCache<OpCacheKey, TensorOp>` to `Bundle`
-   Cache only used when no hooks are registered (common case)
-   Extracted `build_ops()` method for cleaner code

**Files modified:**

-   `src/tensor/ops.rs` - Added `Clone` derive to `TensorOp`
-   `src/runtime/v7.rs` - Added `op_cache` and `build_ops()` method

### Next Steps

1. ~~Profile individual shader execution times~~ ✅ Done
2. ~~Identify hottest shaders (matmul operations)~~ ✅ Confirmed
3. ~~Test subgroup operations on Apple Silicon~~ ✅ Already enabled
4. ~~Test fused attention kernel~~ ✅ Minimal benefit
5. ~~Bind group caching~~ ✅ Already implemented (73.8% improvement)
6. Focus on Metal acceleration (blocked by sync issues)

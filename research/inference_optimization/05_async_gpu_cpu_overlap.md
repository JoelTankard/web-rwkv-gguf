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

## Recommended Experiment

1. **First**: Implement GPU-side sampling - biggest single-token latency win
2. **Second**: Triple buffering for better GPU utilization
3. **Third**: Continuous batching for throughput scenarios

## Profiling

```bash
# Measure current pipeline efficiency
RUST_LOG=trace cargo run --release --example bench 2>&1 | grep -E "(dispatch|load|submit|back)"

# Time breakdown
cargo run --release --example bench -- --model $MODEL --tokens 100 --verbose
```

## Files to Modify

-   `src/runtime/mod.rs`: `TokioRuntime` pipelining
-   `src/runtime/v7.rs`: `RnnJob` async methods
-   `src/tensor/ops.rs`: Add `TensorOp::sample_token`
-   `src/shaders/sample.wgsl`: GPU sampling kernel (new)

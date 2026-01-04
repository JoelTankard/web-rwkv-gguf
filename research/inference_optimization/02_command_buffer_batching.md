# Command Buffer Batching Optimization

## Status: ❌ TESTED - NO BENEFIT

**Result:** Removing Sep markers or modifying command buffer batching provided no measurable performance improvement.

**Tested:** Removing `TensorOp::Sep` entirely - no change in generation speed.

**Conclusion:** WebGPU's implicit barriers between dispatches are sufficient. The Sep markers don't add significant overhead, and removing them doesn't help. The bottleneck is elsewhere (likely matmul compute time, not dispatch overhead).

---

## Current Implementation

Location: `src/tensor/ops.rs:82-175` (`Context::encode`)

The current encoding creates separate command buffers at each `TensorOp::Sep` marker:

```rust
fn encode_inner(&self, op: &TensorOp, profile: bool) -> Vec<CommandBuffer> {
    // ...
    commands
        .into_iter()
        .enumerate()
        .filter(|(_, atoms)| !atoms.is_empty())
        .map(|(batch_idx, atoms)| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            // ... dispatch all atoms in this batch
            encoder.finish()
        })
        .collect()
}
```

And in `v7.rs:697-699`:

```rust
if (index + 1) % model.sep == 0 {
    ops.push(TensorOp::Sep);
}
```

## Problem

1. **Multiple queue submissions**: Each `Sep` creates a new command buffer, requiring separate GPU submissions
2. **Synchronization overhead**: Each `queue.submit()` has driver overhead
3. **Fixed separation**: `model.sep` (default 1024) may not be optimal for all GPUs

## Optimization Ideas

### Idea 1: Adaptive Command Buffer Sizing

Dynamically determine optimal batch size based on GPU capabilities:

```rust
impl Context {
    fn optimal_sep(&self) -> usize {
        // Query GPU limits
        let max_compute_invocations = self.device.limits().max_compute_invocations_per_workgroup;
        let max_buffer_size = self.device.limits().max_buffer_size;

        // Heuristic: larger GPUs can handle more ops per submission
        match max_buffer_size {
            x if x > 2_000_000_000 => 2048,  // High-end GPU
            x if x > 500_000_000 => 1024,    // Mid-range
            _ => 512,                         // Integrated/low-end
        }
    }
}
```

### Idea 2: Single Command Buffer with Barriers

Use compute barriers instead of separate command buffers:

```rust
pub enum TensorOp {
    Atom { ... },
    List(Vec<TensorOp>),
    Barrier,  // New: memory barrier instead of Sep
}

fn encode_inner(&self, op: &TensorOp) -> Vec<CommandBuffer> {
    let mut encoder = self.device.create_command_encoder(&Default::default());
    let mut pass = encoder.begin_compute_pass(&Default::default());

    for atom in atoms {
        match atom {
            TensorOp::Barrier => {
                // Insert memory barrier within same pass
                pass.insert_debug_marker("barrier");
                // WebGPU handles implicit barriers between dispatches
            }
            _ => dispatch(&mut pass, atom),
        }
    }

    drop(pass);
    vec![encoder.finish()]  // Single command buffer
}
```

**Note:** WebGPU already provides implicit barriers between dispatches in the same pass. The `Sep` may be unnecessary for correctness.

### Idea 3: Pipelined Double-Buffering

Overlap command buffer creation with execution:

```rust
struct PipelinedDispatcher {
    current_buffer: Option<CommandBuffer>,
    next_buffer: Option<CommandBuffer>,
}

impl PipelinedDispatcher {
    async fn dispatch(&mut self, ops: TensorOp) {
        // Submit current while building next
        if let Some(current) = self.current_buffer.take() {
            self.context.queue.submit(Some(current));
        }

        // Build next buffer (can overlap with GPU execution)
        self.current_buffer = self.next_buffer.take();
        self.next_buffer = Some(self.context.encode(&ops));
    }
}
```

### Idea 4: Remove Sep Entirely (Experiment)

Test if `Sep` is actually needed:

```rust
// In v7.rs, comment out Sep insertion
// if (index + 1) % model.sep == 0 {
//     ops.push(TensorOp::Sep);
// }
```

WebGPU spec guarantees that dispatches within a compute pass execute in order with proper memory visibility. The `Sep` may be legacy from when explicit synchronization was needed.

## Estimated Impact

| Approach        | Latency Reduction | Risk   | Complexity |
| --------------- | ----------------- | ------ | ---------- |
| Remove Sep      | 5-15%             | Medium | Trivial    |
| Adaptive Sizing | 2-5%              | Low    | Low        |
| Single Buffer   | 5-10%             | Low    | Medium     |
| Double-Buffer   | 10-20%            | Medium | High       |

## Recommended Experiment

1. **First**: Remove `Sep` entirely and benchmark. If no correctness issues, this is free performance.

2. **Second**: If Sep is needed, try adaptive sizing based on layer count and GPU.

## Benchmark Command

```bash
# Baseline
cargo run --release --example bench -- --model $MODEL --prompt "Hello" --tokens 100

# With Sep removed (modify code first)
cargo run --release --example bench -- --model $MODEL --prompt "Hello" --tokens 100
```

## Files to Modify

-   `src/runtime/v7.rs`: Sep insertion logic (line ~697)
-   `src/tensor/ops.rs`: `encode_inner` method
-   `src/runtime/mod.rs`: `Model::DEFAULT_SEP` constant

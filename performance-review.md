# Performance Review

## Summary

-   Inference latency and VRAM usage are currently limited more by GPU buffer churn and command re-encoding than by shader math. The runtime rebuilds large tensor arenas and command lists whenever the token chunk size changes.
-   Resource caches default to very small lifetimes, so most tensor/view buffers are recreated instead of reused, leading to extra VRAM pressure and host-GPU synchronization stalls.
-   Input chunk scheduling rounds down to the nearest 32 tokens, which fragments work into extra launches and keeps intermediate tensors around longer than necessary.

## Hotspots & Recommendations

1. **Resource cache churn in `Context`**

    - Every `ContextBuilder::build` instantiates buffer caches with a limit of four generations and invokes `context.maintain()` before each dispatch (@src/context.rs#135-145, @src/runtime/v7.rs#622-624). This means buffers/bind groups are frequently evicted even while identical workloads repeat, forcing reallocations and GPU stalls.
    - _Recommendation_: raise the cache `limit` for `buffers`, `shapes`, and `bindings` based on resident model size (e.g., proportional to number of active layers) and move `maintain()` to a low-frequency housekeeping task (timer or frame counter) instead of per-dispatch. Also expose a knob so long-lived runtimes can skip `maintain()` entirely when memory headroom exists.

2. **Runtime arena reallocation per chunk size**

    - `Bundle::checkout_buffer` caches tensor arenas by exact `num_token`, but `RnnIter` emits variable chunk sizes due to remainder handling, so almost every flush allocates a new `Runtime<F>` (tens of tensors covering att/ffn scratch, see @src/runtime/v7.rs#323-363 and @src/runtime/v7.rs#545-563).
    - _Recommendation_: bucket chunk sizes (e.g., 32, 64, 96 …) or always round _up_ to the next bucket before calling `checkout_buffer` so arenas are reused. Pair this with `token_chunk_size` user controls that prefer a small set of values.

3. **Token chunk rounding causes extra passes**

    - `RnnIter::next` trims each step down to the nearest multiple of 32 tokens (`MIN_TOKEN_CHUNK_SIZE`), so leftover tokens are processed in an additional pass with all the same setup cost (@src/runtime/infer/rnn.rs#283-333). That increases command submissions and retains buffer arenas longer, hurting latency.
    - _Recommendation_: allow the final chunk in a burst to keep the remainder (only pad when the chunk exceeds `MIN_TOKEN_CHUNK_SIZE`) and gate the “turbo” kernels on `(num_token % 32 == 0)` instead of shrinking the chunk. This reduces launch count and shortens buffer lifetimes, improving throughput and VRAM utilization.

4. **State read-back is serialized per layer**

    - When backing the recurrent state, `State::back` copies each layer into a temporary tensor and awaits them sequentially (@src/runtime/v7.rs#152-170). On large models this saturates command encoders and keeps CPU copies live longer than needed.
    - _Recommendation_: batch copy commands for all layers into a single encoder and await `join_all` on the resulting futures, or keep a persistent staging buffer per batch. This will lower wall-clock latency for checkpointing and reduce peak host memory.

5. **Command encoding underutilizes identical graphs**

    - Each dispatch rebuilds a full `Vec<TensorOp>` and re-encodes WGSL pipelines even when the layer topology is identical (see the layer loop and final `context.encode` in @src/runtime/v7.rs#662-703). Although shader modules are cached, the command buffer creation overhead remains.
    - _Recommendation_: cache the flattened `TensorOp` DAG or pre-encoded `CommandBuffer`s keyed by `(num_token_bucket, quant profile, rescale, sep)` and only rebind data buffers per step. For dynamic hooks, treat them as secondary overlays so the base graph stays reusable. This will shrink CPU overhead and allow deeper queueing.

6. **ResourceCache limits force CPU-GPU sync even under load**
    - `ResourceCache::maintain` immediately evicts entries whose `life >= limit` when `Arc` counts drop to one (@src/tensor/cache.rs#40-116). Because the `Context` cache limit is set to 4, temporary tensors are torn down quickly, forcing reuploads the next time the same view is requested.
    - _Recommendation_: increase the default limit (e.g., 64+) and tie it to observed free VRAM. When memory is tight, use an LRU by byte size instead of generation count so hot tensors survive longer, directly cutting upload bandwidth.

Implementing the above will materially reduce both inference latency and peak memory usage without touching model math.

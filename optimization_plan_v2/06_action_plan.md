# Action Plan: GPU Backend Optimization

## Decision Matrix

Based on the research conducted, here is the recommended action plan.

## Immediate Actions (Do Now)

### 1. Do NOT migrate to gfx-hal

-   **Reason:** Deprecated, no longer maintained
-   **Alternative:** Stay with wgpu

### 2. Profile current performance

-   Enable tracing to identify actual bottlenecks
-   Measure CPU vs GPU time split
-   Identify hot paths in shader execution

```bash
cargo run --example bench --features trace --release
```

## Short-Term Actions (1-2 weeks)

### 3. Shader optimization audit

-   Review memory access patterns in compute shaders
-   Check workgroup size tuning for target hardware
-   Identify opportunities for shared memory usage

### 4. Batch processing review

-   Analyze command buffer submission patterns
-   Look for opportunities to reduce CPU-GPU sync points
-   Consider indirect dispatch for variable workloads

## Medium-Term Actions (If Needed)

### 5. Consider wgpu-hal only if:

-   Profiling shows >10% time in wgpu validation
-   Web support is explicitly dropped from requirements
-   Team has capacity for unsafe code maintenance

### 6. Platform-specific optimizations

-   Use subgroup operations where available (already implemented)
-   Consider platform-specific shader variants
-   Tune for specific GPU architectures

## Long-Term Considerations

### 7. Monitor wgpu development

-   wgpu continues to improve performance
-   New features may provide optimization opportunities
-   WebGPU spec stabilization may unlock new capabilities

### 8. Re-evaluate periodically

-   As the project evolves, requirements may change
-   New GPU abstraction libraries may emerge
-   Hardware capabilities continue to advance

## Decision Tree

```
Is performance acceptable?
├── Yes → No action needed
└── No → Profile the application
         ├── GPU-bound (>80% GPU time) → Optimize shaders
         ├── Memory-bound → Optimize data layout, reduce copies
         └── CPU-bound (>20% in wgpu) → Consider wgpu-hal
                                        ├── Web support needed? → Stay with wgpu
                                        └── Native only? → Evaluate wgpu-hal migration
```

## Success Metrics

| Metric        | Current | Target | Method                |
| ------------- | ------- | ------ | --------------------- |
| Tokens/second | ~40-45  | 50+    | Shader optimization   |
| Load time     | ~6s     | <4s    | Lazy loading, caching |
| Memory usage  | Varies  | -20%   | Buffer pooling        |
| CPU overhead  | ~5%     | <3%    | Batching improvements |

## Risk Assessment

| Action                 | Risk   | Mitigation                           |
| ---------------------- | ------ | ------------------------------------ |
| Stay with wgpu         | Low    | None needed                          |
| Shader optimization    | Low    | Benchmark before/after               |
| wgpu-hal migration     | High   | Extensive testing, abstraction layer |
| Platform-specific code | Medium | Feature flags, CI testing            |

## Conclusion

**Primary recommendation: Stay with wgpu and focus on shader/algorithmic optimizations.**

The research conclusively shows that:

1. gfx-hal is deprecated and not a viable option
2. wgpu-hal provides marginal benefits at high cost
3. GPU compute time dominates, making API overhead less relevant
4. Cross-platform support (including web) is valuable

The highest-ROI optimizations are in the compute shaders themselves, not in the GPU abstraction layer.

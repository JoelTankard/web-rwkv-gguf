# wgpu-hal Migration Guide (If Needed)

This document outlines what a migration from wgpu to wgpu-hal would entail, should it ever become necessary.

## When to Consider wgpu-hal

Only consider this migration if ALL of the following are true:

1. Web support is no longer required
2. Profiling shows >10% of total time in wgpu validation
3. Team has capacity for unsafe Rust maintenance
4. Performance gains justify the migration cost

## Current wgpu Usage in web-rwkv

### Key Integration Points

Based on analysis of the codebase:

```rust
// src/context.rs - Primary wgpu integration
use wgpu::{
    Adapter, BindGroup, Buffer, ComputePipeline, Device, Queue,
    Instance, Limits, Features, ...
};

pub struct Context {
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    // ... caches for pipelines, buffers, bindings
}
```

### wgpu APIs Used

| API               | Usage               | wgpu-hal Equivalent    |
| ----------------- | ------------------- | ---------------------- |
| `Instance`        | Adapter enumeration | `hal::Instance`        |
| `Adapter`         | Device creation     | `hal::ExposedAdapter`  |
| `Device`          | Resource creation   | `hal::Device`          |
| `Queue`           | Command submission  | `hal::Queue`           |
| `Buffer`          | GPU memory          | `hal::Buffer`          |
| `BindGroup`       | Shader bindings     | `hal::BindGroup`       |
| `ComputePipeline` | Compute shaders     | `hal::ComputePipeline` |
| `CommandEncoder`  | Command recording   | `hal::CommandEncoder`  |

## Migration Steps

### Phase 1: Abstraction Layer (1 week)

Create a trait-based abstraction over GPU operations:

```rust
pub trait GpuBackend {
    type Buffer;
    type Pipeline;
    type BindGroup;
    type CommandEncoder;

    fn create_buffer(&self, size: usize, usage: BufferUsage) -> Self::Buffer;
    fn create_pipeline(&self, shader: &str) -> Self::Pipeline;
    fn submit(&self, commands: Vec<Self::CommandEncoder>);
}

// Implement for wgpu
impl GpuBackend for WgpuBackend { ... }

// Later: implement for wgpu-hal
impl GpuBackend for WgpuHalBackend { ... }
```

### Phase 2: wgpu-hal Backend (2-3 weeks)

Implement the abstraction using wgpu-hal directly:

```rust
use wgpu_hal as hal;

pub struct WgpuHalBackend<A: hal::Api> {
    instance: A::Instance,
    adapter: hal::ExposedAdapter<A>,
    device: A::Device,
    queue: A::Queue,
}
```

Key differences from wgpu:

1. **No automatic validation** - Must validate all inputs manually
2. **Explicit barriers** - Must call `transition_buffers()` between usages
3. **Manual lifetime management** - Resources don't have automatic cleanup
4. **Static dispatch** - Must choose backend at compile time or use enums

### Phase 3: Validation Layer (1 week)

Implement minimal validation for debug builds:

```rust
#[cfg(debug_assertions)]
fn validate_buffer_usage(buffer: &Buffer, expected: BufferUsage) {
    assert!(buffer.usage.contains(expected), "Invalid buffer usage");
}

#[cfg(not(debug_assertions))]
fn validate_buffer_usage(_: &Buffer, _: BufferUsage) {}
```

### Phase 4: Testing & Benchmarking (1 week)

1. Run full test suite with both backends
2. Benchmark inference performance
3. Profile CPU overhead reduction
4. Validate numerical correctness

## Code Changes Required

### Context Creation

**Before (wgpu):**

```rust
let (device, queue) = adapter
    .request_device(&DeviceDescriptor { ... })
    .await?;
```

**After (wgpu-hal):**

```rust
let open_device = unsafe {
    adapter.adapter.open(
        features,
        &hal::DeviceDescriptor { ... },
    )?
};
let device = open_device.device;
let queue = open_device.queue;
```

### Buffer Creation

**Before (wgpu):**

```rust
let buffer = device.create_buffer(&BufferDescriptor {
    label: Some("my_buffer"),
    size: 1024,
    usage: BufferUsages::STORAGE,
    mapped_at_creation: false,
});
```

**After (wgpu-hal):**

```rust
let buffer = unsafe {
    device.create_buffer(&hal::BufferDescriptor {
        label: Some("my_buffer"),
        size: 1024,
        usage: hal::BufferUses::STORAGE_READ_WRITE,
        memory_flags: hal::MemoryFlags::empty(),
    })?
};
```

### Command Submission

**Before (wgpu):**

```rust
let mut encoder = device.create_command_encoder(&Default::default());
{
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(x, y, z);
}
queue.submit(Some(encoder.finish()));
```

**After (wgpu-hal):**

```rust
let mut encoder = unsafe { device.create_command_encoder(&Default::default())? };
unsafe {
    encoder.begin_compute_pass(&hal::ComputePassDescriptor { label: None });
    encoder.set_compute_pipeline(&pipeline);
    encoder.set_bind_group(&pipeline_layout, 0, &bind_group, &[]);
    encoder.dispatch([x, y, z]);
    encoder.end_compute_pass();
}
let command_buffer = unsafe { encoder.end_encoding()? };
unsafe { queue.submit(&[&command_buffer], None)? };
```

## Risk Assessment

| Risk                                    | Likelihood | Impact   | Mitigation                          |
| --------------------------------------- | ---------- | -------- | ----------------------------------- |
| Undefined behavior from incorrect usage | High       | Critical | Extensive testing, debug validation |
| Resource leaks                          | Medium     | High     | RAII wrappers, drop implementations |
| Platform-specific bugs                  | Medium     | Medium   | CI testing on all platforms         |
| Maintenance burden                      | High       | Medium   | Good abstraction layer              |

## Estimated Timeline

| Phase             | Duration      | Dependencies |
| ----------------- | ------------- | ------------ |
| Abstraction layer | 1 week        | None         |
| wgpu-hal backend  | 2-3 weeks     | Phase 1      |
| Validation layer  | 1 week        | Phase 2      |
| Testing           | 1 week        | Phase 3      |
| **Total**         | **5-6 weeks** |              |

## Conclusion

Migration to wgpu-hal is technically feasible but:

1. **High effort** - 5-6 weeks of development
2. **Increased risk** - Unsafe code, potential for UB
3. **Limited benefit** - Only reduces CPU overhead (2-5% of total time)
4. **Loses web support** - Major feature regression

**Recommendation: Do not migrate unless profiling proves CPU overhead is a significant bottleneck.**

# GPU Backend Architecture

## Overview

The hybrid backend system detects the available GPU at runtime and routes operations to the optimal backend while maintaining a unified API.

## GPU Detection

```rust
pub enum GpuBackend {
    Metal,      // macOS with Apple Silicon or AMD
    Cuda,       // NVIDIA GPUs (future)
    Vulkan,     // Cross-platform native (future)
    WebGpu,     // Universal fallback
}

pub struct GpuCapabilities {
    pub backend: GpuBackend,
    pub device_name: String,
    pub supports_f16: bool,
    pub supports_subgroups: bool,
    pub max_buffer_size: u64,
    pub max_compute_workgroup_size: u32,
}
```

### Detection Logic

```rust
impl GpuCapabilities {
    pub fn detect(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();

        let backend = match info.backend {
            wgpu::Backend::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal-backend"))]
                { GpuBackend::Metal }
                #[cfg(not(all(target_os = "macos", feature = "metal-backend")))]
                { GpuBackend::WebGpu }
            }
            wgpu::Backend::Vulkan if info.device_type == DeviceType::DiscreteGpu => {
                // Check for NVIDIA
                if info.name.contains("NVIDIA") {
                    #[cfg(feature = "cuda-backend")]
                    { GpuBackend::Cuda }
                    #[cfg(not(feature = "cuda-backend"))]
                    { GpuBackend::WebGpu }
                } else {
                    GpuBackend::WebGpu
                }
            }
            _ => GpuBackend::WebGpu,
        };

        Self {
            backend,
            device_name: info.name.clone(),
            supports_f16: adapter.features().contains(Features::SHADER_F16),
            supports_subgroups: adapter.features().contains(Features::SUBGROUP),
            // ... other capabilities
        }
    }
}
```

## Operation Routing

### Which Operations to Accelerate

Based on profiling, these operations consume the most time:

| Operation  | % of Inference Time | Accelerate?               |
| ---------- | ------------------- | ------------------------- |
| Q4K Matmul | 70-80%              | ✅ Yes - highest priority |
| Attention  | 10-15%              | ✅ Yes - significant      |
| LayerNorm  | 2-5%                | ❌ No - WebGPU sufficient |
| Softmax    | 1-2%                | ❌ No - WebGPU sufficient |
| Embedding  | 1-2%                | ❌ No - WebGPU sufficient |

### Router Interface

```rust
pub trait MatmulBackend {
    fn matmul_vec_q4k(
        &self,
        weights: &Q4KBuffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    );

    fn supports_batching(&self) -> bool;
    fn flush(&self); // Commit batched commands
}

pub struct BackendRouter {
    metal: Option<MetalBackend>,
    cuda: Option<CudaBackend>,
    webgpu: WebGpuBackend,
    active: GpuBackend,
}

impl BackendRouter {
    pub fn matmul(&self) -> &dyn MatmulBackend {
        match self.active {
            GpuBackend::Metal => self.metal.as_ref().unwrap(),
            GpuBackend::Cuda => self.cuda.as_ref().unwrap(),
            GpuBackend::WebGpu => &self.webgpu,
        }
    }
}
```

## Buffer Sharing Strategy

### Option A: Shared Memory (Preferred for Metal)

Metal and wgpu can share the same GPU memory on macOS:

```rust
// Create wgpu buffer with MAP_READ | MAP_WRITE for shared access
let wgpu_buffer = device.create_buffer(&BufferDescriptor {
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
    ..
});

// Get Metal buffer from wgpu buffer (unsafe, requires wgpu-hal)
let metal_buffer = unsafe {
    wgpu_buffer.as_hal::<wgpu::hal::api::Metal, _, _>(|buf| {
        buf.map(|b| b.raw_buffer())
    })
};
```

### Option B: Separate Buffers with Copy

If shared memory isn't available, copy data between backends:

```rust
pub struct DualBuffer {
    wgpu: wgpu::Buffer,
    metal: Option<metal::Buffer>,
    dirty: BufferDirty,
}

enum BufferDirty {
    Clean,
    WgpuDirty,
    MetalDirty,
}

impl DualBuffer {
    fn sync_to_metal(&mut self) { /* copy if dirty */ }
    fn sync_to_wgpu(&mut self) { /* copy if dirty */ }
}
```

## Command Batching

Critical for performance - batch all operations per token:

```rust
pub struct CommandBatcher {
    metal_commands: Option<metal::CommandBuffer>,
    pending_ops: Vec<PendingOp>,
}

impl CommandBatcher {
    pub fn begin_token(&mut self) {
        if let Some(ctx) = &self.metal_ctx {
            self.metal_commands = Some(ctx.queue().new_command_buffer());
        }
    }

    pub fn encode_matmul(&mut self, /* params */) {
        if let Some(cmd) = &self.metal_commands {
            MetalOps::encode_matmul_vec_q4k(ctx, cmd, /* params */);
        } else {
            // Fall back to WebGPU
            self.pending_ops.push(PendingOp::Matmul(/* params */));
        }
    }

    pub fn end_token(&mut self) {
        if let Some(cmd) = self.metal_commands.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
        // Execute WebGPU ops
        self.flush_wgpu();
    }
}
```

## Platform Support Matrix

| Platform              | Native Backend | Fallback | Expected Speed |
| --------------------- | -------------- | -------- | -------------- |
| macOS (Apple Silicon) | Metal          | WebGPU   | 4-5x baseline  |
| macOS (Intel/AMD)     | Metal          | WebGPU   | 2-3x baseline  |
| Windows (NVIDIA)      | CUDA (future)  | WebGPU   | TBD            |
| Windows (AMD/Intel)   | -              | WebGPU   | 1x baseline    |
| Linux (NVIDIA)        | CUDA (future)  | WebGPU   | TBD            |
| Linux (AMD)           | -              | WebGPU   | 1x baseline    |
| Browser (any)         | -              | WebGPU   | 1x baseline    |

## Feature Flags

```toml
[features]
default = ["native"]
native = ["subgroup-ops", "tokio"]
web = []

# Native acceleration backends
metal-backend = []  # macOS Metal acceleration
cuda-backend = []   # NVIDIA CUDA acceleration (future)

# Auto-detect and enable appropriate backend
auto-backend = ["metal-backend"]  # Add cuda-backend when ready
```

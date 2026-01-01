//! Metal context for managing device, queue, and pipelines.

use metal::{
    Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device, Library,
    MTLResourceOptions,
};

use super::kernels::METAL_KERNELS;

#[derive(Debug)]
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    #[allow(dead_code)]
    library: Library,
    pipelines: MetalPipelines,
}

#[derive(Debug)]
struct MetalPipelines {
    matmul_vec_q4k: ComputePipelineState,
    matmul_mat_q4k: ComputePipelineState,
}

impl MetalContext {
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let compile_options = metal::CompileOptions::new();
        compile_options.set_language_version(metal::MTLLanguageVersion::V2_4);
        compile_options.set_fast_math_enabled(true);

        let library = match device.new_library_with_source(METAL_KERNELS, &compile_options) {
            Ok(lib) => lib,
            Err(e) => {
                eprintln!("Failed to compile Metal kernels: {}", e);
                return None;
            }
        };

        let pipelines = Self::create_pipelines(&device, &library)?;

        log::info!(
            "Metal backend initialized: {} ({})",
            device.name(),
            if device.is_low_power() {
                "low power"
            } else {
                "high performance"
            }
        );

        Some(Self {
            device,
            queue,
            library,
            pipelines,
        })
    }

    fn create_pipelines(device: &Device, library: &Library) -> Option<MetalPipelines> {
        let matmul_vec_q4k = Self::create_pipeline(device, library, "matmul_vec_q4k")?;
        let matmul_mat_q4k = Self::create_pipeline(device, library, "matmul_mat_q4k")?;

        Some(MetalPipelines {
            matmul_vec_q4k,
            matmul_mat_q4k,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        function_name: &str,
    ) -> Option<ComputePipelineState> {
        let function = library.get_function(function_name, None).ok()?;
        let descriptor = ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                log::error!("Failed to create pipeline for {}: {}", function_name, e);
                e
            })
            .ok()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn matmul_vec_q4k_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.matmul_vec_q4k
    }

    pub fn matmul_mat_q4k_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.matmul_mat_q4k
    }

    pub fn create_buffer(&self, data: &[u8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    pub fn create_buffer_empty(&self, size: u64) -> Buffer {
        self.device
            .new_buffer(size, MTLResourceOptions::StorageModeShared)
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal context")
    }
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

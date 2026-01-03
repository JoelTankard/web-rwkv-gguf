//! Metal context for managing device, command queue, and pipeline cache.

use std::collections::HashMap;
use std::sync::Arc;

use metal::{
    Buffer as MetalBuffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device,
    Library, MTLResourceOptions,
};

use crate::tensor::TensorError;

/// Metal context holding device, queue, and cached pipelines.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    pipelines: HashMap<String, ComputePipelineState>,
    library: Option<Library>,
}

impl MetalContext {
    /// Create a new Metal context using the system default device.
    pub fn new() -> Result<Self, TensorError> {
        let device =
            Device::system_default().ok_or_else(|| crate::tensor::TensorErrorKind::Deduce)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            queue,
            pipelines: HashMap::new(),
            library: None,
        })
    }

    /// Get the Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the command queue.
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Compile Metal shader source and cache the library.
    pub fn compile_library(&mut self, source: &str) -> Result<(), TensorError> {
        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .map_err(|e| {
                eprintln!("Metal shader compilation error: {}", e);
                crate::tensor::TensorErrorKind::Deduce
            })?;
        self.library = Some(library);
        Ok(())
    }

    /// Get or create a compute pipeline for the given kernel function.
    pub fn get_pipeline(
        &mut self,
        function_name: &str,
    ) -> Result<&ComputePipelineState, TensorError> {
        if self.pipelines.contains_key(function_name) {
            return Ok(self.pipelines.get(function_name).unwrap());
        }

        let library = self
            .library
            .as_ref()
            .ok_or(crate::tensor::TensorErrorKind::Deduce)?;

        let function = library
            .get_function(function_name, None)
            .map_err(|_| crate::tensor::TensorErrorKind::Deduce)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|_| crate::tensor::TensorErrorKind::Deduce)?;

        self.pipelines.insert(function_name.to_string(), pipeline);
        Ok(self.pipelines.get(function_name).unwrap())
    }

    /// Create a new Metal buffer with the given size.
    pub fn new_buffer(&self, size: u64, options: MTLResourceOptions) -> MetalBuffer {
        self.device.new_buffer(size, options)
    }

    /// Create a new Metal buffer from data.
    pub fn new_buffer_with_data<T>(&self, data: &[T], options: MTLResourceOptions) -> MetalBuffer {
        let size = (data.len() * std::mem::size_of::<T>()) as u64;
        let buffer = self.device.new_buffer(size, options);
        unsafe {
            let ptr = buffer.contents() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buffer
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal context")
    }
}

/// Thread-safe wrapper for MetalContext.
pub type SharedMetalContext = Arc<std::sync::Mutex<MetalContext>>;

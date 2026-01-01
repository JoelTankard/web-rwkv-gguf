//! Metal context for managing device, queue, and pipelines.

use std::collections::HashMap;
use std::sync::RwLock;

use metal::{
    Buffer, CommandQueue, ComputePipelineDescriptor, ComputePipelineState, Device, Library,
    MTLResourceOptions,
};

use super::kernels::METAL_KERNELS;

/// Unique identifier for a wgpu buffer, used as cache key for Metal buffers.
/// Uses the tensor's unique ID to identify buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(u64);

impl BufferId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Create a BufferId from a tensor's unique ID.
    pub fn from_tensor_id<T>(id: uid::Id<T>) -> Self {
        // uid::Id::get() returns usize, cast to u64
        Self(id.get() as u64)
    }

    /// Get the raw u64 ID value.
    pub fn get(&self) -> u64 {
        self.0
    }
}

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    #[allow(dead_code)]
    library: Library,
    pipelines: MetalPipelines,
    /// Cache of Metal buffers created from wgpu buffer data.
    /// Key is the wgpu buffer's unique ID.
    buffer_cache: RwLock<HashMap<BufferId, Buffer>>,
}

#[derive(Debug)]
struct MetalPipelines {
    matmul_vec_q4k: ComputePipelineState,
    matmul_mat_q4k: ComputePipelineState,
    // Phase 1: Simple ops
    add_fp16: ComputePipelineState,
    add_f32_to_fp16: ComputePipelineState,
    mul_fp16: ComputePipelineState,
    blit_fp16: ComputePipelineState,
    affine_fp16: ComputePipelineState,
    layer_norm_fp16: ComputePipelineState,
    group_norm_fp16: ComputePipelineState,
    token_shift_fp16: ComputePipelineState,
    token_shift_reversed_fp16: ComputePipelineState,
    lerp_fp16: ComputePipelineState,
    // Phase 2: FP16 matmul and normalization
    matmul_vec_fp16: ComputePipelineState,
    matmul_vec_fp16_tanh: ComputePipelineState,
    matmul_vec_fp16_squared_relu: ComputePipelineState,
    l2_norm_fp16: ComputePipelineState,
    rms_norm_fp16: ComputePipelineState,
    squared_relu_fp16: ComputePipelineState,
    tanh_fp16: ComputePipelineState,
    sigmoid_fp16: ComputePipelineState,
    silu_fp16: ComputePipelineState,
    // Phase 3: Time Mix V7
    pack_kvakk_fp16: ComputePipelineState,
    time_first_fp16: ComputePipelineState,
    time_mix_v7_fp16: ComputePipelineState,
    time_mix_v7_single_fp16: ComputePipelineState,
    control_k_v7_fp16: ComputePipelineState,
    // Phase 4: Channel Mix and Fused Ops
    channel_mix_v7_fp16: ComputePipelineState,
    channel_mix_fp16: ComputePipelineState,
    add_tanh_fp16: ComputePipelineState,
    add_sigmoid_fp16: ComputePipelineState,
    add_squared_relu_fp16: ComputePipelineState,
    add_layer_norm_fp16: ComputePipelineState,
    softmax_fp16: ComputePipelineState,
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
            buffer_cache: RwLock::new(HashMap::new()),
        })
    }

    fn create_pipelines(device: &Device, library: &Library) -> Option<MetalPipelines> {
        let matmul_vec_q4k = Self::create_pipeline(device, library, "matmul_vec_q4k")?;
        let matmul_mat_q4k = Self::create_pipeline(device, library, "matmul_mat_q4k")?;

        // Phase 1: Simple ops
        let add_fp16 = Self::create_pipeline(device, library, "add_fp16")?;
        let add_f32_to_fp16 = Self::create_pipeline(device, library, "add_f32_to_fp16")?;
        let mul_fp16 = Self::create_pipeline(device, library, "mul_fp16")?;
        let blit_fp16 = Self::create_pipeline(device, library, "blit_fp16")?;
        let affine_fp16 = Self::create_pipeline(device, library, "affine_fp16")?;
        let layer_norm_fp16 = Self::create_pipeline(device, library, "layer_norm_fp16")?;
        let group_norm_fp16 = Self::create_pipeline(device, library, "group_norm_fp16")?;
        let token_shift_fp16 = Self::create_pipeline(device, library, "token_shift_fp16")?;
        let token_shift_reversed_fp16 =
            Self::create_pipeline(device, library, "token_shift_reversed_fp16")?;
        let lerp_fp16 = Self::create_pipeline(device, library, "lerp_fp16")?;

        // Phase 2: FP16 matmul and normalization
        let matmul_vec_fp16 = Self::create_pipeline(device, library, "matmul_vec_fp16")?;
        let matmul_vec_fp16_tanh = Self::create_pipeline(device, library, "matmul_vec_fp16_tanh")?;
        let matmul_vec_fp16_squared_relu =
            Self::create_pipeline(device, library, "matmul_vec_fp16_squared_relu")?;
        let l2_norm_fp16 = Self::create_pipeline(device, library, "l2_norm_fp16")?;
        let rms_norm_fp16 = Self::create_pipeline(device, library, "rms_norm_fp16")?;
        let squared_relu_fp16 = Self::create_pipeline(device, library, "squared_relu_fp16")?;
        let tanh_fp16 = Self::create_pipeline(device, library, "tanh_fp16")?;
        let sigmoid_fp16 = Self::create_pipeline(device, library, "sigmoid_fp16")?;
        let silu_fp16 = Self::create_pipeline(device, library, "silu_fp16")?;

        // Phase 3: Time Mix V7
        let pack_kvakk_fp16 = Self::create_pipeline(device, library, "pack_kvakk_fp16")?;
        let time_first_fp16 = Self::create_pipeline(device, library, "time_first_fp16")?;
        let time_mix_v7_fp16 = Self::create_pipeline(device, library, "time_mix_v7_fp16")?;
        let time_mix_v7_single_fp16 =
            Self::create_pipeline(device, library, "time_mix_v7_single_fp16")?;
        let control_k_v7_fp16 = Self::create_pipeline(device, library, "control_k_v7_fp16")?;

        // Phase 4: Channel Mix and Fused Ops
        let channel_mix_v7_fp16 = Self::create_pipeline(device, library, "channel_mix_v7_fp16")?;
        let channel_mix_fp16 = Self::create_pipeline(device, library, "channel_mix_fp16")?;
        let add_tanh_fp16 = Self::create_pipeline(device, library, "add_tanh_fp16")?;
        let add_sigmoid_fp16 = Self::create_pipeline(device, library, "add_sigmoid_fp16")?;
        let add_squared_relu_fp16 =
            Self::create_pipeline(device, library, "add_squared_relu_fp16")?;
        let add_layer_norm_fp16 = Self::create_pipeline(device, library, "add_layer_norm_fp16")?;
        let softmax_fp16 = Self::create_pipeline(device, library, "softmax_fp16")?;

        Some(MetalPipelines {
            matmul_vec_q4k,
            matmul_mat_q4k,
            add_fp16,
            add_f32_to_fp16,
            mul_fp16,
            blit_fp16,
            affine_fp16,
            layer_norm_fp16,
            group_norm_fp16,
            token_shift_fp16,
            token_shift_reversed_fp16,
            lerp_fp16,
            matmul_vec_fp16,
            matmul_vec_fp16_tanh,
            matmul_vec_fp16_squared_relu,
            l2_norm_fp16,
            rms_norm_fp16,
            squared_relu_fp16,
            tanh_fp16,
            sigmoid_fp16,
            silu_fp16,
            pack_kvakk_fp16,
            time_first_fp16,
            time_mix_v7_fp16,
            time_mix_v7_single_fp16,
            control_k_v7_fp16,
            channel_mix_v7_fp16,
            channel_mix_fp16,
            add_tanh_fp16,
            add_sigmoid_fp16,
            add_squared_relu_fp16,
            add_layer_norm_fp16,
            softmax_fp16,
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

    // Phase 1: Simple ops pipeline accessors
    pub fn add_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_fp16
    }

    pub fn add_f32_to_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_f32_to_fp16
    }

    pub fn mul_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.mul_fp16
    }

    pub fn blit_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.blit_fp16
    }

    pub fn affine_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.affine_fp16
    }

    pub fn layer_norm_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.layer_norm_fp16
    }

    pub fn group_norm_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.group_norm_fp16
    }

    pub fn token_shift_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.token_shift_fp16
    }

    pub fn token_shift_reversed_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.token_shift_reversed_fp16
    }

    pub fn lerp_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.lerp_fp16
    }

    // Phase 2: FP16 matmul and normalization pipeline accessors
    pub fn matmul_vec_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.matmul_vec_fp16
    }

    pub fn matmul_vec_fp16_tanh_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.matmul_vec_fp16_tanh
    }

    pub fn matmul_vec_fp16_squared_relu_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.matmul_vec_fp16_squared_relu
    }

    pub fn l2_norm_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.l2_norm_fp16
    }

    pub fn rms_norm_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.rms_norm_fp16
    }

    pub fn squared_relu_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.squared_relu_fp16
    }

    pub fn tanh_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.tanh_fp16
    }

    pub fn sigmoid_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.sigmoid_fp16
    }

    pub fn silu_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.silu_fp16
    }

    // Phase 3: Time Mix V7 pipeline accessors
    pub fn pack_kvakk_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.pack_kvakk_fp16
    }

    pub fn time_first_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.time_first_fp16
    }

    pub fn time_mix_v7_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.time_mix_v7_fp16
    }

    pub fn time_mix_v7_single_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.time_mix_v7_single_fp16
    }

    pub fn control_k_v7_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.control_k_v7_fp16
    }

    // Phase 4: Channel Mix and Fused Ops pipeline accessors
    pub fn channel_mix_v7_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.channel_mix_v7_fp16
    }

    pub fn channel_mix_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.channel_mix_fp16
    }

    pub fn add_tanh_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_tanh_fp16
    }

    pub fn add_sigmoid_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_sigmoid_fp16
    }

    pub fn add_squared_relu_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_squared_relu_fp16
    }

    pub fn add_layer_norm_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.add_layer_norm_fp16
    }

    pub fn softmax_fp16_pipeline(&self) -> &ComputePipelineState {
        &self.pipelines.softmax_fp16
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

    /// Get or create a Metal buffer for the given wgpu buffer data.
    /// The buffer is cached by ID for reuse.
    pub fn get_or_create_buffer(&self, id: BufferId, data: &[u8]) -> Buffer {
        // Check cache first (read lock)
        {
            let cache = self.buffer_cache.read().unwrap();
            if let Some(buffer) = cache.get(&id) {
                return buffer.clone();
            }
        }

        // Create new buffer and cache it (write lock)
        let buffer = self.create_buffer(data);
        {
            let mut cache = self.buffer_cache.write().unwrap();
            cache.insert(id, buffer.clone());
        }
        buffer
    }

    /// Check if a Metal buffer exists in the cache for the given ID.
    pub fn has_buffer(&self, id: BufferId) -> bool {
        let cache = self.buffer_cache.read().unwrap();
        cache.contains_key(&id)
    }

    /// Get a cached Metal buffer by ID, if it exists.
    pub fn get_buffer(&self, id: BufferId) -> Option<Buffer> {
        let cache = self.buffer_cache.read().unwrap();
        cache.get(&id).cloned()
    }
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.device.name())
            .field(
                "buffer_cache_size",
                &self.buffer_cache.read().unwrap().len(),
            )
            .finish()
    }
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new().expect("Failed to create Metal context")
    }
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

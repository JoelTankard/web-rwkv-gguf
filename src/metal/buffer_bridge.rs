//! Buffer bridge for zero-copy sharing between wgpu and Metal.

use std::sync::Arc;
use wgpu::Buffer;

/// Bridge for extracting Metal buffers from wgpu buffers.
pub struct BufferBridge;

impl BufferBridge {
    /// Extract the underlying Metal buffer from a wgpu buffer.
    ///
    /// # Safety
    ///
    /// This uses wgpu's HAL layer to access the underlying Metal buffer.
    /// The caller must ensure the wgpu buffer remains valid for the lifetime
    /// of the returned Metal buffer.
    ///
    /// # Returns
    ///
    /// Returns `Some(metal::Buffer)` if:
    /// - Running on macOS with Metal backend
    /// - The buffer was created by wgpu using Metal
    ///
    /// Returns `None` if:
    /// - Not running on Metal backend
    /// - Buffer has been destroyed
    /// - Any other error accessing the HAL layer
    #[cfg(target_os = "macos")]
    pub unsafe fn extract_metal_buffer(wgpu_buffer: &Buffer) -> Option<metal::Buffer> {
        use wgpu::hal::api::Metal;

        // Get the HAL buffer via wgpu's as_hal API
        // This returns a guard that derefs to wgpu_hal::metal::Buffer
        let hal_buffer_guard = wgpu_buffer.as_hal::<Metal>()?;

        // The HAL buffer has a private `raw` field of type metal::Buffer
        // We need to access it via the as_raw() method which returns a NonNull pointer
        // Then reconstruct a metal::Buffer from that pointer
        //
        // wgpu_hal::metal::Buffer::as_raw() returns BufferPtr (NonNull<MTLBuffer>)
        // We can use metal::Buffer::from_ptr() to create a Buffer from the raw pointer
        //
        // Note: We must retain the buffer since we're creating a new reference to it
        let hal_buffer: &wgpu::hal::metal::Buffer = &*hal_buffer_guard;

        // Access the raw pointer - the HAL buffer stores it internally
        // We use std::ptr operations to get at the underlying metal::Buffer
        //
        // The wgpu_hal::metal::Buffer struct layout is:
        //   pub struct Buffer { raw: metal::Buffer, size: u64 }
        //
        // We can transmute the reference to access the first field
        let raw_ptr: *const metal::Buffer = hal_buffer as *const _ as *const metal::Buffer;
        let metal_buffer: &metal::Buffer = &*raw_ptr;

        // Clone to get an owned buffer (this increments the reference count)
        Some(metal_buffer.clone())
    }

    /// Extract Metal buffer from an Arc<wgpu::Buffer>.
    #[cfg(target_os = "macos")]
    pub unsafe fn extract_from_arc(wgpu_buffer: &Arc<Buffer>) -> Option<metal::Buffer> {
        Self::extract_metal_buffer(wgpu_buffer.as_ref())
    }
}

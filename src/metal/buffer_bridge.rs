//! Bridge between wgpu and Metal buffers.
//!
//! This module provides utilities for Metal backend detection and buffer management.
//! On macOS, wgpu uses Metal under the hood, so buffers share GPU memory.

use std::ops::Deref;

use metal::foreign_types::ForeignType;
use metal::Buffer as MetalBuffer;
use objc::{msg_send, runtime::Object, sel, sel_impl};

/// Check if the wgpu device is using Metal backend.
pub fn is_metal_backend(adapter: &wgpu::Adapter) -> bool {
    adapter.get_info().backend == wgpu::Backend::Metal
}

/// Extract the underlying Metal buffer from a wgpu buffer.
///
/// This is zero-copy - the Metal buffer shares memory with the wgpu buffer.
///
/// # Safety
/// The caller must ensure proper synchronization between wgpu and Metal operations.
/// The returned buffer is only valid while the wgpu buffer is alive.
pub fn get_metal_buffer_from_wgpu(wgpu_buffer: &wgpu::Buffer) -> Option<MetalBuffer> {
    unsafe {
        // Get the hal buffer reference
        let hal_buffer_guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>()?;

        // The guard derefs to wgpu_hal::metal::Buffer
        // wgpu_hal::metal::Buffer struct layout: { raw: metal::Buffer, size: u64 }
        // metal::Buffer is a wrapper around *mut Object (MTLBuffer)
        // We read the first field (the raw pointer) directly
        let hal_buffer_ptr = &*hal_buffer_guard as *const _ as *const *mut Object;
        let raw_ptr: *mut Object = std::ptr::read(hal_buffer_ptr);

        if raw_ptr.is_null() {
            return None;
        }

        // Create a Metal buffer from the raw pointer
        // Retain to increment reference count since we're creating a new handle
        let _: () = objc::msg_send![raw_ptr, retain];
        Some(MetalBuffer::from_ptr(raw_ptr as *mut _))
    }
}

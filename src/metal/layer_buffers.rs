//! Metal layer buffers for intermediate tensors during pure Metal layer execution.
//!
//! These buffers are allocated once and reused across all layers to minimize
//! memory allocation overhead during inference.

use std::sync::Arc;

use metal::Buffer;

use super::MetalContext;
use crate::metal::buffer_bridge::get_metal_buffer_from_wgpu;
use crate::runtime::v7::Runtime;
use crate::tensor::{TensorGpu, TensorShape};

/// Metal buffers for all intermediate tensors used during layer execution.
/// These are extracted from the wgpu Runtime buffers for zero-copy Metal access.
#[derive(Debug)]
pub struct MetalLayerBuffers {
    // Cursor buffer for token positions
    pub cursors: Buffer,

    // Main hidden state
    pub x: Buffer,

    // Attention intermediates
    pub att_x: Buffer,
    pub att_v0: Buffer,
    pub att_rx: Buffer,
    pub att_wx: Buffer,
    pub att_kx: Buffer,
    pub att_vx: Buffer,
    pub att_ax: Buffer,
    pub att_gx: Buffer,
    pub att_r: Buffer,
    pub att_w: Buffer,
    pub att_k: Buffer,
    pub att_v: Buffer,
    pub att_a: Buffer,
    pub att_g: Buffer,
    pub att_o: Buffer,
    pub att_kk: Buffer,
    pub att_vv: Buffer,
    pub att_n: Buffer,

    // LoRA intermediates
    pub aux_w: Buffer,
    pub aux_a: Buffer,
    pub aux_g: Buffer,
    pub aux_v: Buffer,

    // FFN intermediates
    pub ffn_x: Buffer,
    pub ffn_kx: Buffer,
    pub ffn_k: Buffer,
    pub ffn_v: Buffer,

    // Dimensions for dispatch
    pub num_emb: u32,
    pub num_token: u32,
    pub num_head: u32,
    pub head_size: u32,
    pub num_hidden: u32,
}

impl MetalLayerBuffers {
    /// Create Metal layer buffers from wgpu Runtime buffers.
    /// Returns None if any buffer extraction fails.
    pub fn from_runtime<F: crate::num::Float>(
        runtime: &Runtime<F>,
        num_emb: usize,
        num_head: usize,
        num_hidden: usize,
    ) -> Option<Self> {
        let head_size = num_emb / num_head;
        let num_token = runtime.x.shape()[1];

        Some(Self {
            cursors: get_metal_buffer_from_wgpu(&runtime.cursors.buffer)?,
            x: get_metal_buffer_from_wgpu(&runtime.x.buffer)?,
            att_x: get_metal_buffer_from_wgpu(&runtime.att_x.buffer)?,
            att_v0: get_metal_buffer_from_wgpu(&runtime.att_v0.buffer)?,
            att_rx: get_metal_buffer_from_wgpu(&runtime.att_rx.buffer)?,
            att_wx: get_metal_buffer_from_wgpu(&runtime.att_wx.buffer)?,
            att_kx: get_metal_buffer_from_wgpu(&runtime.att_kx.buffer)?,
            att_vx: get_metal_buffer_from_wgpu(&runtime.att_vx.buffer)?,
            att_ax: get_metal_buffer_from_wgpu(&runtime.att_ax.buffer)?,
            att_gx: get_metal_buffer_from_wgpu(&runtime.att_gx.buffer)?,
            att_r: get_metal_buffer_from_wgpu(&runtime.att_r.buffer)?,
            att_w: get_metal_buffer_from_wgpu(&runtime.att_w.buffer)?,
            att_k: get_metal_buffer_from_wgpu(&runtime.att_k.buffer)?,
            att_v: get_metal_buffer_from_wgpu(&runtime.att_v.buffer)?,
            att_a: get_metal_buffer_from_wgpu(&runtime.att_a.buffer)?,
            att_g: get_metal_buffer_from_wgpu(&runtime.att_g.buffer)?,
            att_o: get_metal_buffer_from_wgpu(&runtime.att_o.buffer)?,
            att_kk: get_metal_buffer_from_wgpu(&runtime.att_kk.buffer)?,
            att_vv: get_metal_buffer_from_wgpu(&runtime.att_vv.buffer)?,
            att_n: get_metal_buffer_from_wgpu(&runtime.att_n.buffer)?,
            aux_w: get_metal_buffer_from_wgpu(&runtime.aux_w.buffer)?,
            aux_a: get_metal_buffer_from_wgpu(&runtime.aux_a.buffer)?,
            aux_g: get_metal_buffer_from_wgpu(&runtime.aux_g.buffer)?,
            aux_v: get_metal_buffer_from_wgpu(&runtime.aux_v.buffer)?,
            ffn_x: get_metal_buffer_from_wgpu(&runtime.ffn_x.buffer)?,
            ffn_kx: get_metal_buffer_from_wgpu(&runtime.ffn_kx.buffer)?,
            ffn_k: get_metal_buffer_from_wgpu(&runtime.ffn_k.buffer)?,
            ffn_v: get_metal_buffer_from_wgpu(&runtime.ffn_v.buffer)?,
            num_emb: num_emb as u32,
            num_token: num_token as u32,
            num_head: num_head as u32,
            head_size: head_size as u32,
            num_hidden: num_hidden as u32,
        })
    }

    /// Get the total element count for the main tensors (num_emb * num_token).
    pub fn main_count(&self) -> u32 {
        self.num_emb * self.num_token
    }

    /// Get the hidden layer element count (num_hidden * num_token).
    pub fn hidden_count(&self) -> u32 {
        self.num_hidden * self.num_token
    }
}

/// Metal buffer for layer state (attention and FFN state).
///
/// The state tensor has shape [batch, head_size + 1, num_emb] stored as vec4<f32>.
/// - Row 0: token shift state (previous token's hidden)
/// - Rows 1..head_size+1: HEAD_SIZE state rows for the recurrence
///
/// Both att and ffn share the same underlying buffer but at different offsets.
#[derive(Debug)]
pub struct MetalLayerState {
    /// The full state buffer for this layer
    pub buffer: Buffer,
    /// Offset in bytes to the attention state portion
    pub att_offset: u64,
    /// Offset in bytes to the FFN state portion  
    pub ffn_offset: u64,
    /// Stride per state row (num_emb) - used by time_mix kernel
    pub state_stride: u32,
    /// Number of channels (num_emb)
    pub num_emb: u32,
    /// Head size for computing offsets
    pub head_size: u32,
}

impl MetalLayerState {
    /// Create Metal state from wgpu state tensor.
    ///
    /// The state tensor layout is [num_emb, head_size + 2, batch, 1] where:
    /// - att state uses columns 0..head_size+1
    /// - ffn state uses column head_size+1
    pub fn from_state(
        state_buffer: &wgpu::Buffer,
        num_emb: usize,
        head_size: usize,
    ) -> Option<Self> {
        let buffer = get_metal_buffer_from_wgpu(state_buffer)?;

        // State layout: [batch, head_size + 1, num_emb] stored as vec4<f32>
        // - Row 0: token shift state
        // - Rows 1..head_size+1: recurrence state matrix
        // Each row has num_emb elements (C)
        let att_offset = 0u64;
        let ffn_offset = ((head_size + 1) * num_emb * 4) as u64; // f32 = 4 bytes
        let state_stride = num_emb as u32; // C elements per row

        Some(Self {
            buffer,
            att_offset,
            ffn_offset,
            state_stride,
            num_emb: num_emb as u32,
            head_size: head_size as u32,
        })
    }

    /// Get the attention state buffer (same buffer, att_offset should be used)
    pub fn att(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the FFN state buffer (same buffer, ffn_offset should be used)
    pub fn ffn(&self) -> &Buffer {
        &self.buffer
    }
}

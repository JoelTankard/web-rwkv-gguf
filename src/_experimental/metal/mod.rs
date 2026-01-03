//! Native Metal backend for accelerated operations on macOS.
//!
//! This module provides direct Metal access for operations that benefit from
//! native Metal SIMD instructions, particularly Int8 matrix multiplication.
//!
//! # Feature Flag
//!
//! Enable with `--features metal-acceleration`
//!
//! # Architecture
//!
//! The Metal backend integrates with the lazy tensor graph system:
//! 1. Operations are built lazily into a compute graph
//! 2. At resolve time, eligible subgraphs are compiled to Metal command buffers
//! 3. Execution happens with minimal sync points
//!
//! # Performance
//!
//! - WebGPU Int8: ~10-15 tok/s (no native int8 SIMD)
//! - Metal Int8: ~150-200 tok/s (native simdgroup_matrix)

mod buffer_bridge;
mod context;
pub mod kernels;
pub mod ops;
pub mod v7_kernels;
pub mod v7_layer;

pub use buffer_bridge::BufferBridge;
pub use context::MetalContext;
pub use ops::{execute_metal_ffn_q4k, sync_pending_metal_commands};
pub use v7_layer::execute_metal_v7_layer;

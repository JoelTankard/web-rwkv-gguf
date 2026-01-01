//! Native Metal backend for accelerated inference on macOS.
//!
//! This module provides hardware-accelerated quantized matrix multiplication
//! using Metal's `simdgroup_matrix` operations, which can significantly
//! outperform WebGPU's software dequantization approach.

mod context;
mod kernels;
mod ops;

pub use context::MetalContext;
pub use ops::MetalOps;

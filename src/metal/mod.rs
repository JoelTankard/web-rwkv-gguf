//! Native Metal backend for accelerated inference on macOS.
//!
//! This module provides hardware-accelerated quantized matrix multiplication
//! using Metal's `simdgroup_matrix` operations, which can significantly
//! outperform WebGPU's software dequantization approach.
//!
//! ## Architecture
//!
//! The Metal backend provides three levels of abstraction:
//!
//! 1. **MetalOps** - Low-level kernel encoders for individual operations
//! 2. **MetalBatcher** - Batches multiple operations into a single command buffer
//! 3. **MetalLayerExecutor** - High-level executor for entire RWKV layers
//! 4. **MetalLayerDispatcher** - Pure Metal layer execution (sync once per layer)
//!
//! For best performance, use `MetalLayerDispatcher` which executes entire layers
//! in pure Metal, synchronizing only once per layer instead of per-operation.

mod batcher;
mod buffer_bridge;
mod context;
mod kernels;
mod layer_buffers;
mod layer_dispatcher;
mod layer_executor;
mod layer_weights;
mod ops;

pub use batcher::MetalBatcher;
pub use buffer_bridge::{get_metal_buffer_from_wgpu, is_metal_backend};
pub use context::{BufferId, MetalContext};
pub use layer_buffers::{MetalLayerBuffers, MetalLayerState};
pub use layer_dispatcher::MetalLayerDispatcher;
pub use layer_executor::{LayerConfig, MetalLayerExecutor};
pub use layer_weights::{MetalLayerWeights, MetalMatrixWeights, MetalModelWeights};
pub use ops::MetalOps;

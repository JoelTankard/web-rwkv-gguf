//! Metal command batching for efficient GPU execution.
//!
//! This module provides a command batcher that collects Metal operations
//! and executes them in a single command buffer for optimal performance.

use std::sync::Arc;

use metal::{Buffer, CommandBuffer};

use super::{context::BufferId, MetalContext, MetalOps};

/// Batches Metal operations for efficient execution.
///
/// Usage:
/// 1. Call `begin()` at the start of a token
/// 2. Call `encode_matmul_vec_q4k()` for each matmul operation
/// 3. Call `commit_and_wait()` at the end of the token
pub struct MetalBatcher {
    ctx: Arc<MetalContext>,
    command_buffer: Option<CommandBuffer>,
}

impl MetalBatcher {
    pub fn new(ctx: Arc<MetalContext>) -> Self {
        Self {
            ctx,
            command_buffer: None,
        }
    }

    /// Begin a new batch of operations.
    /// Creates a new command buffer for encoding.
    pub fn begin(&mut self) {
        let cmd = self.ctx.queue().new_command_buffer().to_owned();
        self.command_buffer = Some(cmd);
    }

    /// Check if a batch is currently active.
    pub fn is_active(&self) -> bool {
        self.command_buffer.is_some()
    }

    /// Encode a Q4K vector-matrix multiplication into the current batch.
    /// Panics if `begin()` was not called first.
    pub fn encode_matmul_vec_q4k(
        &self,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let cmd = self
            .command_buffer
            .as_ref()
            .expect("MetalBatcher::begin() must be called before encoding operations");
        MetalOps::encode_matmul_vec_q4k(&self.ctx, cmd, weights, input, output, k, m);
    }

    /// Commit the current batch and wait for completion.
    /// Returns the number of operations that were batched.
    pub fn commit_and_wait(&mut self) {
        if let Some(cmd) = self.command_buffer.take() {
            cmd.commit();
            cmd.wait_until_completed();
        }
    }

    /// Get the underlying Metal context.
    pub fn context(&self) -> &Arc<MetalContext> {
        &self.ctx
    }

    /// Get or create a Metal buffer for the given data.
    /// The buffer is cached by ID for reuse.
    pub fn get_or_create_buffer(&self, id: BufferId, data: &[u8]) -> Buffer {
        self.ctx.get_or_create_buffer(id, data)
    }

    /// Create an empty Metal buffer of the given size.
    pub fn create_buffer_empty(&self, size: u64) -> Buffer {
        self.ctx.create_buffer_empty(size)
    }
}

impl std::fmt::Debug for MetalBatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBatcher")
            .field("active", &self.is_active())
            .finish()
    }
}

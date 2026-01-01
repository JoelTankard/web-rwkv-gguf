//! Metal operations for quantized matrix multiplication.

use metal::{Buffer, CommandBufferRef, MTLSize};

use super::context::MetalContext;

pub struct MetalOps;

impl MetalOps {
    pub fn matmul_vec_q4k(
        ctx: &MetalContext,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_vec_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        // New kernel: 4 simdgroups per threadgroup, each simdgroup (32 threads) handles one row
        // So 128 threads per threadgroup, processing 4 rows per threadgroup
        let simdgroups_per_tg = 4u32;
        let threads_per_simdgroup = 32u32;
        let threads_per_tg = simdgroups_per_tg * threads_per_simdgroup; // 128
        let rows_per_tg = simdgroups_per_tg; // 4

        let num_threadgroups = (m + rows_per_tg - 1) / rows_per_tg;

        let thread_groups = MTLSize::new(num_threadgroups as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(threads_per_tg as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Encode matmul into an existing command buffer (for batching)
    pub fn encode_matmul_vec_q4k(
        ctx: &MetalContext,
        command_buffer: &CommandBufferRef,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
    ) {
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_vec_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );

        let simdgroups_per_tg = 4u32;
        let threads_per_simdgroup = 32u32;
        let threads_per_tg = simdgroups_per_tg * threads_per_simdgroup;
        let rows_per_tg = simdgroups_per_tg;
        let num_threadgroups = (m + rows_per_tg - 1) / rows_per_tg;

        let thread_groups = MTLSize::new(num_threadgroups as u64, 1, 1);
        let threads_per_threadgroup = MTLSize::new(threads_per_tg as u64, 1, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();
    }

    pub fn matmul_mat_q4k(
        ctx: &MetalContext,
        weights: &Buffer,
        input: &Buffer,
        output: &Buffer,
        k: u32,
        m: u32,
        t: u32,
    ) {
        let command_buffer = ctx.queue().new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(ctx.matmul_mat_q4k_pipeline());
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &m as *const u32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &t as *const u32 as *const _,
        );

        let tile_m = 8u32;
        let tile_t = 8u32;

        let thread_groups = MTLSize::new(
            ((m + tile_m - 1) / tile_m) as u64,
            ((t + tile_t - 1) / tile_t) as u64,
            1,
        );
        let threads_per_threadgroup = MTLSize::new(tile_m as u64, tile_t as u64, 1);

        encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

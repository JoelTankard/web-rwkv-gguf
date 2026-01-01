//! Metal Shading Language kernels for quantized matrix multiplication.
//!
//! These kernels use `simdgroup_matrix` operations for hardware-accelerated
//! matrix multiplication with inline Q4_K dequantization.

pub const METAL_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Q4_K block structure (144 bytes per 256 elements)
// Layout: [d: f16, dmin: f16, scales: 12B, qs: 128B]
constant uint QK_K = 256;
constant uint K_SCALE_SIZE = 12;

// Optimized: Precompute scale/min for a sub-block
inline void get_scale_min_k4(
    const device uint8_t* scales,
    uint sub_block,
    thread uint8_t& sc,
    thread uint8_t& m
) {
    if (sub_block < 4) {
        sc = scales[sub_block] & 0x3F;
        m = scales[sub_block + 4] & 0x3F;
    } else {
        sc = (scales[sub_block + 4] & 0xF) | ((scales[sub_block - 4] >> 6) << 4);
        m = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
    }
}

// Optimized: Dequantize 8 values at once from a sub-block
inline void dequant_8(
    half d,
    half dmin,
    uint8_t sc,
    uint8_t m,
    const device uint8_t* qs,
    uint offset,
    thread float4& out0,
    thread float4& out1
) {
    float fd = float(d) * float(sc);
    float fm = float(dmin) * float(m);
    
    uint8_t q0 = qs[offset + 0];
    uint8_t q1 = qs[offset + 1];
    uint8_t q2 = qs[offset + 2];
    uint8_t q3 = qs[offset + 3];
    
    out0 = float4(
        fd * float(q0 & 0xF) - fm,
        fd * float(q0 >> 4) - fm,
        fd * float(q1 & 0xF) - fm,
        fd * float(q1 >> 4) - fm
    );
    out1 = float4(
        fd * float(q2 & 0xF) - fm,
        fd * float(q2 >> 4) - fm,
        fd * float(q3 & 0xF) - fm,
        fd * float(q3 >> 4) - fm
    );
}

// Optimized vector-matrix multiplication with Q4_K weights
// Uses simdgroup parallelism with multiple elements per thread
kernel void matmul_vec_q4k(
    device const uint8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles 4 rows (4 simdgroups)
    uint row = tgid * 4 + simd_group;
    if (row >= M) return;
    
    uint num_blocks = K / QK_K;
    uint block_size = 144;
    
    float partial_sum = 0.0f;
    
    // Process all blocks, each thread handles 8 elements per block
    for (uint block = 0; block < num_blocks; block++) {
        uint block_offset = (row * num_blocks + block) * block_size;
        device const uint8_t* block_ptr = weights + block_offset;
        
        half d = *reinterpret_cast<device const half*>(block_ptr);
        half dmin = *reinterpret_cast<device const half*>(block_ptr + 2);
        device const uint8_t* scales = block_ptr + 4;
        device const uint8_t* qs = block_ptr + 4 + K_SCALE_SIZE;
        
        uint input_base = block * QK_K;
        
        // Each thread processes 8 consecutive elements
        uint elem_start = simd_lane * 8;
        uint sub_block = elem_start / 32;
        
        uint8_t sc, m_val;
        get_scale_min_k4(scales, sub_block, sc, m_val);
        
        float fd = float(d) * float(sc);
        float fm = float(dmin) * float(m_val);
        
        // Inline dequantization for better performance
        uint qs_offset = elem_start / 2;
        uint8_t q0 = qs[qs_offset + 0];
        uint8_t q1 = qs[qs_offset + 1];
        uint8_t q2 = qs[qs_offset + 2];
        uint8_t q3 = qs[qs_offset + 3];
        
        float4 w0 = float4(
            fd * float(q0 & 0xF) - fm,
            fd * float(q0 >> 4) - fm,
            fd * float(q1 & 0xF) - fm,
            fd * float(q1 >> 4) - fm
        );
        float4 w1 = float4(
            fd * float(q2 & 0xF) - fm,
            fd * float(q2 >> 4) - fm,
            fd * float(q3 & 0xF) - fm,
            fd * float(q3 >> 4) - fm
        );
        
        // Use float4 loads for coalesced memory access
        float4 in0 = float4(
            input[input_base + elem_start + 0],
            input[input_base + elem_start + 1],
            input[input_base + elem_start + 2],
            input[input_base + elem_start + 3]
        );
        float4 in1 = float4(
            input[input_base + elem_start + 4],
            input[input_base + elem_start + 5],
            input[input_base + elem_start + 6],
            input[input_base + elem_start + 7]
        );
        
        partial_sum += dot(in0, w0) + dot(in1, w1);
    }
    
    // Single simd_sum at the end
    float total = simd_sum(partial_sum);
    
    if (simd_lane == 0) {
        output[row] = total;
    }
}

// Matrix-matrix multiplication with Q4_K weights using simdgroup parallelism
// output[M, T] = input[K, T] @ weights[K, M]
kernel void matmul_mat_q4k(
    device const uint8_t* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles 4 rows x 1 column
    uint row = tgid.x * 4 + simd_group;
    uint col = tgid.y;
    
    if (row >= M || col >= T) return;
    
    uint num_blocks = K / QK_K;
    uint block_size = 144;
    
    float partial_sum = 0.0f;
    
    // Each thread in simdgroup processes different K elements
    for (uint block = 0; block < num_blocks; block++) {
        uint block_offset = (row * num_blocks + block) * block_size;
        device const uint8_t* block_ptr = weights + block_offset;
        
        half d = *reinterpret_cast<device const half*>(block_ptr);
        half dmin = *reinterpret_cast<device const half*>(block_ptr + 2);
        device const uint8_t* scales = block_ptr + 4;
        device const uint8_t* qs = block_ptr + 4 + K_SCALE_SIZE;
        
        uint input_base = block * QK_K;
        uint elem_start = simd_lane * 8;
        uint sub_block = elem_start / 32;
        
        uint8_t sc, m_val;
        get_scale_min_k4(scales, sub_block, sc, m_val);
        
        float4 w0, w1;
        dequant_8(d, dmin, sc, m_val, qs, elem_start / 2, w0, w1);
        
        float4 in0 = float4(
            input[(input_base + elem_start + 0) * T + col],
            input[(input_base + elem_start + 1) * T + col],
            input[(input_base + elem_start + 2) * T + col],
            input[(input_base + elem_start + 3) * T + col]
        );
        float4 in1 = float4(
            input[(input_base + elem_start + 4) * T + col],
            input[(input_base + elem_start + 5) * T + col],
            input[(input_base + elem_start + 6) * T + col],
            input[(input_base + elem_start + 7) * T + col]
        );
        
        partial_sum += dot(in0, w0) + dot(in1, w1);
    }
    
    float total = simd_sum(partial_sum);
    
    if (simd_lane == 0) {
        output[row * T + col] = total;
    }
}
"#;

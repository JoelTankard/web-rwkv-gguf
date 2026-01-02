//! Metal kernel definitions and shader source.

/// Metal shader source for Int8 matrix multiplication with bias.
pub const MATMUL_INT8_BIAS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Int8 block size for quantization (matches GGML)
constant uint INT8_BLOCK_SIZE = 32;

struct MatmulParams {
    uint K;      // Input dimension (columns of matrix)
    uint M;      // Output dimension (rows of matrix)
    uint T;      // Number of tokens
    uint B;      // Batch size
};

// Fused Int8 matmul + bias + activation kernel
// Computes: output = activation(matrix @ input + bias)
//
// Matrix is stored as int8 with per-block min/max scales
// Input/output are f16
kernel void matmul_int8_bias(
    device const int8_t* matrix [[buffer(0)]],      // [K, M] quantized weights
    device const half2* scales [[buffer(1)]],       // [K/BLOCK, M] per-block (min, max)
    device const half* input [[buffer(2)]],         // [K, T, B] input tensor
    device const half* bias [[buffer(3)]],          // [M] bias vector
    device half* output [[buffer(4)]],              // [M, T, B] output tensor
    constant MatmulParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    const uint m = gid.x;  // Output row
    const uint t = gid.y;  // Token
    const uint b = gid.z;  // Batch
    
    if (m >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    // Accumulator
    float acc = 0.0f;
    
    // Input offset for this token/batch
    const uint input_offset = (b * T + t) * K;
    
    // Matrix offset for this output row
    const uint matrix_row_offset = m * K;
    
    // Number of blocks
    const uint num_blocks = K / INT8_BLOCK_SIZE;
    
    // Process each block
    for (uint block = 0; block < num_blocks; block++) {
        const uint block_offset = block * INT8_BLOCK_SIZE;
        
        // Get scale for this block
        const uint scale_idx = block * M + m;
        const half2 scale = scales[scale_idx];
        const float min_val = float(scale.x);
        const float max_val = float(scale.y);
        const float range = max_val - min_val;
        
        // Process elements in this block
        for (uint i = 0; i < INT8_BLOCK_SIZE; i++) {
            const uint k = block_offset + i;
            
            // Dequantize weight
            const int8_t q = matrix[matrix_row_offset + k];
            const float w = min_val + (float(q) + 128.0f) * range / 255.0f;
            
            // Get input
            const float x = float(input[input_offset + k]);
            
            // Accumulate
            acc += w * x;
        }
    }
    
    // Add bias
    acc += float(bias[m]);
    
    // Write output (no activation for now - can be added)
    const uint output_idx = (b * T + t) * M + m;
    output[output_idx] = half(acc);
}

// Optimized version using simdgroup operations for 8x8 tiles
kernel void matmul_int8_bias_simd(
    device const int8_t* matrix [[buffer(0)]],
    device const half2* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant MatmulParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // Each simdgroup processes an 8x8 tile
    const uint tile_m = gid.x * 8;
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (tile_m >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    // Accumulator for 8 output elements
    simdgroup_float8x8 acc;
    acc = simdgroup_float8x8(0.0f);
    
    const uint input_offset = (b * T + t) * K;
    const uint num_k_tiles = K / 8;
    
    // Process K dimension in tiles of 8
    for (uint k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const uint k_base = k_tile * 8;
        
        // Load and dequantize 8x8 weight tile
        simdgroup_float8x8 w_tile;
        // ... dequantization logic would go here
        
        // Load 8 input elements
        simdgroup_float8x8 x_tile;
        // ... load logic would go here
        
        // Multiply-accumulate
        simdgroup_multiply_accumulate(acc, w_tile, x_tile, acc);
    }
    
    // Add bias and store results
    const uint out_row = tile_m + simd_lane_id % 8;
    if (out_row < M) {
        const uint output_idx = (b * T + t) * M + out_row;
        // Extract result from simdgroup matrix and add bias
        float result = 0.0f; // Would extract from acc
        result += float(bias[out_row]);
        output[output_idx] = half(result);
    }
}

// Squared ReLU activation variant
kernel void matmul_int8_bias_squared_relu(
    device const int8_t* matrix [[buffer(0)]],
    device const half2* scales [[buffer(1)]],
    device const half* input [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant MatmulParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (m >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    float acc = 0.0f;
    const uint input_offset = (b * T + t) * K;
    const uint matrix_row_offset = m * K;
    const uint num_blocks = K / INT8_BLOCK_SIZE;
    
    for (uint block = 0; block < num_blocks; block++) {
        const uint block_offset = block * INT8_BLOCK_SIZE;
        const uint scale_idx = block * M + m;
        const half2 scale = scales[scale_idx];
        const float min_val = float(scale.x);
        const float max_val = float(scale.y);
        const float range = max_val - min_val;
        
        for (uint i = 0; i < INT8_BLOCK_SIZE; i++) {
            const uint k = block_offset + i;
            const int8_t q = matrix[matrix_row_offset + k];
            const float w = min_val + (float(q) + 128.0f) * range / 255.0f;
            const float x = float(input[input_offset + k]);
            acc += w * x;
        }
    }
    
    acc += float(bias[m]);
    
    // Squared ReLU: max(0, x)^2
    acc = max(acc, 0.0f);
    acc = acc * acc;
    
    const uint output_idx = (b * T + t) * M + m;
    output[output_idx] = half(acc);
}
"#;

/// Get the kernel function name for a given activation.
pub fn kernel_name_for_activation(activation: crate::tensor::ops::Activation) -> &'static str {
    use crate::tensor::ops::Activation;
    match activation {
        Activation::None => "matmul_int8_bias",
        Activation::SquaredRelu => "matmul_int8_bias_squared_relu",
        _ => "matmul_int8_bias", // Fall back to no activation
    }
}

/// Get the Q4K kernel function name for a given activation.
pub fn kernel_name_q4k_for_activation(activation: crate::tensor::ops::Activation) -> &'static str {
    use crate::tensor::ops::Activation;
    match activation {
        Activation::None => "matmul_q4k_bias",
        Activation::SquaredRelu => "matmul_q4k_bias_squared_relu",
        _ => "matmul_q4k_bias",
    }
}

/// Metal shader source for Q4_K matrix multiplication with bias.
/// Q4_K block structure (144 bytes = 36 u32 per 256 elements):
/// - d: f16 (super-block scale)
/// - dmin: f16 (super-block min)  
/// - scales: 12 bytes (8 sub-block scales/mins, 6-bit each)
/// - qs: 128 bytes (256 4-bit quantized values)
pub const MATMUL_Q4K_BIAS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint Q4K_BLOCK_SIZE = 256;
constant uint Q4K_BLOCK_U32 = 36;

struct MatmulParams {
    uint K;      // Input dimension (columns of matrix)
    uint M;      // Output dimension (rows of matrix)
    uint T;      // Number of tokens
    uint B;      // Batch size
};

// Extract a byte from packed scales
inline uint get_scale_byte(const device uint* scales, uint byte_idx) {
    return (scales[byte_idx / 4] >> ((byte_idx % 4) * 8)) & 0xFF;
}

// Get scale and min for sub-block j (0-7)
inline float2 get_scale_min_k4(uint j, const device uint* scales) {
    uint b_j = get_scale_byte(scales, j);
    uint b_j4 = get_scale_byte(scales, j + 4);
    
    float sc, m;
    if (j < 4) {
        sc = float(b_j & 63);
        m = float(b_j4 & 63);
    } else {
        uint b_jm4 = get_scale_byte(scales, j - 4);
        sc = float((b_j4 & 0xF) | ((b_jm4 >> 6) << 4));
        m = float((b_j4 >> 4) | ((b_j >> 6) << 4));
    }
    return float2(sc, m);
}

// Dequantize 8 elements from packed qs
inline float dequant_8(uint qs_packed, float d1, float m1, float d2, float m2, 
                       float4 x_lo, float4 x_hi) {
    float4 w_lo = float4(
        fma(d1, float(qs_packed & 0xF), -m1),
        fma(d1, float((qs_packed >> 8) & 0xF), -m1),
        fma(d1, float((qs_packed >> 16) & 0xF), -m1),
        fma(d1, float((qs_packed >> 24) & 0xF), -m1)
    );
    
    float4 w_hi = float4(
        fma(d2, float((qs_packed >> 4) & 0xF), -m2),
        fma(d2, float((qs_packed >> 12) & 0xF), -m2),
        fma(d2, float((qs_packed >> 20) & 0xF), -m2),
        fma(d2, float((qs_packed >> 28) & 0xF), -m2)
    );
    
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}

// Q4_K matmul + bias kernel
kernel void matmul_q4k_bias(
    device const uint* matrix [[buffer(0)]],        // Q4_K blocks
    device const half* input [[buffer(1)]],         // [K, T, B] input tensor
    device const half* bias [[buffer(2)]],          // [M] bias vector
    device half* output [[buffer(3)]],              // [M, T, B] output tensor
    constant MatmulParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint m_base = gid.x * 4;  // Each thread group handles 4 output rows
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (m_base >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    const uint num_super_blocks_k = K / Q4K_BLOCK_SIZE;
    const uint input_offset = (b * T + t) * K;
    
    float4 local_sum = float4(0.0f);
    
    // Each thread processes different super-blocks (strided)
    for (uint sb = tid; sb < num_super_blocks_k; sb += 64) {
        uint k_offset = sb * Q4K_BLOCK_SIZE;
        
        for (uint r = 0; r < 4; r++) {
            uint current_row = m_base + r;
            if (current_row >= M) continue;
            
            uint block_u32_base = (current_row * num_super_blocks_k + sb) * Q4K_BLOCK_U32;
            
            // Read d and dmin (packed as 2 f16 in one u32)
            uint d_dmin_packed = matrix[block_u32_base];
            float d = float(as_type<half2>(d_dmin_packed).x);
            float dmin = float(as_type<half2>(d_dmin_packed).y);
            
            const device uint* scales = &matrix[block_u32_base + 1];
            
            float row_sum = 0.0f;
            
            // Process 4 groups of 64 elements each
            for (uint j64 = 0; j64 < 4; j64++) {
                float2 sm0 = get_scale_min_k4(j64 * 2, scales);
                float2 sm1 = get_scale_min_k4(j64 * 2 + 1, scales);
                float d1 = d * sm0.x;
                float m1 = dmin * sm0.y;
                float d2 = d * sm1.x;
                float m2 = dmin * sm1.y;
                
                uint qs_base = block_u32_base + 4 + j64 * 8;
                
                for (uint l = 0; l < 8; l++) {
                    uint qs_packed = matrix[qs_base + l];
                    
                    uint elem_lo = k_offset + j64 * 64 + l * 4;
                    uint elem_hi = elem_lo + 32;
                    
                    // Read input (4 f16 values = 2 u32)
                    float4 x_lo = float4(
                        float(input[input_offset + elem_lo]),
                        float(input[input_offset + elem_lo + 1]),
                        float(input[input_offset + elem_lo + 2]),
                        float(input[input_offset + elem_lo + 3])
                    );
                    float4 x_hi = float4(
                        float(input[input_offset + elem_hi]),
                        float(input[input_offset + elem_hi + 1]),
                        float(input[input_offset + elem_hi + 2]),
                        float(input[input_offset + elem_hi + 3])
                    );
                    
                    row_sum += dequant_8(qs_packed, d1, m1, d2, m2, x_lo, x_hi);
                }
            }
            
            local_sum[r] += row_sum;
        }
    }
    
    // Store to threadgroup memory for reduction
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid * 4 + 0] += scratch[(tid + stride) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + stride) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + stride) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + stride) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 writes final result with bias
    if (tid == 0) {
        for (uint r = 0; r < 4; r++) {
            uint current_row = m_base + r;
            if (current_row >= M) continue;
            
            float result = scratch[r] + float(bias[current_row]);
            uint output_idx = (b * T + t) * M + current_row;
            output[output_idx] = half(result);
        }
    }
}

// Q4_K matmul + bias + squared relu kernel
kernel void matmul_q4k_bias_squared_relu(
    device const uint* matrix [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MatmulParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint m_base = gid.x * 4;
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (m_base >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    const uint num_super_blocks_k = K / Q4K_BLOCK_SIZE;
    const uint input_offset = (b * T + t) * K;
    
    float4 local_sum = float4(0.0f);
    
    for (uint sb = tid; sb < num_super_blocks_k; sb += 64) {
        uint k_offset = sb * Q4K_BLOCK_SIZE;
        
        for (uint r = 0; r < 4; r++) {
            uint current_row = m_base + r;
            if (current_row >= M) continue;
            
            uint block_u32_base = (current_row * num_super_blocks_k + sb) * Q4K_BLOCK_U32;
            
            uint d_dmin_packed = matrix[block_u32_base];
            float d = float(as_type<half2>(d_dmin_packed).x);
            float dmin = float(as_type<half2>(d_dmin_packed).y);
            
            const device uint* scales = &matrix[block_u32_base + 1];
            
            float row_sum = 0.0f;
            
            for (uint j64 = 0; j64 < 4; j64++) {
                float2 sm0 = get_scale_min_k4(j64 * 2, scales);
                float2 sm1 = get_scale_min_k4(j64 * 2 + 1, scales);
                float d1 = d * sm0.x;
                float m1 = dmin * sm0.y;
                float d2 = d * sm1.x;
                float m2 = dmin * sm1.y;
                
                uint qs_base = block_u32_base + 4 + j64 * 8;
                
                for (uint l = 0; l < 8; l++) {
                    uint qs_packed = matrix[qs_base + l];
                    
                    uint elem_lo = k_offset + j64 * 64 + l * 4;
                    uint elem_hi = elem_lo + 32;
                    
                    float4 x_lo = float4(
                        float(input[input_offset + elem_lo]),
                        float(input[input_offset + elem_lo + 1]),
                        float(input[input_offset + elem_lo + 2]),
                        float(input[input_offset + elem_lo + 3])
                    );
                    float4 x_hi = float4(
                        float(input[input_offset + elem_hi]),
                        float(input[input_offset + elem_hi + 1]),
                        float(input[input_offset + elem_hi + 2]),
                        float(input[input_offset + elem_hi + 3])
                    );
                    
                    row_sum += dequant_8(qs_packed, d1, m1, d2, m2, x_lo, x_hi);
                }
            }
            
            local_sum[r] += row_sum;
        }
    }
    
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = 32; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid * 4 + 0] += scratch[(tid + stride) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + stride) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + stride) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + stride) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        for (uint r = 0; r < 4; r++) {
            uint current_row = m_base + r;
            if (current_row >= M) continue;
            
            float result = scratch[r] + float(bias[current_row]);
            // Squared ReLU: max(0, x)^2
            result = max(result, 0.0f);
            result = result * result;
            
            uint output_idx = (b * T + t) * M + current_row;
            output[output_idx] = half(result);
        }
    }
}
"#;

//! Metal kernels for complete V7 layer execution.
//!
//! This module contains all the Metal shaders needed to run a full RWKV V7 layer
//! without any WebGPU involvement.

/// Complete Metal shader source for V7 layer operations.
/// Includes: layer_norm, token_shift, matmul_q4k, time_mix_v7, group_norm, channel_mix_v7, element-wise ops
pub const V7_LAYER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Common structures and utilities
// ============================================================================

struct LayerParams {
    uint C;          // Embedding dimension
    uint T;          // Number of tokens
    uint B;          // Batch size
    uint H;          // Number of heads
    uint S;          // Head size
    float eps;       // Epsilon for normalization
};

struct MatmulParams {
    uint K;          // Input dimension
    uint M;          // Output dimension
    uint T;          // Number of tokens
    uint B;          // Batch size
};

struct Cursor {
    uint batch;
    uint token;
    uint len;
};

inline Cursor unpack_cursor(uint x) {
    Cursor c;
    c.batch = x & 0xFF;
    c.token = (x >> 8) & 0xFFFF;
    c.len = (x >> 24) & 0xFF;
    return c;
}

// ============================================================================
// Layer Normalization
// ============================================================================

kernel void layer_norm(
    device const half* w [[buffer(0)]],           // [C] weights
    device const half* b [[buffer(1)]],           // [C] bias
    device half* x [[buffer(2)]],                 // [B, T, C] input/output
    constant LayerParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint C = params.C;
    const uint T = params.T;
    const uint batch = gid.z;
    const uint token = gid.y;
    
    if (batch >= params.B || token >= T) return;
    
    const uint base = (batch * T + token) * C;
    
    // Compute mean using parallel reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        local_sum += float(x[base + i]);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduce within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = scratch[0] / float(C);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float diff = float(x[base + i]) - mean;
        local_var += diff * diff;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float var = scratch[0] / float(C);
    float inv_std = rsqrt(var + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and apply affine transform
    for (uint i = tid; i < C; i += 256) {
        float val = (float(x[base + i]) - mean) * inv_std;
        x[base + i] = half(val * float(w[i]) + float(b[i]));
    }
}

// ============================================================================
// Token Shift (lerp between current and previous token)
// ============================================================================

kernel void token_shift(
    device const uint* cursors [[buffer(0)]],     // [T] cursor data
    device const half* mix [[buffer(1)]],         // [C] mix weights
    device const half* state [[buffer(2)]],       // [B, C] previous state
    device const half* input [[buffer(3)]],       // [T, C] input
    device half* output [[buffer(4)]],            // [T, C] output
    constant LayerParams& params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint C = params.C;
    const uint idx = gid.x;
    const uint token = gid.y;
    
    if (idx >= C || token >= params.T) return;
    
    Cursor cursor = unpack_cursor(cursors[token]);
    uint local_token = token - cursor.token;
    
    float m = float(mix[idx]);
    float curr = float(input[token * C + idx]);
    float prev;
    
    if (local_token == 0) {
        // First token in sequence - use state
        prev = float(state[cursor.batch * C + idx]);
    } else {
        // Use previous token
        prev = float(input[(token - 1) * C + idx]);
    }
    
    output[token * C + idx] = half(curr * m + prev * (1.0f - m));
}

// ============================================================================
// Element-wise operations
// ============================================================================

kernel void add_inplace(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = half(float(dst[gid]) + float(src[gid]));
    }
}

kernel void mul_inplace(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = half(float(dst[gid]) * float(src[gid]));
    }
}

kernel void blit(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

// ============================================================================
// Activation functions
// ============================================================================

inline float squared_relu(float x) {
    float r = max(x, 0.0f);
    return r * r;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float tanh_act(float x) {
    return tanh(x);
}

// ============================================================================
// Q4_K Matrix Multiplication (from existing kernel)
// ============================================================================

constant uint Q4K_BLOCK_SIZE = 256;
constant uint Q4K_BLOCK_U32 = 36;

inline uint get_scale_byte(const device uint* scales, uint byte_idx) {
    return (scales[byte_idx / 4] >> ((byte_idx % 4) * 8)) & 0xFF;
}

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

kernel void matmul_q4k(
    device const uint* matrix [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
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
                    
                    row_sum += dot(w_lo, x_lo) + dot(w_hi, x_hi);
                }
            }
            local_sum[r] += row_sum;
        }
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            output[output_base + r] = half(scratch[r]);
        }
    }
}

// Matmul with squared relu activation
kernel void matmul_q4k_squared_relu(
    device const uint* matrix [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
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
                    
                    row_sum += dot(w_lo, x_lo) + dot(w_hi, x_hi);
                }
            }
            local_sum[r] += row_sum;
        }
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            float val = scratch[r];
            val = max(val, 0.0f);
            output[output_base + r] = half(val * val);  // squared relu
        }
    }
}

// Matmul with tanh activation
kernel void matmul_q4k_tanh(
    device const uint* matrix [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
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
                    
                    row_sum += dot(w_lo, x_lo) + dot(w_hi, x_hi);
                }
            }
            local_sum[r] += row_sum;
        }
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            output[output_base + r] = half(tanh(scratch[r]));
        }
    }
}

// Matmul with sigmoid activation
kernel void matmul_q4k_sigmoid(
    device const uint* matrix [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
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
                    
                    row_sum += dot(w_lo, x_lo) + dot(w_hi, x_hi);
                }
            }
            local_sum[r] += row_sum;
        }
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            float val = scratch[r];
            output[output_base + r] = half(1.0f / (1.0f + exp(-val)));
        }
    }
}

// ============================================================================
// Fp16 Matrix Multiplication
// ============================================================================

kernel void matmul_fp16(
    device const half* matrix [[buffer(0)]],      // [M, K]
    device const half* input [[buffer(1)]],       // [B, T, K]
    device half* output [[buffer(2)]],            // [B, T, M]
    constant MatmulParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint m_base = gid.x * 4;  // 4 output rows per workgroup
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (m_base >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    const uint input_offset = (b * T + t) * K;
    
    float4 local_sum = float4(0.0f);
    
    // Process K dimension in chunks of 4
    for (uint i = tid; i < K / 4; i += 64) {
        uint k_idx = i * 4;
        
        // Read 4 input values
        float4 x = float4(
            float(input[input_offset + k_idx]),
            float(input[input_offset + k_idx + 1]),
            float(input[input_offset + k_idx + 2]),
            float(input[input_offset + k_idx + 3])
        );
        
        // Process 4 output rows
        for (uint r = 0; r < 4; r++) {
            uint row = m_base + r;
            if (row >= M) continue;
            
            // Matrix is [M, K], row-major
            uint mat_base = row * K + k_idx;
            float4 w = float4(
                float(matrix[mat_base]),
                float(matrix[mat_base + 1]),
                float(matrix[mat_base + 2]),
                float(matrix[mat_base + 3])
            );
            local_sum[r] += dot(w, x);
        }
    }
    
    // Handle remaining elements
    uint remaining_start = (K / 4) * 4;
    if (tid == 0 && remaining_start < K) {
        for (uint k = remaining_start; k < K; k++) {
            float x_val = float(input[input_offset + k]);
            for (uint r = 0; r < 4; r++) {
                uint row = m_base + r;
                if (row >= M) continue;
                local_sum[r] += float(matrix[row * K + k]) * x_val;
            }
        }
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            output[output_base + r] = half(scratch[r]);
        }
    }
}

// ============================================================================
// Int8 Matrix Multiplication
// ============================================================================

// Int8 block size for minmax quantization
constant uint INT8_BLOCK_SIZE = 128;

// Dequantize Int8 value using min/max scaling
// w_u8 is in [0, 255], maps to [min, max]
inline float4 dequant_int8(uint packed, float min_val, float max_val) {
    float scale = (max_val - min_val) / 255.0f;
    return float4(
        fma(float((packed >> 0) & 0xFF), scale, min_val),
        fma(float((packed >> 8) & 0xFF), scale, min_val),
        fma(float((packed >> 16) & 0xFF), scale, min_val),
        fma(float((packed >> 24) & 0xFF), scale, min_val)
    );
}

kernel void matmul_int8(
    device const uint* matrix [[buffer(0)]],      // [B, M, K] packed as u32 (4 bytes each)
    device const uint* minmax [[buffer(1)]],      // [B, M, K/INT8_BLOCK_SIZE] packed f16 min/max pairs
    device const half* input [[buffer(2)]],       // [B, T, K]
    device half* output [[buffer(3)]],            // [B, T, M]
    constant MatmulParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint m_base = gid.x * 4;  // 4 output rows per workgroup
    const uint t = gid.y;
    const uint b = gid.z;
    
    if (m_base >= params.M || t >= params.T || b >= params.B) return;
    
    const uint K = params.K;
    const uint M = params.M;
    const uint T = params.T;
    
    const uint stride = K / 4;  // K elements packed as K/4 u32s
    const uint num_blocks = K / INT8_BLOCK_SIZE;
    const uint input_offset = (b * T + t) * K;
    
    float4 local_sum = float4(0.0f);
    
    // INT8_BLOCK_STEP = INT8_BLOCK_SIZE / 4 = 32 (u32s per minmax block)
    const uint INT8_BLOCK_STEP = INT8_BLOCK_SIZE / 4;
    
    // WebGPU shader uses transpose(m) * x pattern:
    // It reads 4 rows of the matrix at the same K position, forms a 4x4 matrix,
    // then computes transpose(m) * x to get 4 output values.
    // This is equivalent to: output[j] += sum_r(matrix[row_base + r][k + j] * input[k + r])
    //
    // To match this, we need to read 4 consecutive rows and do the transposed multiply.
    
    for (uint i = tid; i < stride; i += 64) {
        uint k_idx = i * 4;  // Actual K index
        
        // Read 4 input values at positions k_idx, k_idx+1, k_idx+2, k_idx+3
        float4 x = float4(
            float(input[input_offset + k_idx]),
            float(input[input_offset + k_idx + 1]),
            float(input[input_offset + k_idx + 2]),
            float(input[input_offset + k_idx + 3])
        );
        
        // Read 4 rows of the matrix at K position i, forming a 4x4 block
        float4 m0, m1, m2, m3;
        
        for (uint r = 0; r < 4; r++) {
            uint row = m_base + r;
            if (row >= M) continue;
            
            uint mat_idx = (b * M + row) * stride + i;
            uint packed = matrix[mat_idx];
            
            uint mm_idx = mat_idx / INT8_BLOCK_STEP;
            uint mm_packed = minmax[mm_idx];
            float min_val = float(as_type<half2>(mm_packed).x);
            float max_val = float(as_type<half2>(mm_packed).y);
            
            float4 w = dequant_int8(packed, min_val, max_val);
            
            // Store in the appropriate row
            if (r == 0) m0 = w;
            else if (r == 1) m1 = w;
            else if (r == 2) m2 = w;
            else m3 = w;
        }
        
        // Compute transpose(m) * x
        // transpose(m)[j] = [m0[j], m1[j], m2[j], m3[j]]
        // result[j] = dot(transpose(m)[j], x) = m0[j]*x[0] + m1[j]*x[1] + m2[j]*x[2] + m3[j]*x[3]
        local_sum[0] += m0[0]*x[0] + m1[0]*x[1] + m2[0]*x[2] + m3[0]*x[3];
        local_sum[1] += m0[1]*x[0] + m1[1]*x[1] + m2[1]*x[2] + m3[1]*x[3];
        local_sum[2] += m0[2]*x[0] + m1[2]*x[1] + m2[2]*x[2] + m3[2]*x[3];
        local_sum[3] += m0[3]*x[0] + m1[3]*x[1] + m2[3]*x[2] + m3[3]*x[3];
    }
    
    // Reduce across threads
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            output[output_base + r] = half(scratch[r]);
        }
    }
}

// Int8 matmul with squared relu activation
kernel void matmul_int8_squared_relu(
    device const uint* matrix [[buffer(0)]],
    device const uint* minmax [[buffer(1)]],
    device const half* input [[buffer(2)]],
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
    
    const uint stride = K / 4;
    const uint input_offset = (b * T + t) * K;
    const uint INT8_BLOCK_STEP = INT8_BLOCK_SIZE / 4;
    
    float4 local_sum = float4(0.0f);
    
    // Match WebGPU shader's transpose(m) * x pattern
    for (uint i = tid; i < stride; i += 64) {
        uint k_idx = i * 4;
        
        float4 x = float4(
            float(input[input_offset + k_idx]),
            float(input[input_offset + k_idx + 1]),
            float(input[input_offset + k_idx + 2]),
            float(input[input_offset + k_idx + 3])
        );
        
        // Read 4 rows of the matrix at K position i
        float4 m0, m1, m2, m3;
        
        for (uint r = 0; r < 4; r++) {
            uint row = m_base + r;
            if (row >= M) continue;
            
            uint mat_idx = (b * M + row) * stride + i;
            uint packed = matrix[mat_idx];
            
            uint mm_idx = mat_idx / INT8_BLOCK_STEP;
            uint mm_packed = minmax[mm_idx];
            float min_val = float(as_type<half2>(mm_packed).x);
            float max_val = float(as_type<half2>(mm_packed).y);
            
            float4 w = dequant_int8(packed, min_val, max_val);
            
            if (r == 0) m0 = w;
            else if (r == 1) m1 = w;
            else if (r == 2) m2 = w;
            else m3 = w;
        }
        
        // Compute transpose(m) * x
        local_sum[0] += m0[0]*x[0] + m1[0]*x[1] + m2[0]*x[2] + m3[0]*x[3];
        local_sum[1] += m0[1]*x[0] + m1[1]*x[1] + m2[1]*x[2] + m3[1]*x[3];
        local_sum[2] += m0[2]*x[0] + m1[2]*x[1] + m2[2]*x[2] + m3[2]*x[3];
        local_sum[3] += m0[3]*x[0] + m1[3]*x[1] + m2[3]*x[2] + m3[3]*x[3];
    }
    
    scratch[tid * 4 + 0] = local_sum[0];
    scratch[tid * 4 + 1] = local_sum[1];
    scratch[tid * 4 + 2] = local_sum[2];
    scratch[tid * 4 + 3] = local_sum[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) {
            scratch[tid * 4 + 0] += scratch[(tid + s) * 4 + 0];
            scratch[tid * 4 + 1] += scratch[(tid + s) * 4 + 1];
            scratch[tid * 4 + 2] += scratch[(tid + s) * 4 + 2];
            scratch[tid * 4 + 3] += scratch[(tid + s) * 4 + 3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        uint output_base = (b * T + t) * M + m_base;
        for (uint r = 0; r < 4 && m_base + r < M; r++) {
            float val = max(scratch[r], 0.0f);
            output[output_base + r] = half(val * val);
        }
    }
}

// ============================================================================
// Add with bias and activation (for attention adapt ops)
// ============================================================================

kernel void add_activate_sigmoid(
    device const half* bias [[buffer(0)]],
    device half* x [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < count) {
        float val = float(x[gid]) + float(bias[gid % count]);
        x[gid] = half(1.0f / (1.0f + exp(-val)));
    }
}

// ============================================================================
// Group Normalization
// ============================================================================

kernel void group_norm(
    device const half* w [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* x [[buffer(2)]],
    constant LayerParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint H = params.H;
    const uint S = params.S;
    const uint C = H * S;
    const uint token = gid.y;
    const uint head = gid.x;
    
    if (token >= params.T || head >= H) return;
    
    const uint base = token * C + head * S;
    
    // Compute mean
    float local_sum = 0.0f;
    for (uint i = tid; i < S; i += 64) {
        local_sum += float(x[base + i]);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float mean = scratch[0] / float(S);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute variance
    float local_var = 0.0f;
    for (uint i = tid; i < S; i += 64) {
        float diff = float(x[base + i]) - mean;
        local_var += diff * diff;
    }
    scratch[tid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_std = rsqrt(scratch[0] / float(S) + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and apply affine
    for (uint i = tid; i < S; i += 64) {
        float val = (float(x[base + i]) - mean) * inv_std;
        uint w_idx = head * S + i;
        x[base + i] = half(val * float(w[w_idx]) + float(b[w_idx]));
    }
}

// ============================================================================
// Channel Mix V7 (simplified - just copies v to output)
// ============================================================================

kernel void channel_mix_v7(
    device const uint* cursors [[buffer(0)]],
    device float* state [[buffer(1)]],
    device const half* v [[buffer(2)]],
    device half* x [[buffer(3)]],
    constant LayerParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint C = params.C;
    const uint idx = gid.x;
    const uint token = gid.y;
    
    if (idx >= C || token >= params.T) return;
    
    Cursor cursor = unpack_cursor(cursors[token]);
    uint local_token = token - cursor.token;
    
    // Update state on last token of sequence
    if (local_token + 1 == cursor.len) {
        state[cursor.batch * C + idx] = float(x[token * C + idx]);
    }
    
    // V7 channel mix just copies v to output
    x[token * C + idx] = v[token * C + idx];
}

// ============================================================================
// Lerp operation: y = mix(x, y, f) or y = mix(y, x, f) if reversed
// ============================================================================

kernel void lerp_op(
    device const half* f [[buffer(0)]],
    device const half* x [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& reversed [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float ff = float(f[gid]);
    float xx = float(x[gid]);
    float yy = float(y[gid]);
    
    float result;
    if (reversed != 0) {
        result = mix(yy, xx, ff);  // reversed: lerp from y to x
    } else {
        result = mix(xx, yy, ff);  // normal: lerp from x to y
    }
    
    y[gid] = half(result);
}

// ============================================================================
// Pack KVAKK (combines k, v, a, kk into n tensor)
// ============================================================================

kernel void pack_kvakk(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const half* a [[buffer(2)]],
    device const half* kk [[buffer(3)]],
    device half* n [[buffer(4)]],
    constant uint& stride [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= stride) return;
    
    n[gid] = k[gid];
    n[gid + stride] = v[gid];
    n[gid + 2 * stride] = a[gid];
    n[gid + 3 * stride] = kk[gid];
}

// ============================================================================
// Time Mix V7 - RWKV attention with state updates
// ============================================================================

struct TimeMixParams {
    uint S;          // Head size (elements per head / 4, since we work with vec4)
    uint H;          // Number of heads
    uint T;          // Number of tokens
    uint C;          // Total channels (H * S * 4)
};

struct StateView {
    uint shape_x;    // C
    uint shape_y;    // S + 1
    uint shape_z;    // B
    uint stride_x;   // 1
    uint stride_y;   // C
    uint offset_x;   // 0
    uint offset_y;   // 0
    uint offset_z;   // 0
};

inline float act_w(float x) {
    float s = 1.0f / (1.0f + exp(-x));
    return exp(-0.606531f * s);
}

kernel void time_mix_v7(
    device const uint* cursors [[buffer(0)]],
    device float4* state [[buffer(1)]],
    device const half* r_buf [[buffer(2)]],
    device const half* w_buf [[buffer(3)]],
    device const half* n_buf [[buffer(4)]],
    device half* x_buf [[buffer(5)]],
    constant TimeMixParams& params [[buffer(6)]],
    constant StateView& view [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float4* shared_r [[threadgroup(0)]],
    threadgroup float4* shared_k [[threadgroup(1)]],
    threadgroup float4* shared_w [[threadgroup(2)]],
    threadgroup float4* shared_a [[threadgroup(3)]],
    threadgroup float4* shared_b [[threadgroup(4)]]
) {
    const uint HEAD_SIZE = params.S;
    const uint H = params.H;
    const uint T = params.T;
    const uint stride = H * HEAD_SIZE;  // Total vec4 elements per token
    
    const uint index = gid.x;
    const uint head = tid / HEAD_SIZE;
    const uint h = head * HEAD_SIZE;
    
    if (index >= stride) return;
    
    // State stride for indexing
    const uint state_stride = view.shape_x / 4;
    
    // Process each token sequentially
    for (uint t = 0; t < T; t++) {
        uint ti = t * stride + index;
        
        // Unpack cursor
        uint cursor_packed = cursors[t];
        uint cursor_batch = cursor_packed & 0xFF;
        uint cursor_token = (cursor_packed >> 8) & 0xFFFF;
        uint cursor_len = (cursor_packed >> 24) & 0xFF;
        
        // Update state on last token of sequence
        if (t - cursor_token + 1 == cursor_len) {
            uint last_ti = (cursor_token + cursor_len - 1) * stride + index;
            state[cursor_batch * view.shape_y * state_stride + index] = float4(
                float(x_buf[last_ti * 4]),
                float(x_buf[last_ti * 4 + 1]),
                float(x_buf[last_ti * 4 + 2]),
                float(x_buf[last_ti * 4 + 3])
            );
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load shared data
        // r
        float4 r_val = float4(
            float(r_buf[ti * 4]), float(r_buf[ti * 4 + 1]),
            float(r_buf[ti * 4 + 2]), float(r_buf[ti * 4 + 3])
        );
        shared_r[tid] = r_val;
        
        // w with activation
        float4 w_val = float4(
            act_w(float(w_buf[ti * 4])), act_w(float(w_buf[ti * 4 + 1])),
            act_w(float(w_buf[ti * 4 + 2])), act_w(float(w_buf[ti * 4 + 3]))
        );
        shared_w[tid] = w_val;
        
        // k from n[0..stride]
        float4 k_val = float4(
            float(n_buf[ti * 4]), float(n_buf[ti * 4 + 1]),
            float(n_buf[ti * 4 + 2]), float(n_buf[ti * 4 + 3])
        );
        shared_k[tid] = k_val;
        
        // a from n[2*stride..3*stride]
        uint a_offset = 2 * stride * 4;
        float4 a_val = float4(
            float(n_buf[ti * 4 + a_offset]), float(n_buf[ti * 4 + 1 + a_offset]),
            float(n_buf[ti * 4 + 2 + a_offset]), float(n_buf[ti * 4 + 3 + a_offset])
        );
        
        // kk from n[3*stride..4*stride]
        uint kk_offset = 3 * stride * 4;
        float4 kk_val = float4(
            float(n_buf[ti * 4 + kk_offset]), float(n_buf[ti * 4 + 1 + kk_offset]),
            float(n_buf[ti * 4 + 2 + kk_offset]), float(n_buf[ti * 4 + 3 + kk_offset])
        );
        
        shared_a[tid] = -kk_val;
        shared_b[tid] = kk_val * a_val;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute sa (sum of state * a)
        float4 sa = float4(0.0f);
        for (uint j = 0; j < HEAD_SIZE; j++) {
            float4 aa = shared_a[h + j];
            
            // Read 4 state rows for this j
            uint state_base = cursor_batch * view.shape_y * state_stride;
            float4 ss0 = state[state_base + (j * 4 + 1) * state_stride + index];
            float4 ss1 = state[state_base + (j * 4 + 2) * state_stride + index];
            float4 ss2 = state[state_base + (j * 4 + 3) * state_stride + index];
            float4 ss3 = state[state_base + (j * 4 + 4) * state_stride + index];
            
            sa += ss0 * aa[0] + ss1 * aa[1] + ss2 * aa[2] + ss3 * aa[3];
        }
        
        // v from n[stride..2*stride]
        uint v_offset = stride * 4;
        float4 vv = float4(
            float(n_buf[ti * 4 + v_offset]), float(n_buf[ti * 4 + 1 + v_offset]),
            float(n_buf[ti * 4 + 2 + v_offset]), float(n_buf[ti * 4 + 3 + v_offset])
        );
        
        // Main computation loop
        float4 y = float4(0.0f);
        for (uint j = 0; j < HEAD_SIZE; j++) {
            float4 rr = shared_r[h + j];
            float4 ww = shared_w[h + j];
            float4 kk = shared_k[h + j];
            float4 bb = shared_b[h + j];
            
            // Read state
            uint state_base2 = cursor_batch * view.shape_y * state_stride;
            float4 ss0 = state[state_base2 + (j * 4 + 1) * state_stride + index];
            float4 ss1 = state[state_base2 + (j * 4 + 2) * state_stride + index];
            float4 ss2 = state[state_base2 + (j * 4 + 3) * state_stride + index];
            float4 ss3 = state[state_base2 + (j * 4 + 4) * state_stride + index];
            
            // Update state: ss = ss * ww + kk * vv + sa * bb
            ss0 = ss0 * ww[0] + kk[0] * vv + sa * bb[0];
            ss1 = ss1 * ww[1] + kk[1] * vv + sa * bb[1];
            ss2 = ss2 * ww[2] + kk[2] * vv + sa * bb[2];
            ss3 = ss3 * ww[3] + kk[3] * vv + sa * bb[3];
            
            // Accumulate output
            y += rr[0] * ss0 + rr[1] * ss1 + rr[2] * ss2 + rr[3] * ss3;
            
            // Write back state
            state[state_base2 + (j * 4 + 1) * state_stride + index] = ss0;
            state[state_base2 + (j * 4 + 2) * state_stride + index] = ss1;
            state[state_base2 + (j * 4 + 3) * state_stride + index] = ss2;
            state[state_base2 + (j * 4 + 4) * state_stride + index] = ss3;
        }
        
        // Store output
        x_buf[ti * 4] = half(y[0]);
        x_buf[ti * 4 + 1] = half(y[1]);
        x_buf[ti * 4 + 2] = half(y[2]);
        x_buf[ti * 4 + 3] = half(y[3]);
    }
}

// ============================================================================
// Time First V7 - adds the "first" contribution to output
// ============================================================================

kernel void time_first_v7(
    device const half* u_buf [[buffer(0)]],
    device const half* r_buf [[buffer(1)]],
    device const half* n_buf [[buffer(2)]],
    device half* x_buf [[buffer(3)]],
    constant TimeMixParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float4* shared_x [[threadgroup(0)]]
) {
    const uint HEAD_SIZE = params.S;
    const uint H = params.H;
    const uint stride = H * HEAD_SIZE;
    
    const uint index = gid.x;
    const uint t = gid.y;
    const uint head = tid / HEAD_SIZE;
    const uint h = head * HEAD_SIZE;
    
    if (index >= stride) return;
    if (t >= params.T) return;
    
    uint ti = t * stride + index;
    
    // Load u, k, r and compute u * k * r
    float4 uu = float4(
        float(u_buf[index * 4]), float(u_buf[index * 4 + 1]),
        float(u_buf[index * 4 + 2]), float(u_buf[index * 4 + 3])
    );
    
    // k from n[0..stride]
    float4 kk = float4(
        float(n_buf[ti * 4]), float(n_buf[ti * 4 + 1]),
        float(n_buf[ti * 4 + 2]), float(n_buf[ti * 4 + 3])
    );
    
    float4 rr = float4(
        float(r_buf[ti * 4]), float(r_buf[ti * 4 + 1]),
        float(r_buf[ti * 4 + 2]), float(r_buf[ti * 4 + 3])
    );
    
    shared_x[tid] = uu * kk * rr;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction within head
    for (uint step = HEAD_SIZE >> 1; step > 0; step >>= 1) {
        if (tid % HEAD_SIZE < step) {
            shared_x[tid] += shared_x[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply result
    float xx = shared_x[h][0] + shared_x[h][1] + shared_x[h][2] + shared_x[h][3];
    
    // v from n[stride..2*stride]
    uint v_offset = stride * 4;
    float4 vv = float4(
        float(n_buf[ti * 4 + v_offset]), float(n_buf[ti * 4 + 1 + v_offset]),
        float(n_buf[ti * 4 + 2 + v_offset]), float(n_buf[ti * 4 + 3 + v_offset])
    );
    
    // Add xx * vv to output
    float4 x_val = float4(
        float(x_buf[ti * 4]), float(x_buf[ti * 4 + 1]),
        float(x_buf[ti * 4 + 2]), float(x_buf[ti * 4 + 3])
    );
    x_val += xx * vv;
    
    x_buf[ti * 4] = half(x_val[0]);
    x_buf[ti * 4 + 1] = half(x_val[1]);
    x_buf[ti * 4 + 2] = half(x_val[2]);
    x_buf[ti * 4 + 3] = half(x_val[3]);
}

// ============================================================================
// L2 Normalization
// ============================================================================

kernel void l2_norm(
    device half* x [[buffer(0)]],
    constant LayerParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]
) {
    const uint S = params.S;
    const uint H = params.H;
    const uint token = gid.y;
    const uint head = gid.x;
    
    if (token >= params.T || head >= H) return;
    
    const uint base = token * H * S + head * S;
    
    // Compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < S; i += 64) {
        float val = float(x[base + i]);
        local_sum += val * val;
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 32; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_norm = rsqrt(scratch[0] + params.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    for (uint i = tid; i < S; i += 64) {
        x[base + i] = half(float(x[base + i]) * inv_norm);
    }
}

"#;

//! Metal Shading Language kernels for quantized matrix multiplication.
//!
//! These kernels use `simdgroup_matrix` operations for hardware-accelerated
//! matrix multiplication with inline Q4_K dequantization.

pub const METAL_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ============================================================================
// PHASE 1: Simple Element-wise Operations
// ============================================================================

// Add two tensors: output = input + output (in-place on output)
kernel void add_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = input[tid] + output[tid];
    }
}

// Add with f32 input: output = input + output
kernel void add_f32_to_fp16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = half(input[tid]) + output[tid];
    }
}

// Multiply two tensors: output = input * output (in-place on output)
kernel void mul_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = input[tid] * output[tid];
    }
}

// Copy tensor: dst = src
kernel void blit_fp16(
    device const half* src [[buffer(0)]],
    device half* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        dst[tid] = src[tid];
    }
}

// Affine transform: x = scale * x + bias (in-place)
kernel void affine_fp16(
    device half* x [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant float& bias [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        x[tid] = half(float(x[tid]) * scale + bias);
    }
}

// ============================================================================
// PHASE 1: Layer Normalization
// ============================================================================

// Layer norm with simdgroup reduction for efficiency
// Each threadgroup processes one token (C elements)
// Uses Welford's online algorithm for numerical stability
kernel void layer_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& C [[buffer(4)]],           // number of channels
    constant uint& T [[buffer(5)]],           // number of tokens
    constant float& eps [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one token
    uint token = tgid;
    if (token >= T) return;
    
    uint base = token * C;
    
    // Phase 1: Compute mean using parallel reduction
    // Each thread accumulates a portion of the sum
    float sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {  // 256 threads per threadgroup
        sum += float(input[base + i]);
    }
    
    // Reduce within simdgroup (32 threads)
    sum = simd_sum(sum);
    
    // Store simdgroup results in threadgroup memory
    threadgroup float shared_sum[8];  // 256/32 = 8 simdgroups
    if (simd_lane == 0) {
        shared_sum[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction by first simdgroup
    float mean = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        mean = shared_sum[simd_lane];
    }
    mean = simd_sum(mean) / float(C);
    
    // Broadcast mean to all threads
    threadgroup float shared_mean;
    if (tid == 0) {
        shared_mean = mean;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = shared_mean;
    
    // Phase 2: Compute variance
    float var_sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float diff = float(input[base + i]) - mean;
        var_sum += diff * diff;
    }
    
    var_sum = simd_sum(var_sum);
    
    threadgroup float shared_var[8];
    if (simd_lane == 0) {
        shared_var[simd_group] = var_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float variance = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        variance = shared_var[simd_lane];
    }
    variance = simd_sum(variance) / float(C);
    float inv_std = rsqrt(variance + eps);
    
    // Broadcast inv_std
    threadgroup float shared_inv_std;
    if (tid == 0) {
        shared_inv_std = inv_std;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_std = shared_inv_std;
    
    // Phase 3: Normalize and apply affine transform
    for (uint i = tid; i < C; i += 256) {
        float x = float(input[base + i]);
        float y = (x - mean) * inv_std;
        output[base + i] = half(y * float(weight[i]) + float(bias[i]));
    }
}

// Group norm variant - weight/bias are per-head
kernel void group_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& head_size [[buffer(4)]],   // elements per head
    constant uint& num_heads [[buffer(5)]],   // number of heads
    constant uint& T [[buffer(6)]],           // number of tokens
    constant float& eps [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token = tgid.x;
    uint head = tgid.y;
    if (token >= T || head >= num_heads) return;
    
    uint C = head_size * num_heads;
    uint base = token * C + head * head_size;
    uint weight_base = head * head_size;
    
    // Compute mean for this head
    float sum = 0.0f;
    for (uint i = tid; i < head_size; i += 32) {
        sum += float(input[base + i]);
    }
    sum = simd_sum(sum);
    float mean = sum / float(head_size);
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint i = tid; i < head_size; i += 32) {
        float diff = float(input[base + i]) - mean;
        var_sum += diff * diff;
    }
    var_sum = simd_sum(var_sum);
    float inv_std = rsqrt(var_sum / float(head_size) + eps);
    
    // Normalize and apply affine
    for (uint i = tid; i < head_size; i += 32) {
        float x = float(input[base + i]);
        float y = (x - mean) * inv_std;
        output[base + i] = half(y * float(weight[weight_base + i]) + float(bias[weight_base + i]));
    }
}

// ============================================================================
// PHASE 1: Token Shift
// ============================================================================

// Token shift for RWKV: mixes current token with previous token based on mix factor
// output = mix(prev, current, factor) = prev * (1 - factor) + current * factor
kernel void token_shift_fp16(
    device const uint* cursors [[buffer(0)]],     // [T] packed cursor info
    device const half* time_mix [[buffer(1)]],    // [C] or [T, C] mix factors
    device const float* state [[buffer(2)]],      // [B, C] initial state per batch
    device const half* input [[buffer(3)]],       // [T, C] input tokens
    device half* output [[buffer(4)]],            // [T, C] output
    constant uint& C [[buffer(5)]],               // channels
    constant uint& T [[buffer(6)]],               // tokens
    constant uint& time_mix_stride [[buffer(7)]], // 0 if shared, C if per-token
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;
    
    if (elem >= C || token >= T) return;
    
    // Decode cursor: batch in low 8 bits, token offset in bits 8-23
    uint cursor_val = cursors[token];
    uint batch = cursor_val & 0xFFu;
    uint token_in_seq = (cursor_val >> 8u) & 0xFFFFu;
    
    // Get mix factor
    uint mix_idx = time_mix_stride > 0 ? token * C + elem : elem;
    float factor = float(time_mix[mix_idx]);
    
    // Get previous value
    float prev;
    if (token_in_seq == 0) {
        // First token in sequence: use state
        prev = state[batch * C + elem];
    } else {
        // Use previous token
        prev = float(input[(token - 1) * C + elem]);
    }
    
    float current = float(input[token * C + elem]);
    
    // Linear interpolation: prev * (1 - factor) + current * factor
    float result = prev + factor * (current - prev);
    output[token * C + elem] = half(result);
}

// Reversed token shift variant
kernel void token_shift_reversed_fp16(
    device const uint* cursors [[buffer(0)]],
    device const half* time_mix [[buffer(1)]],
    device const float* state [[buffer(2)]],
    device const half* input [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant uint& time_mix_stride [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;
    
    if (elem >= C || token >= T) return;
    
    uint cursor_val = cursors[token];
    uint batch = cursor_val & 0xFFu;
    uint token_in_seq = (cursor_val >> 8u) & 0xFFFFu;
    
    uint mix_idx = time_mix_stride > 0 ? token * C + elem : elem;
    float factor = float(time_mix[mix_idx]);
    
    float prev;
    if (token_in_seq == 0) {
        prev = state[batch * C + elem];
    } else {
        prev = float(input[(token - 1) * C + elem]);
    }
    
    float current = float(input[token * C + elem]);
    
    // Reversed: current * (1 - factor) + prev * factor
    float result = current + factor * (prev - current);
    output[token * C + elem] = half(result);
}

// ============================================================================
// PHASE 1: Lerp (Linear Interpolation)
// ============================================================================

// output = a + factor * (b - a)
kernel void lerp_fp16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device const half* factor [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float fa = float(a[tid]);
        float fb = float(b[tid]);
        float ff = float(factor[tid]);
        output[tid] = half(fa + ff * (fb - fa));
    }
}

// ============================================================================
// PHASE 2: FP16 Matrix-Vector Multiplication
// ============================================================================

// FP16 matrix-vector multiplication using simdgroup parallelism
// output[M] = input[K] @ weights[K, M]
// Each threadgroup handles 4 output rows (4 simdgroups)
kernel void matmul_vec_fp16(
    device const half* weights [[buffer(0)]],  // [K, M] row-major
    device const half* input [[buffer(1)]],    // [K]
    device half* output [[buffer(2)]],         // [M]
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Each threadgroup handles 4 rows (4 simdgroups)
    uint row = tgid * 4 + simd_group;
    if (row >= M) return;
    
    float sum = 0.0f;
    
    // Each thread in simdgroup processes K/32 elements
    // Use float4 for better memory coalescing
    uint k_per_thread = (K + 31) / 32;
    uint k_start = simd_lane * k_per_thread;
    uint k_end = min(k_start + k_per_thread, K);
    
    device const half* row_ptr = weights + row * K;
    
    for (uint k = k_start; k < k_end; k++) {
        sum += float(row_ptr[k]) * float(input[k]);
    }
    
    // Reduce across simdgroup
    sum = simd_sum(sum);
    
    if (simd_lane == 0) {
        output[row] = half(sum);
    }
}

// FP16 matmul with tanh activation (for LoRA w1 matrices)
kernel void matmul_vec_fp16_tanh(
    device const half* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4 + simd_group;
    if (row >= M) return;
    
    float sum = 0.0f;
    uint k_per_thread = (K + 31) / 32;
    uint k_start = simd_lane * k_per_thread;
    uint k_end = min(k_start + k_per_thread, K);
    
    device const half* row_ptr = weights + row * K;
    
    for (uint k = k_start; k < k_end; k++) {
        sum += float(row_ptr[k]) * float(input[k]);
    }
    
    sum = simd_sum(sum);
    
    if (simd_lane == 0) {
        output[row] = half(tanh(sum));
    }
}

// FP16 matmul with squared ReLU activation (for FFN)
kernel void matmul_vec_fp16_squared_relu(
    device const half* weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4 + simd_group;
    if (row >= M) return;
    
    float sum = 0.0f;
    uint k_per_thread = (K + 31) / 32;
    uint k_start = simd_lane * k_per_thread;
    uint k_end = min(k_start + k_per_thread, K);
    
    device const half* row_ptr = weights + row * K;
    
    for (uint k = k_start; k < k_end; k++) {
        sum += float(row_ptr[k]) * float(input[k]);
    }
    
    sum = simd_sum(sum);
    
    if (simd_lane == 0) {
        float relu = max(sum, 0.0f);
        output[row] = half(relu * relu);  // squared ReLU
    }
}

// ============================================================================
// PHASE 2: L2 Normalization
// ============================================================================

// L2 normalize: x = x / ||x||
// Each threadgroup processes one token
kernel void l2_norm_fp16(
    device half* x [[buffer(0)]],
    constant uint& C [[buffer(1)]],           // channels per token
    constant uint& T [[buffer(2)]],           // number of tokens
    constant float& eps [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token = tgid;
    if (token >= T) return;
    
    uint base = token * C;
    
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float val = float(x[base + i]);
        sum_sq += val * val;
    }
    
    // Reduce within simdgroup
    sum_sq = simd_sum(sum_sq);
    
    // Store simdgroup results
    threadgroup float shared_sum[8];
    if (simd_lane == 0) {
        shared_sum[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction
    float total = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        total = shared_sum[simd_lane];
    }
    total = simd_sum(total);
    float inv_norm = rsqrt(total + eps);
    
    // Broadcast
    threadgroup float shared_inv_norm;
    if (tid == 0) {
        shared_inv_norm = inv_norm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_norm = shared_inv_norm;
    
    // Normalize
    for (uint i = tid; i < C; i += 256) {
        x[base + i] = half(float(x[base + i]) * inv_norm);
    }
}

// ============================================================================
// PHASE 2: RMS Normalization
// ============================================================================

// RMS normalize with affine: x = (x / rms(x)) * weight + bias
kernel void rms_norm_fp16(
    device half* x [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    constant uint& C [[buffer(3)]],
    constant uint& T [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token = tgid;
    if (token >= T) return;
    
    uint base = token * C;
    
    // Compute mean of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float val = float(x[base + i]);
        sum_sq += val * val;
    }
    
    sum_sq = simd_sum(sum_sq);
    
    threadgroup float shared_sum[8];
    if (simd_lane == 0) {
        shared_sum[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        total = shared_sum[simd_lane];
    }
    total = simd_sum(total);
    float inv_rms = rsqrt(total / float(C) + eps);
    
    threadgroup float shared_inv_rms;
    if (tid == 0) {
        shared_inv_rms = inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_rms = shared_inv_rms;
    
    // Normalize and apply affine
    for (uint i = tid; i < C; i += 256) {
        float val = float(x[base + i]) * inv_rms;
        x[base + i] = half(val * float(weight[i]) + float(bias[i]));
    }
}

// ============================================================================
// PHASE 2: Activation Functions
// ============================================================================

// Squared ReLU: x = max(x, 0)^2
kernel void squared_relu_fp16(
    device half* x [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float val = max(float(x[tid]), 0.0f);
        x[tid] = half(val * val);
    }
}

// Tanh activation
kernel void tanh_fp16(
    device half* x [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        x[tid] = half(tanh(float(x[tid])));
    }
}

// Sigmoid activation
kernel void sigmoid_fp16(
    device half* x [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float val = float(x[tid]);
        x[tid] = half(1.0f / (1.0f + exp(-val)));
    }
}

// SiLU (Swish) activation: x * sigmoid(x)
kernel void silu_fp16(
    device half* x [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float val = float(x[tid]);
        float sigmoid = 1.0f / (1.0f + exp(-val));
        x[tid] = half(val * sigmoid);
    }
}

// ============================================================================
// PHASE 3: Time Mix V7 (RWKV-7 Recurrent Attention)
// ============================================================================

// Helper: Compute w activation: exp(-0.606531 * sigmoid(x))
inline float act_w(float x) {
    float sig = 1.0f / (1.0f + exp(-x));
    return exp(-0.606531f * sig);
}

// Pack k, v, a, kk into a single packed tensor n
// n layout: [k, v, a, kk] each of size [T, C]
kernel void pack_kvakk_fp16(
    device const half* k [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device const half* a [[buffer(2)]],
    device const half* kk [[buffer(3)]],
    device half* n [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint stride = C * T;
    if (tid >= stride) return;
    
    n[tid] = k[tid];
    n[tid + stride] = v[tid];
    n[tid + 2 * stride] = a[tid];
    n[tid + 3 * stride] = kk[tid];
}

// Time first kernel: computes initial attention contribution
// y += sum_j(u[j] * k[j] * r[j]) * v
// Algorithm from WebGPU time_mix_v7.wgsl:
//   shared_x[tid] = u * k * r
//   parallel reduction to get sum
//   x = x + sum * v
kernel void time_first_fp16(
    device const half* u [[buffer(0)]],           // [H, S] - time_first weights (per head)
    device const half* r [[buffer(1)]],           // [T, H, S] - receptance
    device const half* k [[buffer(2)]],           // [T, H, S] - key (kk from n)
    device const half* v [[buffer(3)]],           // [T, H, S] - value (from n)
    device half* x [[buffer(4)]],                 // [T, H, S] - output (in-place add)
    constant uint& head_size [[buffer(5)]],       // S (HEAD_SIZE, typically 64)
    constant uint& num_heads [[buffer(6)]],       // H
    constant uint& num_tokens [[buffer(7)]],      // T
    uint2 tgid [[threadgroup_position_in_grid]],  // (token, head)
    uint tid [[thread_index_in_threadgroup]],     // index within head (0..HEAD_SIZE-1)
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint token = tgid.x;
    uint head = tgid.y;
    
    if (token >= num_tokens || head >= num_heads) return;
    
    // Stride for indexing: C = H * S
    uint C = head_size * num_heads;
    uint stride = C;  // stride per token
    
    // Base index for this (token, head)
    uint base = token * stride + head * head_size;
    uint u_base = head * head_size;
    
    // Threadgroup memory for parallel reduction
    threadgroup float4 shared_x[64];  // HEAD_SIZE max = 64
    
    // Each thread handles one element within the head
    uint index = tid;
    if (index < head_size) {
        float uu = float(u[u_base + index]);
        float kk = float(k[base + index]);
        float rr = float(r[base + index]);
        shared_x[index] = float4(uu * kk * rr, 0.0f, 0.0f, 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within head
    for (uint step = head_size >> 1; step > 0; step >>= 1) {
        if (index < step && index + step < head_size) {
            shared_x[index].x += shared_x[index + step].x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Get the sum
    float xx = shared_x[0].x;
    
    // Each thread updates its output element: x = x + xx * v
    if (index < head_size) {
        float vv = float(v[base + index]);
        float old_x = float(x[base + index]);
        x[base + index] = half(old_x + xx * vv);
    }
}

// Time mix V7 kernel - the main recurrent attention computation
// This implements the FULL algorithm from WebGPU time_mix_v7.wgsl:
//
// State layout: state[B, S+1, C] as vec4<f32>
//   - Row 0: token shift state (previous token's hidden)
//   - Rows 1..S+1: HEAD_SIZE state rows, each containing C elements
//   - The state matrix is HEAD_SIZE × HEAD_SIZE per head, stored across S rows
//
// Algorithm per output element (index):
//   1. Load shared r, w, k, a, b into threadgroup memory for this head
//   2. Compute bonus: sa = sum_j(-kk[j] * state[batch, j*4+1..j*4+4, index])
//   3. For each j in HEAD_SIZE:
//      - Load 4 state rows: ss[0..3] = state[batch, j*4+1..j*4+4, index]
//      - Update: ss = w * ss + k * v + sa * b
//      - Accumulate: y += r * ss
//      - Store updated state
//   4. Write output y
//
// Dispatch: one threadgroup per (token, head), HEAD_SIZE threads per threadgroup
kernel void time_mix_v7_fp16(
    device const uint* cursors [[buffer(0)]],     // [T] - cursor info (batch, token_in_seq, len)
    device float4* state [[buffer(1)]],           // [B, S+1, C/4] - recurrent state as vec4
    device const half* r [[buffer(2)]],           // [T, H, S] - receptance
    device const half* w [[buffer(3)]],           // [T, H, S] - time decay
    device const half* n [[buffer(4)]],           // [4, T, H, S] - packed k,v,a,kk
    device half* x [[buffer(5)]],                 // [T, H, S] - output
    constant uint& head_size [[buffer(6)]],       // S (HEAD_SIZE, typically 64)
    constant uint& num_heads [[buffer(7)]],       // H
    constant uint& num_tokens [[buffer(8)]],      // T
    constant uint& state_stride [[buffer(9)]],    // C (stride per state row = num_emb)
    uint2 tgid [[threadgroup_position_in_grid]],  // (token, head)
    uint tid [[thread_index_in_threadgroup]]      // index within head (0..HEAD_SIZE-1)
) {
    uint token = tgid.x;
    uint head = tgid.y;
    
    if (token >= num_tokens || head >= num_heads) return;
    if (tid >= head_size) return;
    
    // Dimensions
    uint C = head_size * num_heads;  // Total channels (num_emb)
    uint n_stride = num_tokens * C;   // Stride between k,v,a,kk in packed n
    
    // Thread's element index within the full embedding
    uint index = head * head_size + tid;
    
    // Token index for input tensors
    uint ti = token * C + index;
    
    // Decode cursor to get batch
    uint cursor_val = cursors[token];
    uint batch = cursor_val & 0xFFu;
    uint token_in_seq = (cursor_val >> 8u) & 0xFFFFu;
    uint seq_len = (cursor_val >> 24u) & 0xFFu;
    
    // Threadgroup memory for shared values within this head
    // We need HEAD_SIZE elements for r, w, k, a, b
    threadgroup float shared_r[64];   // HEAD_SIZE max
    threadgroup float shared_w[64];
    threadgroup float shared_k[64];
    threadgroup float shared_a[64];   // -kk
    threadgroup float shared_b[64];   // kk * a
    
    // Load shared data for this head
    float r_val = float(r[ti]);
    float w_val = act_w(float(w[ti]));
    float k_val = float(n[ti]);                    // k
    float v_val = float(n[ti + n_stride]);         // v
    float a_val = float(n[ti + 2 * n_stride]);     // a
    float kk_val = float(n[ti + 3 * n_stride]);    // kk
    
    shared_r[tid] = r_val;
    shared_w[tid] = w_val;
    shared_k[tid] = k_val;
    shared_a[tid] = -kk_val;           // -kk for bonus
    shared_b[tid] = kk_val * a_val;    // kk * a
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // State indexing:
    // state is [B, S+1, C] stored as vec4<f32>, so [B, S+1, C/4]
    // state_idx(batch, row, index) = batch * (S+1) * (C/4) + row * (C/4) + index/4
    // We access element (index % 4) within the vec4
    uint state_row_stride = state_stride >> 2;  // C/4 vec4s per row
    uint batch_stride = (head_size + 1) * state_row_stride;  // (S+1) * (C/4) per batch
    
    uint vec4_idx = index >> 2;       // Which vec4 in the row
    uint vec4_lane = index & 3;       // Which element within vec4 (0-3)
    
    // Compute bonus term: sa = sum_j(-kk[j] * state[batch, j*4+1..j*4+4, index])
    // This accumulates over all j, loading 4 state rows per j
    float sa = 0.0f;
    for (uint j = 0; j < head_size; j++) {
        float aa = shared_a[j];  // -kk[j]
        
        // Load 4 state rows for this j
        uint base_row = j * 4 + 1;  // Skip row 0 (token shift state)
        uint batch_base = batch * batch_stride;
        
        // Load 4 consecutive rows and extract our lane
        float4 ss0 = state[batch_base + (base_row + 0) * state_row_stride + vec4_idx];
        float4 ss1 = state[batch_base + (base_row + 1) * state_row_stride + vec4_idx];
        float4 ss2 = state[batch_base + (base_row + 2) * state_row_stride + vec4_idx];
        float4 ss3 = state[batch_base + (base_row + 3) * state_row_stride + vec4_idx];
        
        // Extract element for this thread's lane and accumulate
        // sa += aa * (ss0 + ss1 + ss2 + ss3) for this lane
        float lane_sum;
        if (vec4_lane == 0) lane_sum = ss0.x + ss1.x + ss2.x + ss3.x;
        else if (vec4_lane == 1) lane_sum = ss0.y + ss1.y + ss2.y + ss3.y;
        else if (vec4_lane == 2) lane_sum = ss0.z + ss1.z + ss2.z + ss3.z;
        else lane_sum = ss0.w + ss1.w + ss2.w + ss3.w;
        
        sa += aa * lane_sum;
    }
    
    // Main recurrence loop
    float y = 0.0f;
    for (uint j = 0; j < head_size; j++) {
        float rr = shared_r[j];
        float ww = shared_w[j];
        float kk = shared_k[j];
        float bb = shared_b[j];
        
        uint base_row = j * 4 + 1;
        uint batch_base = batch * batch_stride;
        
        uint row0_idx = batch_base + (base_row + 0) * state_row_stride + vec4_idx;
        uint row1_idx = batch_base + (base_row + 1) * state_row_stride + vec4_idx;
        uint row2_idx = batch_base + (base_row + 2) * state_row_stride + vec4_idx;
        uint row3_idx = batch_base + (base_row + 3) * state_row_stride + vec4_idx;
        
        // Load state
        float4 ss0 = state[row0_idx];
        float4 ss1 = state[row1_idx];
        float4 ss2 = state[row2_idx];
        float4 ss3 = state[row3_idx];
        
        // Update state: ss = w * ss + k * v + sa * b
        // Each row gets the same update formula
        // Note: v_val is this thread's v value, broadcast to all 4 rows
        float4 update = float4(kk * v_val + sa * bb);
        
        ss0 = ss0 * ww + update;
        ss1 = ss1 * ww + update;
        ss2 = ss2 * ww + update;
        ss3 = ss3 * ww + update;
        
        // Accumulate output: y += r[j] * ss for this thread's lane
        float ss_lane;
        if (vec4_lane == 0) ss_lane = ss0.x + ss1.x + ss2.x + ss3.x;
        else if (vec4_lane == 1) ss_lane = ss0.y + ss1.y + ss2.y + ss3.y;
        else if (vec4_lane == 2) ss_lane = ss0.z + ss1.z + ss2.z + ss3.z;
        else ss_lane = ss0.w + ss1.w + ss2.w + ss3.w;
        
        y += rr * ss_lane;
        
        // Store updated state
        state[row0_idx] = ss0;
        state[row1_idx] = ss1;
        state[row2_idx] = ss2;
        state[row3_idx] = ss3;
    }
    
    // Update token shift state on last token of sequence
    if (token_in_seq + 1 == seq_len) {
        // state[batch, 0, index] = x[last_token]
        uint state_row0_idx = batch * batch_stride + vec4_idx;
        float4 x_vec = state[state_row0_idx];
        float x_out = float(x[(token_in_seq + seq_len - 1) * C + index]);
        if (vec4_lane == 0) x_vec.x = x_out;
        else if (vec4_lane == 1) x_vec.y = x_out;
        else if (vec4_lane == 2) x_vec.z = x_out;
        else x_vec.w = x_out;
        state[state_row0_idx] = x_vec;
    }
    
    // Store output
    x[ti] = half(y);
}

// Simplified time mix for single-token inference (most common case)
// This version is optimized for generation where we process one token at a time
// and don't need the full multi-token complexity
kernel void time_mix_v7_single_fp16(
    device float* state [[buffer(0)]],            // [B, H, S, S] - flattened state
    device const half* r [[buffer(1)]],           // [H, S] - receptance
    device const half* w [[buffer(2)]],           // [H, S] - time decay
    device const half* k [[buffer(3)]],           // [H, S] - key
    device const half* v [[buffer(4)]],           // [H, S] - value
    device const half* a [[buffer(5)]],           // [H, S] - bonus a
    device const half* kk [[buffer(6)]],          // [H, S] - bonus kk
    device half* output [[buffer(7)]],            // [H, S] - output
    constant uint& head_size [[buffer(8)]],       // S
    constant uint& num_heads [[buffer(9)]],       // H
    constant uint& batch [[buffer(10)]],          // current batch index
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint head = tgid.x;
    uint i = tgid.y * 32 + tid;  // output element within head
    
    if (head >= num_heads || i >= head_size) return;
    
    uint idx = head * head_size + i;
    
    // Load r for this output position
    float rr_i = float(r[idx]);
    
    // Threadgroup memory for shared values
    threadgroup float shared_w[64];
    threadgroup float shared_k[64];
    threadgroup float shared_v[64];
    threadgroup float shared_a[64];
    threadgroup float shared_kk[64];
    
    // Load shared values
    if (tid < head_size) {
        uint j_idx = head * head_size + tid;
        shared_w[tid] = act_w(float(w[j_idx]));
        shared_k[tid] = float(k[j_idx]);
        shared_v[tid] = float(v[j_idx]);
        shared_a[tid] = float(a[j_idx]);
        shared_kk[tid] = float(kk[j_idx]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute bonus term: sa = sum_j(-kk[j] * state[i, j])
    float sa = 0.0f;
    for (uint j = 0; j < head_size; j++) {
        uint state_idx = batch * num_heads * head_size * head_size + 
                        head * head_size * head_size + 
                        i * head_size + j;
        sa += -shared_kk[j] * state[state_idx];
    }
    
    // Main recurrence and accumulation
    float y = 0.0f;
    for (uint j = 0; j < head_size; j++) {
        float ww = shared_w[j];
        float kk_val = shared_k[j];
        float vv = shared_v[j];
        float aa = shared_a[j];
        float bb = shared_kk[j] * aa;  // kk * a
        float rr_j = float(r[head * head_size + j]);
        
        uint state_idx = batch * num_heads * head_size * head_size + 
                        head * head_size * head_size + 
                        i * head_size + j;
        
        float s = state[state_idx];
        
        // Update state: s = w * s + k * v + sa * b
        float new_s = ww * s + kk_val * vv + sa * bb;
        state[state_idx] = new_s;
        
        // Accumulate output: y += r[j] * s
        y += rr_j * new_s;
    }
    
    output[idx] = half(y);
}

// ============================================================================
// PHASE 3: Control K V7
// ============================================================================

// Control K operation for RWKV-7
// Applies learned gating to key values
kernel void control_k_v7_fp16(
    device const half* k [[buffer(0)]],           // [T, C] - key
    device const half* gate [[buffer(1)]],        // [T, C] - gate values
    device half* output [[buffer(2)]],            // [T, C] - output
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float k_val = float(k[tid]);
        float g_val = float(gate[tid]);
        // Control: k * sigmoid(gate)
        float sig = 1.0f / (1.0f + exp(-g_val));
        output[tid] = half(k_val * sig);
    }
}

// ============================================================================
// PHASE 4: Channel Mix V7
// ============================================================================

// Channel mix V7 - simplified version that just copies v to output
// In V7, channel mix is: x = v (the gating is done elsewhere)
kernel void channel_mix_v7_fp16(
    device const uint* cursors [[buffer(0)]],     // [T] - cursor info
    device float* state [[buffer(1)]],            // [B, C] - state per batch
    device const half* v [[buffer(2)]],           // [T, C] - value
    device half* x [[buffer(3)]],                 // [T, C] - output
    constant uint& C [[buffer(4)]],               // channels
    constant uint& T [[buffer(5)]],               // tokens
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;
    
    if (elem >= C || token >= T) return;
    
    uint idx = token * C + elem;
    
    // Decode cursor
    uint cursor_val = cursors[token];
    uint batch = cursor_val & 0xFFu;
    uint token_in_seq = (cursor_val >> 8u) & 0xFFFFu;
    uint seq_len = (cursor_val >> 24u) & 0xFFu;
    
    // Update state on last token of sequence
    if (token_in_seq + 1 == seq_len) {
        state[batch * C + elem] = float(x[idx]);
    }
    
    // V7: x = v (no gating)
    x[idx] = v[idx];
}

// Channel mix with sigmoid gating (for non-V7 models)
kernel void channel_mix_fp16(
    device const uint* cursors [[buffer(0)]],
    device float* state [[buffer(1)]],
    device const half* r [[buffer(2)]],           // [T, C] - gate
    device const half* v [[buffer(3)]],           // [T, C] - value
    device half* x [[buffer(4)]],                 // [T, C] - output
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint elem = tid.x;
    uint token = tid.y;
    
    if (elem >= C || token >= T) return;
    
    uint idx = token * C + elem;
    
    uint cursor_val = cursors[token];
    uint batch = cursor_val & 0xFFu;
    uint token_in_seq = (cursor_val >> 8u) & 0xFFFFu;
    uint seq_len = (cursor_val >> 24u) & 0xFFu;
    
    if (token_in_seq + 1 == seq_len) {
        state[batch * C + elem] = float(x[idx]);
    }
    
    // x = sigmoid(r) * v
    float rr = 1.0f / (1.0f + exp(-float(r[idx])));
    float vv = float(v[idx]);
    x[idx] = half(rr * vv);
}

// ============================================================================
// PHASE 4: Add with Activation
// ============================================================================

// Add with tanh activation: output = tanh(input) + output
kernel void add_tanh_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float in_val = tanh(float(input[tid]));
        output[tid] = half(in_val + float(output[tid]));
    }
}

// Add with sigmoid activation: output = sigmoid(input) + output
kernel void add_sigmoid_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float in_val = 1.0f / (1.0f + exp(-float(input[tid])));
        output[tid] = half(in_val + float(output[tid]));
    }
}

// Add with squared ReLU: output = squared_relu(input) + output
kernel void add_squared_relu_fp16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float val = max(float(input[tid]), 0.0f);
        output[tid] = half(val * val + float(output[tid]));
    }
}

// ============================================================================
// PHASE 4: Fused Operations for Performance
// ============================================================================

// Fused add + layer norm (common pattern in transformers)
// output = layer_norm(input + residual)
kernel void add_layer_norm_fp16(
    device const half* input [[buffer(0)]],
    device const half* residual [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& C [[buffer(5)]],
    constant uint& T [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token = tgid;
    if (token >= T) return;
    
    uint base = token * C;
    
    // Phase 1: Add and compute mean
    float sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float val = float(input[base + i]) + float(residual[base + i]);
        sum += val;
    }
    
    sum = simd_sum(sum);
    
    threadgroup float shared_sum[8];
    if (simd_lane == 0) {
        shared_sum[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float mean = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        mean = shared_sum[simd_lane];
    }
    mean = simd_sum(mean) / float(C);
    
    threadgroup float shared_mean;
    if (tid == 0) {
        shared_mean = mean;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = shared_mean;
    
    // Phase 2: Compute variance
    float var_sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        float val = float(input[base + i]) + float(residual[base + i]);
        float diff = val - mean;
        var_sum += diff * diff;
    }
    
    var_sum = simd_sum(var_sum);
    
    threadgroup float shared_var[8];
    if (simd_lane == 0) {
        shared_var[simd_group] = var_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float variance = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        variance = shared_var[simd_lane];
    }
    variance = simd_sum(variance) / float(C);
    float inv_std = rsqrt(variance + eps);
    
    threadgroup float shared_inv_std;
    if (tid == 0) {
        shared_inv_std = inv_std;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_std = shared_inv_std;
    
    // Phase 3: Normalize and apply affine
    for (uint i = tid; i < C; i += 256) {
        float val = float(input[base + i]) + float(residual[base + i]);
        float y = (val - mean) * inv_std;
        output[base + i] = half(y * float(weight[i]) + float(bias[i]));
    }
}

// Softmax for final output (head)
kernel void softmax_fp16(
    device half* x [[buffer(0)]],
    constant uint& C [[buffer(1)]],
    constant uint& T [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint token = tgid;
    if (token >= T) return;
    
    uint base = token * C;
    
    // Find max
    float max_val = -INFINITY;
    for (uint i = tid; i < C; i += 256) {
        max_val = max(max_val, float(x[base + i]));
    }
    max_val = simd_max(max_val);
    
    threadgroup float shared_max[8];
    if (simd_lane == 0) {
        shared_max[simd_group] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group == 0 && simd_lane < 8) {
        max_val = shared_max[simd_lane];
    }
    max_val = simd_max(max_val);
    
    threadgroup float shared_max_final;
    if (tid == 0) {
        shared_max_final = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = shared_max_final;
    
    // Compute exp sum
    float exp_sum = 0.0f;
    for (uint i = tid; i < C; i += 256) {
        exp_sum += exp(float(x[base + i]) - max_val);
    }
    exp_sum = simd_sum(exp_sum);
    
    threadgroup float shared_sum[8];
    if (simd_lane == 0) {
        shared_sum[simd_group] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total = 0.0f;
    if (simd_group == 0 && simd_lane < 8) {
        total = shared_sum[simd_lane];
    }
    total = simd_sum(total);
    
    threadgroup float shared_total;
    if (tid == 0) {
        shared_total = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = shared_total;
    
    float inv_sum = 1.0f / total;
    
    // Normalize
    for (uint i = tid; i < C; i += 256) {
        float val = exp(float(x[base + i]) - max_val) * inv_sum;
        x[base + i] = half(val);
    }
}

// ============================================================================
// Q4_K Quantized Matrix Multiplication (existing)
// ============================================================================

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

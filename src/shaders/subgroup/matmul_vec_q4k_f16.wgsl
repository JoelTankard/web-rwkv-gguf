// Q4_K vector-matrix multiplication with f16 arithmetic AND subgroup operations
// Combines the best of both optimizations for maximum performance

#ifdef SHADER_F16
enable f16;
#endif

struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B] logical matrix shape
@group(0) @binding(1) var<uniform> source: View;                            // [K, T, B] input
@group(0) @binding(2) var<uniform> destination: View;                       // [M, T, B] output

@group(0) @binding(3) var<storage, read> matrix: array<u32>;                // Q4_K blocks (144 bytes per 256 elements)

#ifdef IN_FP16
@group(0) @binding(4) var<storage, read> input: array<vec2<u32>>;           // (B, T, K)
#else
@group(0) @binding(4) var<storage, read> input: array<vec4<f32>>;           // (B, T, K)
#endif
#ifdef OUT_FP16
@group(0) @binding(5) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, M)
#else
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, M)
#endif

const Q4K_BLOCK_SIZE: u32 = 256u;
const Q4K_BLOCK_U32: u32 = 36u;
const NUM_SUBGROUPS: u32 = BLOCK_SIZE / MIN_SUBGROUP_SIZE;

var<workgroup> sketch: array<vec4<f32>, NUM_SUBGROUPS>;

// Shared memory cache for input vector tile (64 vec4 = 256 elements)
#ifdef SHADER_F16
var<workgroup> input_cache: array<vec4<f16>, 64>;
#else
var<workgroup> input_cache: array<vec4<f32>, 64>;
#endif

// ACTIVATION_DEFINE

fn compute_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

#ifdef SHADER_F16
fn unpack4x16half(x: vec2<u32>) -> vec4<f16> {
    return vec4<f16>(unpack2x16float(x.x), unpack2x16float(x.y));
}
#endif

fn get_scale_byte(scales_u32: array<u32, 3>, byte_idx: u32) -> u32 {
    return (scales_u32[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

#ifdef SHADER_F16
// Precompute all 8 scale/min pairs for a super-block
fn precompute_scales_h(scales_u32: array<u32, 3>, d: f16, dmin: f16) -> array<vec4<f16>, 4> {
    var result: array<vec4<f16>, 4>;
    
    // j=0,1 (first 64 elements)
    let b0 = get_scale_byte(scales_u32, 0u);
    let b1 = get_scale_byte(scales_u32, 1u);
    let b4 = get_scale_byte(scales_u32, 4u);
    let b5 = get_scale_byte(scales_u32, 5u);
    result[0] = vec4<f16>(
        d * f16(b0 & 63u), -dmin * f16(b4 & 63u),
        d * f16(b1 & 63u), -dmin * f16(b5 & 63u)
    );
    
    // j=2,3 (second 64 elements)
    let b2 = get_scale_byte(scales_u32, 2u);
    let b3 = get_scale_byte(scales_u32, 3u);
    let b6 = get_scale_byte(scales_u32, 6u);
    let b7 = get_scale_byte(scales_u32, 7u);
    result[1] = vec4<f16>(
        d * f16(b2 & 63u), -dmin * f16(b6 & 63u),
        d * f16(b3 & 63u), -dmin * f16(b7 & 63u)
    );
    
    // j=4,5 (third 64 elements)
    let sc4 = (b4 & 0xFu) | ((b0 >> 6u) << 4u);
    let m4 = (b4 >> 4u) | ((b4 >> 6u) << 4u);
    let sc5 = (b5 & 0xFu) | ((b1 >> 6u) << 4u);
    let m5 = (b5 >> 4u) | ((b5 >> 6u) << 4u);
    result[2] = vec4<f16>(
        d * f16(sc4), -dmin * f16(m4),
        d * f16(sc5), -dmin * f16(m5)
    );
    
    // j=6,7 (fourth 64 elements)
    let sc6 = (b6 & 0xFu) | ((b2 >> 6u) << 4u);
    let m6 = (b6 >> 4u) | ((b6 >> 6u) << 4u);
    let sc7 = (b7 & 0xFu) | ((b3 >> 6u) << 4u);
    let m7 = (b7 >> 4u) | ((b7 >> 6u) << 4u);
    result[3] = vec4<f16>(
        d * f16(sc6), -dmin * f16(m6),
        d * f16(sc7), -dmin * f16(m7)
    );
    
    return result;
}

// Optimized f16 dequantization with precomputed scales
// dm contains: (d*scale0, -dmin*min0, d*scale1, -dmin*min1)
fn dequant_8_opt(qs_packed: u32, dm: vec4<f16>, x_lo: vec4<f16>, x_hi: vec4<f16>) -> f16 {
    let d1 = dm.x; let m1 = dm.y;
    let d2 = dm.z; let m2 = dm.w;
    
    let q0 = f16(qs_packed & 0xFu);
    let q1 = f16((qs_packed >> 8u) & 0xFu);
    let q2 = f16((qs_packed >> 16u) & 0xFu);
    let q3 = f16((qs_packed >> 24u) & 0xFu);
    let w_lo = vec4<f16>(fma(d1, q0, m1), fma(d1, q1, m1), fma(d1, q2, m1), fma(d1, q3, m1));
    
    let q4 = f16((qs_packed >> 4u) & 0xFu);
    let q5 = f16((qs_packed >> 12u) & 0xFu);
    let q6 = f16((qs_packed >> 20u) & 0xFu);
    let q7 = f16((qs_packed >> 28u) & 0xFu);
    let w_hi = vec4<f16>(fma(d2, q4, m2), fma(d2, q5, m2), fma(d2, q6, m2), fma(d2, q7, m2));
    
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}
#else
fn get_scale_min_k4(j: u32, scales_u32: array<u32, 3>) -> vec2<f32> {
    let b_j = get_scale_byte(scales_u32, j);
    let b_j4 = get_scale_byte(scales_u32, j + 4u);
    
    let sc_lo = b_j & 63u;
    let m_lo = b_j4 & 63u;
    
    let j_minus_4 = select(0u, j - 4u, j >= 4u);
    let b_jm4 = get_scale_byte(scales_u32, j_minus_4);
    let sc_hi = (b_j4 & 0xFu) | ((b_jm4 >> 6u) << 4u);
    let m_hi = (b_j4 >> 4u) | ((b_j >> 6u) << 4u);
    
    let is_lo = j < 4u;
    let sc = select(sc_hi, sc_lo, is_lo);
    let m = select(m_hi, m_lo, is_lo);
    return vec2<f32>(f32(sc), f32(m));
}

fn dequant_8(qs_packed: u32, d1: f32, m1: f32, d2: f32, m2: f32, x_lo: vec4<f32>, x_hi: vec4<f32>) -> f32 {
    let q0 = f32(qs_packed & 0xFu);
    let q1 = f32((qs_packed >> 8u) & 0xFu);
    let q2 = f32((qs_packed >> 16u) & 0xFu);
    let q3 = f32((qs_packed >> 24u) & 0xFu);
    let w_lo = vec4<f32>(fma(d1, q0, -m1), fma(d1, q1, -m1), fma(d1, q2, -m1), fma(d1, q3, -m1));
    
    let q4 = f32((qs_packed >> 4u) & 0xFu);
    let q5 = f32((qs_packed >> 12u) & 0xFu);
    let q6 = f32((qs_packed >> 20u) & 0xFu);
    let q7 = f32((qs_packed >> 28u) & 0xFu);
    let w_hi = vec4<f32>(fma(d2, q4, -m2), fma(d2, q5, -m2), fma(d2, q6, -m2), fma(d2, q7, -m2));
    
    return dot(w_lo, x_lo) + dot(w_hi, x_hi);
}
#endif

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(
    @builtin(global_invocation_id) invocation_id: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32
) {
    let k = va.shape.x;
    let m = va.shape.y;
    let channel = invocation_id.x / BLOCK_SIZE;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u);
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;
    let row = channel * 4u;
    
    var local_sum = vec4<f32>(0.0);
    
    // Process super-blocks with input caching
    for (var sb = 0u; sb < num_super_blocks_k; sb++) {
        let k_offset = sb * Q4K_BLOCK_SIZE;
        
        // Cooperatively load input tile into shared memory (64 vec4 = 256 elements)
        if tid < 64u {
            let input_idx = bb + k_offset / 4u + tid;
#ifdef SHADER_F16
#ifdef IN_FP16
            input_cache[tid] = unpack4x16half(input[input_idx]);
#else
            input_cache[tid] = vec4<f16>(input[input_idx]);
#endif
#else
#ifdef IN_FP16
            input_cache[tid] = unpack4x16float(input[input_idx]);
#else
            input_cache[tid] = input[input_idx];
#endif
#endif
        }
        workgroupBarrier();
        
        // Each thread processes a subset of rows
        for (var r = 0u; r < 4u; r++) {
            let current_row = row + r;
            if current_row >= m {
                continue;
            }
            
            let block_u32_base = (current_row * num_super_blocks_k + sb) * Q4K_BLOCK_U32;
            
#ifdef SHADER_F16
            let d_dmin = unpack2x16float(matrix[block_u32_base]);
            let d = f16(d_dmin.x);
            let dmin = f16(d_dmin.y);
            
            let scales_u32 = array<u32, 3>(
                matrix[block_u32_base + 1u],
                matrix[block_u32_base + 2u],
                matrix[block_u32_base + 3u]
            );
            
            // Precompute all scale/min pairs once
            let dm = precompute_scales_h(scales_u32, d, dmin);
            
            var row_sum = f16(0.0);
            
            // Process 4 groups of 64 elements each - manually unrolled with precomputed scales
            {
                let qs_base = block_u32_base + 4u;
                row_sum += dequant_8_opt(matrix[qs_base + 0u], dm[0], input_cache[0u], input_cache[8u]);
                row_sum += dequant_8_opt(matrix[qs_base + 1u], dm[0], input_cache[1u], input_cache[9u]);
                row_sum += dequant_8_opt(matrix[qs_base + 2u], dm[0], input_cache[2u], input_cache[10u]);
                row_sum += dequant_8_opt(matrix[qs_base + 3u], dm[0], input_cache[3u], input_cache[11u]);
                row_sum += dequant_8_opt(matrix[qs_base + 4u], dm[0], input_cache[4u], input_cache[12u]);
                row_sum += dequant_8_opt(matrix[qs_base + 5u], dm[0], input_cache[5u], input_cache[13u]);
                row_sum += dequant_8_opt(matrix[qs_base + 6u], dm[0], input_cache[6u], input_cache[14u]);
                row_sum += dequant_8_opt(matrix[qs_base + 7u], dm[0], input_cache[7u], input_cache[15u]);
            }
            {
                let qs_base = block_u32_base + 4u + 8u;
                row_sum += dequant_8_opt(matrix[qs_base + 0u], dm[1], input_cache[16u], input_cache[24u]);
                row_sum += dequant_8_opt(matrix[qs_base + 1u], dm[1], input_cache[17u], input_cache[25u]);
                row_sum += dequant_8_opt(matrix[qs_base + 2u], dm[1], input_cache[18u], input_cache[26u]);
                row_sum += dequant_8_opt(matrix[qs_base + 3u], dm[1], input_cache[19u], input_cache[27u]);
                row_sum += dequant_8_opt(matrix[qs_base + 4u], dm[1], input_cache[20u], input_cache[28u]);
                row_sum += dequant_8_opt(matrix[qs_base + 5u], dm[1], input_cache[21u], input_cache[29u]);
                row_sum += dequant_8_opt(matrix[qs_base + 6u], dm[1], input_cache[22u], input_cache[30u]);
                row_sum += dequant_8_opt(matrix[qs_base + 7u], dm[1], input_cache[23u], input_cache[31u]);
            }
            {
                let qs_base = block_u32_base + 4u + 16u;
                row_sum += dequant_8_opt(matrix[qs_base + 0u], dm[2], input_cache[32u], input_cache[40u]);
                row_sum += dequant_8_opt(matrix[qs_base + 1u], dm[2], input_cache[33u], input_cache[41u]);
                row_sum += dequant_8_opt(matrix[qs_base + 2u], dm[2], input_cache[34u], input_cache[42u]);
                row_sum += dequant_8_opt(matrix[qs_base + 3u], dm[2], input_cache[35u], input_cache[43u]);
                row_sum += dequant_8_opt(matrix[qs_base + 4u], dm[2], input_cache[36u], input_cache[44u]);
                row_sum += dequant_8_opt(matrix[qs_base + 5u], dm[2], input_cache[37u], input_cache[45u]);
                row_sum += dequant_8_opt(matrix[qs_base + 6u], dm[2], input_cache[38u], input_cache[46u]);
                row_sum += dequant_8_opt(matrix[qs_base + 7u], dm[2], input_cache[39u], input_cache[47u]);
            }
            {
                let qs_base = block_u32_base + 4u + 24u;
                row_sum += dequant_8_opt(matrix[qs_base + 0u], dm[3], input_cache[48u], input_cache[56u]);
                row_sum += dequant_8_opt(matrix[qs_base + 1u], dm[3], input_cache[49u], input_cache[57u]);
                row_sum += dequant_8_opt(matrix[qs_base + 2u], dm[3], input_cache[50u], input_cache[58u]);
                row_sum += dequant_8_opt(matrix[qs_base + 3u], dm[3], input_cache[51u], input_cache[59u]);
                row_sum += dequant_8_opt(matrix[qs_base + 4u], dm[3], input_cache[52u], input_cache[60u]);
                row_sum += dequant_8_opt(matrix[qs_base + 5u], dm[3], input_cache[53u], input_cache[61u]);
                row_sum += dequant_8_opt(matrix[qs_base + 6u], dm[3], input_cache[54u], input_cache[62u]);
                row_sum += dequant_8_opt(matrix[qs_base + 7u], dm[3], input_cache[55u], input_cache[63u]);
            }
            
            local_sum[r] += f32(row_sum);
#else
            // f32 fallback path
            let d_dmin = unpack2x16float(matrix[block_u32_base]);
            let d = d_dmin.x;
            let dmin = d_dmin.y;
            
            let scales_u32 = array<u32, 3>(
                matrix[block_u32_base + 1u],
                matrix[block_u32_base + 2u],
                matrix[block_u32_base + 3u]
            );
            
            var row_sum = 0.0f;
            
            for (var j64 = 0u; j64 < 4u; j64++) {
                let sm0 = get_scale_min_k4(j64 * 2u, scales_u32);
                let sm1 = get_scale_min_k4(j64 * 2u + 1u, scales_u32);
                let d1 = d * sm0.x;
                let m1 = dmin * sm0.y;
                let d2 = d * sm1.x;
                let m2 = dmin * sm1.y;
                
                let qs_base = block_u32_base + 4u + j64 * 8u;
                
                for (var l = 0u; l < 8u; l++) {
                    let qs_packed = matrix[qs_base + l];
                    let cache_base = j64 * 16u + l;
                    row_sum += dequant_8(qs_packed, d1, m1, d2, m2, input_cache[cache_base], input_cache[cache_base + 8u]);
                }
            }
            
            local_sum[r] += row_sum;
#endif
        }
        workgroupBarrier();
    }
    
    // Use subgroup reduction instead of manual workgroup reduction
    local_sum = subgroupAdd(local_sum);
    
    if subgroup_invocation_id == 0u {
        sketch[subgroup_id] = local_sum;
    }
    workgroupBarrier();
    
    // Reduce across subgroups (optimized for common subgroup sizes)
#ifdef SUBGROUP_SIZE_32_32
    if tid < 2u { sketch[tid] += sketch[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u { sketch[tid] += sketch[tid + 1u]; }
    workgroupBarrier();
#else
#ifdef SUBGROUP_SIZE_32_64
    if subgroup_size == 32u {
        if tid < 2u { sketch[tid] += sketch[tid + 2u]; }
        workgroupBarrier();
    }
    if tid < 1u { sketch[tid] += sketch[tid + 1u]; }
    workgroupBarrier();
#else
#ifdef SUBGROUP_SIZE_64_64
    if tid < 1u { sketch[tid] += sketch[tid + 1u]; }
    workgroupBarrier();
#else
    for (var step = num_subgroups >> 1u; step > 0u; step >>= 1u) {
        if tid < step {
            sketch[tid] += sketch[tid + step];
        }
        workgroupBarrier();
    }
#endif
#endif
#endif

    if tid == 0u {
        let btc = compute_index(destination, batch, token, channel);
        let out = ACT(sketch[0]);
#ifdef OUT_FP16
        output[btc] = pack4x16float(out);
#else
        output[btc] = out;
#endif
    }
}

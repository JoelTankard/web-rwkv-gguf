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
var<workgroup> input_cache: array<vec4<f32>, 64>;

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

fn get_scale_byte(scales_u32: array<u32, 3>, byte_idx: u32) -> u32 {
    return (scales_u32[byte_idx / 4u] >> ((byte_idx % 4u) * 8u)) & 0xFFu;
}

fn get_scale_min_k4(j: u32, scales_u32: array<u32, 3>) -> vec2<f32> {
    var sc: u32;
    var m: u32;
    if j < 4u {
        sc = get_scale_byte(scales_u32, j) & 63u;
        m = get_scale_byte(scales_u32, j + 4u) & 63u;
    } else {
        let b1 = get_scale_byte(scales_u32, j + 4u);
        let b2 = get_scale_byte(scales_u32, j - 4u);
        let b3 = get_scale_byte(scales_u32, j);
        sc = (b1 & 0xFu) | ((b2 >> 6u) << 4u);
        m = (b1 >> 4u) | ((b3 >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(m));
}

// Vectorized dequantization: process 8 elements from one u32 of packed qs
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
#ifdef IN_FP16
            input_cache[tid] = unpack4x16float(input[input_idx]);
#else
            input_cache[tid] = input[input_idx];
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
            
            let d_dmin = unpack2x16float(matrix[block_u32_base]);
            let d = d_dmin.x;
            let dmin = d_dmin.y;
            
            let scales_u32 = array<u32, 3>(
                matrix[block_u32_base + 1u],
                matrix[block_u32_base + 2u],
                matrix[block_u32_base + 3u]
            );
            
            var row_sum = 0.0f;
            
            // Process 4 groups of 64 elements each - manually unrolled
            {
                let sm0 = get_scale_min_k4(0u, scales_u32);
                let sm1 = get_scale_min_k4(1u, scales_u32);
                let d1 = d * sm0.x; let m1 = dmin * sm0.y;
                let d2 = d * sm1.x; let m2 = dmin * sm1.y;
                let qs_base = block_u32_base + 4u;
                
                row_sum += dequant_8(matrix[qs_base + 0u], d1, m1, d2, m2, input_cache[0u], input_cache[8u]);
                row_sum += dequant_8(matrix[qs_base + 1u], d1, m1, d2, m2, input_cache[1u], input_cache[9u]);
                row_sum += dequant_8(matrix[qs_base + 2u], d1, m1, d2, m2, input_cache[2u], input_cache[10u]);
                row_sum += dequant_8(matrix[qs_base + 3u], d1, m1, d2, m2, input_cache[3u], input_cache[11u]);
                row_sum += dequant_8(matrix[qs_base + 4u], d1, m1, d2, m2, input_cache[4u], input_cache[12u]);
                row_sum += dequant_8(matrix[qs_base + 5u], d1, m1, d2, m2, input_cache[5u], input_cache[13u]);
                row_sum += dequant_8(matrix[qs_base + 6u], d1, m1, d2, m2, input_cache[6u], input_cache[14u]);
                row_sum += dequant_8(matrix[qs_base + 7u], d1, m1, d2, m2, input_cache[7u], input_cache[15u]);
            }
            {
                let sm0 = get_scale_min_k4(2u, scales_u32);
                let sm1 = get_scale_min_k4(3u, scales_u32);
                let d1 = d * sm0.x; let m1 = dmin * sm0.y;
                let d2 = d * sm1.x; let m2 = dmin * sm1.y;
                let qs_base = block_u32_base + 4u + 8u;
                
                row_sum += dequant_8(matrix[qs_base + 0u], d1, m1, d2, m2, input_cache[16u], input_cache[24u]);
                row_sum += dequant_8(matrix[qs_base + 1u], d1, m1, d2, m2, input_cache[17u], input_cache[25u]);
                row_sum += dequant_8(matrix[qs_base + 2u], d1, m1, d2, m2, input_cache[18u], input_cache[26u]);
                row_sum += dequant_8(matrix[qs_base + 3u], d1, m1, d2, m2, input_cache[19u], input_cache[27u]);
                row_sum += dequant_8(matrix[qs_base + 4u], d1, m1, d2, m2, input_cache[20u], input_cache[28u]);
                row_sum += dequant_8(matrix[qs_base + 5u], d1, m1, d2, m2, input_cache[21u], input_cache[29u]);
                row_sum += dequant_8(matrix[qs_base + 6u], d1, m1, d2, m2, input_cache[22u], input_cache[30u]);
                row_sum += dequant_8(matrix[qs_base + 7u], d1, m1, d2, m2, input_cache[23u], input_cache[31u]);
            }
            {
                let sm0 = get_scale_min_k4(4u, scales_u32);
                let sm1 = get_scale_min_k4(5u, scales_u32);
                let d1 = d * sm0.x; let m1 = dmin * sm0.y;
                let d2 = d * sm1.x; let m2 = dmin * sm1.y;
                let qs_base = block_u32_base + 4u + 16u;
                
                row_sum += dequant_8(matrix[qs_base + 0u], d1, m1, d2, m2, input_cache[32u], input_cache[40u]);
                row_sum += dequant_8(matrix[qs_base + 1u], d1, m1, d2, m2, input_cache[33u], input_cache[41u]);
                row_sum += dequant_8(matrix[qs_base + 2u], d1, m1, d2, m2, input_cache[34u], input_cache[42u]);
                row_sum += dequant_8(matrix[qs_base + 3u], d1, m1, d2, m2, input_cache[35u], input_cache[43u]);
                row_sum += dequant_8(matrix[qs_base + 4u], d1, m1, d2, m2, input_cache[36u], input_cache[44u]);
                row_sum += dequant_8(matrix[qs_base + 5u], d1, m1, d2, m2, input_cache[37u], input_cache[45u]);
                row_sum += dequant_8(matrix[qs_base + 6u], d1, m1, d2, m2, input_cache[38u], input_cache[46u]);
                row_sum += dequant_8(matrix[qs_base + 7u], d1, m1, d2, m2, input_cache[39u], input_cache[47u]);
            }
            {
                let sm0 = get_scale_min_k4(6u, scales_u32);
                let sm1 = get_scale_min_k4(7u, scales_u32);
                let d1 = d * sm0.x; let m1 = dmin * sm0.y;
                let d2 = d * sm1.x; let m2 = dmin * sm1.y;
                let qs_base = block_u32_base + 4u + 24u;
                
                row_sum += dequant_8(matrix[qs_base + 0u], d1, m1, d2, m2, input_cache[48u], input_cache[56u]);
                row_sum += dequant_8(matrix[qs_base + 1u], d1, m1, d2, m2, input_cache[49u], input_cache[57u]);
                row_sum += dequant_8(matrix[qs_base + 2u], d1, m1, d2, m2, input_cache[50u], input_cache[58u]);
                row_sum += dequant_8(matrix[qs_base + 3u], d1, m1, d2, m2, input_cache[51u], input_cache[59u]);
                row_sum += dequant_8(matrix[qs_base + 4u], d1, m1, d2, m2, input_cache[52u], input_cache[60u]);
                row_sum += dequant_8(matrix[qs_base + 5u], d1, m1, d2, m2, input_cache[53u], input_cache[61u]);
                row_sum += dequant_8(matrix[qs_base + 6u], d1, m1, d2, m2, input_cache[54u], input_cache[62u]);
                row_sum += dequant_8(matrix[qs_base + 7u], d1, m1, d2, m2, input_cache[55u], input_cache[63u]);
            }
            
            local_sum[r] += row_sum;
        }
        workgroupBarrier();
    }
    
    // Use subgroup reduction
    local_sum = subgroupAdd(local_sum);
    
    if subgroup_invocation_id == 0u {
        sketch[subgroup_id] = local_sum;
    }
    workgroupBarrier();
    
    // Reduce across subgroups
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

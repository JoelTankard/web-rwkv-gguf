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
const Q4K_BLOCK_BYTES: u32 = 144u;
const Q4K_BLOCK_U32: u32 = 36u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;

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

fn reduce_sum(index: u32, stride: u32) {
    if index < stride {
        sketch[index] += sketch[index + stride];
    }
    workgroupBarrier();
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

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let k = va.shape.x;
    let m = va.shape.y;
    let index = invocation_id.x % BLOCK_SIZE;
    let channel = invocation_id.x / BLOCK_SIZE;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = compute_index(source, batch, token, 0u);
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;
    let row = channel * 4u;
    
    var local_sum = vec4<f32>(0.0);
    
    for (var sb = index; sb < num_super_blocks_k; sb += BLOCK_SIZE) {
        let k_offset = sb * Q4K_BLOCK_SIZE;
        
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
            var is = 0u;
            
            for (var j64 = 0u; j64 < 4u; j64++) {
                let sm0 = get_scale_min_k4(is, scales_u32);
                let sm1 = get_scale_min_k4(is + 1u, scales_u32);
                let d1 = d * sm0.x;
                let m1 = dmin * sm0.y;
                let d2 = d * sm1.x;
                let m2 = dmin * sm1.y;
                
                let qs_u32_base = block_u32_base + 4u + j64 * 8u;
                
                for (var l = 0u; l < 8u; l++) {
                    let qs_packed = matrix[qs_u32_base + l];
                    
                    let elem_lo = k_offset + j64 * 64u + l * 4u;
                    let elem_hi = elem_lo + 32u;
                    let input_idx_lo = bb + elem_lo / 4u;
                    let input_idx_hi = bb + elem_hi / 4u;
                    
#ifdef IN_FP16
                    let x_lo = unpack4x16float(input[input_idx_lo]);
                    let x_hi = unpack4x16float(input[input_idx_hi]);
#else
                    let x_lo = input[input_idx_lo];
                    let x_hi = input[input_idx_hi];
#endif
                    
                    let q0 = f32(qs_packed & 0xFu);
                    let q1 = f32((qs_packed >> 8u) & 0xFu);
                    let q2 = f32((qs_packed >> 16u) & 0xFu);
                    let q3 = f32((qs_packed >> 24u) & 0xFu);
                    let w_lo = vec4<f32>(d1 * q0 - m1, d1 * q1 - m1, d1 * q2 - m1, d1 * q3 - m1);
                    
                    let q4 = f32((qs_packed >> 4u) & 0xFu);
                    let q5 = f32((qs_packed >> 12u) & 0xFu);
                    let q6 = f32((qs_packed >> 20u) & 0xFu);
                    let q7 = f32((qs_packed >> 28u) & 0xFu);
                    let w_hi = vec4<f32>(d2 * q4 - m2, d2 * q5 - m2, d2 * q6 - m2, d2 * q7 - m2);
                    
                    row_sum += dot(w_lo, x_lo) + dot(w_hi, x_hi);
                }
                is += 2u;
            }
            local_sum[r] += row_sum;
        }
    }
    
    sketch[index] = local_sum;
    workgroupBarrier();

    reduce_sum(index, 64u);
    reduce_sum(index, 32u);
    reduce_sum(index, 16u);
    reduce_sum(index, 8u);
    reduce_sum(index, 4u);
    reduce_sum(index, 2u);
    reduce_sum(index, 1u);

    if index == 0u {
        let btc = compute_index(destination, batch, token, channel);
        let out = ACT(sketch[0]);
#ifdef OUT_FP16
        output[btc] = pack4x16float(out);
#else
        output[btc] = out;
#endif
    }
}

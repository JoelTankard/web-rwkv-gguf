struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B] logical matrix shape
@group(0) @binding(1) var<uniform> source: View;                            // [K, T, B] input
@group(0) @binding(2) var<uniform> destination: View;                       // [M, T, B] output

@group(0) @binding(3) var<storage, read> matrix: array<u32>;                // Q6_K blocks (210 bytes per 256 elements)

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

const Q6K_BLOCK_SIZE: u32 = 256u;
const Q6K_BLOCK_BYTES: u32 = 210u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> input_cache: array<vec4<f32>, 64u>;

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

fn read_block_byte(block_byte_base: u32, byte_offset: u32) -> u32 {
    let u32_idx = block_byte_base + byte_offset / 4u;
    let byte_in_u32 = byte_offset % 4u;
    return (matrix[u32_idx] >> (byte_in_u32 * 8u)) & 0xFFu;
}

fn get_scale_i8(block_byte_base: u32, idx: u32) -> f32 {
    let byte_val = read_block_byte(block_byte_base, 192u + idx);
    return f32(i32(byte_val) - 128);
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
    let num_super_blocks_k = k / Q6K_BLOCK_SIZE;
    let row = channel * 4u;
    
    var local_sum = vec4<f32>(0.0);
    
    for (var sb = index; sb < num_super_blocks_k; sb += BLOCK_SIZE) {
        let k_offset = sb * Q6K_BLOCK_SIZE;
        
        if index < 64u {
            let input_idx = bb + k_offset / 4u + index;
#ifdef IN_FP16
            input_cache[index] = unpack4x16float(input[input_idx]);
#else
            input_cache[index] = input[input_idx];
#endif
        }
        workgroupBarrier();
        
        for (var r = 0u; r < 4u; r++) {
            let current_row = row + r;
            if current_row >= m {
                continue;
            }
            
            let block_byte_base = (current_row * num_super_blocks_k + sb) * Q6K_BLOCK_BYTES / 4u;
            
            let d_packed = matrix[block_byte_base + 52u];
            let d = unpack2x16float(d_packed).x;
            
            var row_sum = 0.0f;
            var ql_idx = 0u;
            var qh_idx = 0u;
            var sc_idx = 0u;
            var out_idx = 0u;
            
            for (var n128 = 0u; n128 < 2u; n128++) {
                for (var l = 0u; l < 32u; l += 4u) {
                    let is = l / 16u;
                    
                    let ql_u32_0 = matrix[block_byte_base + ql_idx / 4u + l / 4u];
                    let ql_u32_1 = matrix[block_byte_base + ql_idx / 4u + (l + 32u) / 4u];
                    let qh_u32 = matrix[block_byte_base + 32u + qh_idx / 4u + l / 4u];
                    
                    let sc0 = get_scale_i8(block_byte_base, sc_idx + is);
                    let sc2 = get_scale_i8(block_byte_base, sc_idx + is + 2u);
                    let sc4 = get_scale_i8(block_byte_base, sc_idx + is + 4u);
                    let sc6 = get_scale_i8(block_byte_base, sc_idx + is + 6u);
                    
                    let q1_0 = f32(i32((ql_u32_0 & 0xFu) | (((qh_u32 >> 0u) & 3u) << 4u)) - 32);
                    let q1_1 = f32(i32(((ql_u32_0 >> 8u) & 0xFu) | (((qh_u32 >> 8u) & 3u) << 4u)) - 32);
                    let q1_2 = f32(i32(((ql_u32_0 >> 16u) & 0xFu) | (((qh_u32 >> 16u) & 3u) << 4u)) - 32);
                    let q1_3 = f32(i32(((ql_u32_0 >> 24u) & 0xFu) | (((qh_u32 >> 24u) & 3u) << 4u)) - 32);
                    
                    let q2_0 = f32(i32((ql_u32_1 & 0xFu) | (((qh_u32 >> 2u) & 3u) << 4u)) - 32);
                    let q2_1 = f32(i32(((ql_u32_1 >> 8u) & 0xFu) | (((qh_u32 >> 10u) & 3u) << 4u)) - 32);
                    let q2_2 = f32(i32(((ql_u32_1 >> 16u) & 0xFu) | (((qh_u32 >> 18u) & 3u) << 4u)) - 32);
                    let q2_3 = f32(i32(((ql_u32_1 >> 24u) & 0xFu) | (((qh_u32 >> 26u) & 3u) << 4u)) - 32);
                    
                    let q3_0 = f32(i32((ql_u32_0 >> 4u) & 0xFu | (((qh_u32 >> 4u) & 3u) << 4u)) - 32);
                    let q3_1 = f32(i32(((ql_u32_0 >> 12u) & 0xFu) | (((qh_u32 >> 12u) & 3u) << 4u)) - 32);
                    let q3_2 = f32(i32(((ql_u32_0 >> 20u) & 0xFu) | (((qh_u32 >> 20u) & 3u) << 4u)) - 32);
                    let q3_3 = f32(i32(((ql_u32_0 >> 28u) & 0xFu) | (((qh_u32 >> 28u) & 3u) << 4u)) - 32);
                    
                    let q4_0 = f32(i32((ql_u32_1 >> 4u) & 0xFu | (((qh_u32 >> 6u) & 3u) << 4u)) - 32);
                    let q4_1 = f32(i32(((ql_u32_1 >> 12u) & 0xFu) | (((qh_u32 >> 14u) & 3u) << 4u)) - 32);
                    let q4_2 = f32(i32(((ql_u32_1 >> 20u) & 0xFu) | (((qh_u32 >> 22u) & 3u) << 4u)) - 32);
                    let q4_3 = f32(i32(((ql_u32_1 >> 28u) & 0xFu) | (((qh_u32 >> 30u) & 3u) << 4u)) - 32);
                    
                    let w1 = d * sc0 * vec4<f32>(q1_0, q1_1, q1_2, q1_3);
                    let w2 = d * sc2 * vec4<f32>(q2_0, q2_1, q2_2, q2_3);
                    let w3 = d * sc4 * vec4<f32>(q3_0, q3_1, q3_2, q3_3);
                    let w4 = d * sc6 * vec4<f32>(q4_0, q4_1, q4_2, q4_3);
                    
                    let cache_base = out_idx / 4u;
                    let x1 = input_cache[cache_base + l / 4u];
                    let x2 = input_cache[cache_base + (l + 32u) / 4u];
                    let x3 = input_cache[cache_base + (l + 64u) / 4u];
                    let x4 = input_cache[cache_base + (l + 96u) / 4u];
                    
                    row_sum += dot(w1, x1) + dot(w2, x2) + dot(w3, x3) + dot(w4, x4);
                }
                
                ql_idx += 64u;
                qh_idx += 32u;
                sc_idx += 8u;
                out_idx += 128u;
            }
            local_sum[r] += row_sum;
        }
        workgroupBarrier();
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

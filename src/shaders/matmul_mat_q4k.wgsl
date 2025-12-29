struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

struct Input {
    @builtin(workgroup_id) bid: vec3<u32>,
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(local_invocation_index) index: u32,
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B] logical matrix shape
@group(0) @binding(1) var<uniform> vb: View;                                // [K, N, B] input
@group(0) @binding(2) var<uniform> destination: View;                       // [M, N, B] output

@group(0) @binding(3) var<storage, read> matrix: array<u32>;                // Q4_K blocks (144 bytes per 256 elements)
#ifdef IN_FP16
@group(0) @binding(4) var<storage, read> xb: array<vec2<u32>>;              // (B, N, K)
#else
@group(0) @binding(4) var<storage, read> xb: array<vec4<f32>>;              // (B, N, K)
#endif
#ifdef OUT_FP16
@group(0) @binding(5) var<storage, read_write> output: array<vec2<u32>>;    // (B, N, M)
#else
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, N, M)
#endif

const TILE_SIZE: u32 = BLOCK_SIZE * 4u;
const Q4K_BLOCK_SIZE: u32 = 256u;
const Q4K_BLOCK_U32: u32 = 36u;

#ifdef IN_FP16
var<workgroup> sb: array<array<vec2<u32>, BLOCK_SIZE>, TILE_SIZE>;
#else
var<workgroup> sb: array<array<vec4<f32>, BLOCK_SIZE>, TILE_SIZE>;
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

fn dequant_q4k_element(block_u32_base: u32, element_idx: u32, d: f32, dmin: f32, scales_u32: array<u32, 3>) -> f32 {
    let j64 = element_idx / 64u;
    let is = j64 * 2u;
    let in_64 = element_idx % 64u;
    
    let sm0 = get_scale_min_k4(is, scales_u32);
    let sm1 = get_scale_min_k4(is + 1u, scales_u32);
    
    let qs_byte_offset = 4u + (j64 * 32u) + (in_64 % 32u);
    let qs_u32_idx = block_u32_base + (qs_byte_offset / 4u);
    let qs_byte_in_u32 = qs_byte_offset % 4u;
    let qs_byte = (matrix[qs_u32_idx] >> (qs_byte_in_u32 * 8u)) & 0xFFu;
    
    if in_64 < 32u {
        let q4 = qs_byte & 0xFu;
        return d * sm0.x * f32(q4) - dmin * sm0.y;
    } else {
        let q4 = qs_byte >> 4u;
        return d * sm1.x * f32(q4) - dmin * sm1.y;
    }
}

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn matmul(in: Input) {
    let k = va.shape.x;
    let m = va.shape.y;
    let b = in.bid.xy * TILE_SIZE;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    let stride = rb.x;
    
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;

    var local_sum: mat4x4<f32>;
    
    for (var k_tile = 0u; k_tile < stride; k_tile += BLOCK_SIZE) {
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            let i = in.tid.x;
            let x = k_tile + i;
            let y = b.y + j;
            if all(vec2<u32>(x, y) < rb) {
                sb[j][i] = xb[compute_index(vb, in.uid.z, y, x)];
            } else {
#ifdef IN_FP16
                sb[j][i] = vec2<u32>(0u);
#else
                sb[j][i] = vec4<f32>(0.0);
#endif
            }
        }
        workgroupBarrier();

        if all(u < vec2<u32>(m, rb.y)) {
            for (var x = 0u; x < BLOCK_SIZE; x += 1u) {
                let k_idx = (k_tile + x) * 4u;
                if k_idx >= k {
                    break;
                }
                
                var aa: mat4x4<f32>;
                for (var r = 0u; r < 4u; r++) {
                    let row = u.x + r;
                    if row < m {
                        let sb_idx = row * num_super_blocks_k + (k_idx / Q4K_BLOCK_SIZE);
                        let block_u32_base = sb_idx * Q4K_BLOCK_U32;
                        let elem_in_block = k_idx % Q4K_BLOCK_SIZE;
                        
                        let d_dmin = unpack2x16float(matrix[block_u32_base]);
                        let d = d_dmin.x;
                        let dmin = d_dmin.y;
                        let scales_u32 = array<u32, 3>(
                            matrix[block_u32_base + 1u],
                            matrix[block_u32_base + 2u],
                            matrix[block_u32_base + 3u]
                        );
                        
                        aa[r] = vec4<f32>(
                            dequant_q4k_element(block_u32_base, elem_in_block, d, dmin, scales_u32),
                            dequant_q4k_element(block_u32_base, elem_in_block + 1u, d, dmin, scales_u32),
                            dequant_q4k_element(block_u32_base, elem_in_block + 2u, d, dmin, scales_u32),
                            dequant_q4k_element(block_u32_base, elem_in_block + 3u, d, dmin, scales_u32)
                        );
                    } else {
                        aa[r] = vec4<f32>(0.0);
                    }
                }
                
#ifdef IN_FP16
                let bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][x]),
                    unpack4x16float(sb[t.y + 1u][x]),
                    unpack4x16float(sb[t.y + 2u][x]),
                    unpack4x16float(sb[t.y + 3u][x]),
                );
#else
                let bb = mat4x4<f32>(
                    sb[t.y][x],
                    sb[t.y + 1u][x],
                    sb[t.y + 2u][x],
                    sb[t.y + 3u][x],
                );
#endif
                local_sum += transpose(aa) * bb;
            }
        }
        workgroupBarrier();
    }

    if all(u < vec2<u32>(m, rb.y)) {
        local_sum[0] = ACT(local_sum[0]);
        local_sum[1] = ACT(local_sum[1]);
        local_sum[2] = ACT(local_sum[2]);
        local_sum[3] = ACT(local_sum[3]);
#ifdef OUT_FP16
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x)] = pack4x16float(local_sum[0]);
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x)] = pack4x16float(local_sum[1]);
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x)] = pack4x16float(local_sum[2]);
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x)] = pack4x16float(local_sum[3]);
#else
        output[compute_index(destination, in.uid.z, u.y + 0u, in.uid.x)] = local_sum[0];
        output[compute_index(destination, in.uid.z, u.y + 1u, in.uid.x)] = local_sum[1];
        output[compute_index(destination, in.uid.z, u.y + 2u, in.uid.x)] = local_sum[2];
        output[compute_index(destination, in.uid.z, u.y + 3u, in.uid.x)] = local_sum[3];
#endif
    }
}

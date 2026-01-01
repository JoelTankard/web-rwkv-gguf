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

// Optimized tile configuration
// BLOCK_SIZE x BLOCK_SIZE threads per workgroup
// Each thread computes a 4x4 output tile
// Total output tile: (BLOCK_SIZE*4) x (BLOCK_SIZE*4) = TILE_SIZE x TILE_SIZE
const TILE_SIZE: u32 = BLOCK_SIZE * 4u;

const Q4K_BLOCK_SIZE: u32 = 256u;
const Q4K_BLOCK_U32: u32 = 36u;

// Shared memory for input tile (TILE_SIZE rows x BLOCK_SIZE vec4s)
#ifdef IN_FP16
var<workgroup> sb: array<array<vec2<u32>, BLOCK_SIZE>, TILE_SIZE>;
#else
var<workgroup> sb: array<array<vec4<f32>, BLOCK_SIZE>, TILE_SIZE>;
#endif

// Shared memory for dequantized weight tile (TILE_SIZE rows x BLOCK_SIZE vec4s)
var<workgroup> sa: array<array<vec4<f32>, BLOCK_SIZE>, TILE_SIZE>;

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

// Dequantize a vec4 of elements from Q4_K at a given K position
// Uses transposed layout: block[sb_k][row] for coalesced memory access
fn dequant_q4k_vec4(row: u32, k_pos: u32, num_super_blocks_k: u32, num_rows: u32) -> vec4<f32> {
    let sb_idx = k_pos / Q4K_BLOCK_SIZE;
    let pos_in_sb = k_pos % Q4K_BLOCK_SIZE;
    
    // Transposed layout: block[sb_idx][row] instead of block[row][sb_idx]
    let block_u32_base = (sb_idx * num_rows + row) * Q4K_BLOCK_U32;
    let d_dmin = unpack2x16float(matrix[block_u32_base]);
    let d = d_dmin.x;
    let dmin = d_dmin.y;
    
    let scales_u32 = array<u32, 3>(
        matrix[block_u32_base + 1u],
        matrix[block_u32_base + 2u],
        matrix[block_u32_base + 3u]
    );
    
    // Determine j64 (which 64-element group) and position within
    let j64 = pos_in_sb / 64u;
    let pos_in_j64 = pos_in_sb % 64u;
    
    // Get scale/min for this position
    let sm_idx = j64 * 2u + select(0u, 1u, pos_in_j64 >= 32u);
    let sm = get_scale_min_k4(sm_idx, scales_u32);
    let d_scale = d * sm.x;
    let m_val = dmin * sm.y;
    
    // Get the packed qs byte
    let l = (pos_in_j64 % 32u) / 4u;
    let qs_packed = matrix[block_u32_base + 4u + j64 * 8u + l];
    
    var w: vec4<f32>;
    if pos_in_j64 < 32u {
        // Low nibbles
        let q0 = f32(qs_packed & 0xFu);
        let q1 = f32((qs_packed >> 8u) & 0xFu);
        let q2 = f32((qs_packed >> 16u) & 0xFu);
        let q3 = f32((qs_packed >> 24u) & 0xFu);
        w = vec4<f32>(fma(d_scale, q0, -m_val), fma(d_scale, q1, -m_val), fma(d_scale, q2, -m_val), fma(d_scale, q3, -m_val));
    } else {
        // High nibbles
        let q0 = f32((qs_packed >> 4u) & 0xFu);
        let q1 = f32((qs_packed >> 12u) & 0xFu);
        let q2 = f32((qs_packed >> 20u) & 0xFu);
        let q3 = f32((qs_packed >> 28u) & 0xFu);
        w = vec4<f32>(fma(d_scale, q0, -m_val), fma(d_scale, q1, -m_val), fma(d_scale, q2, -m_val), fma(d_scale, q3, -m_val));
    }
    
    return w;
}

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn matmul(in: Input) {
    let k = va.shape.x;
    let m = va.shape.y;
    let b = in.bid.xy * TILE_SIZE;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let ra = vec2<u32>(va.shape.x / 4u, va.shape.y);
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    let stride = min(ra.x, rb.x);
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;

    var local_sum: mat4x4<f32>;
    
    // Process K dimension in BLOCK_SIZE vec4 chunks (BLOCK_SIZE * 4 elements)
    for (var kk = 0u; kk < stride; kk += BLOCK_SIZE) {
        // Load TILE_SIZE rows of input and weights, each with BLOCK_SIZE vec4s
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            let i = in.tid.x;
            let x = kk + i;
            
            // Load weight row (dequantize Q4_K on the fly)
            var y = b.x + j;
            if y < ra.y && x < ra.x {
                let k_pos = x * 4u;  // Convert vec4 index to element index
                sa[j][i] = dequant_q4k_vec4(y, k_pos, num_super_blocks_k, m);
            } else {
                sa[j][i] = vec4<f32>(0.0);
            }

            // Load input row
            y = b.y + j;
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

        // Each thread multiplies and sums up 4x4 blocks along the reduced dimension
        if all(u < vec2<u32>(ra.y, rb.y)) {
            for (var x = 0u; x < BLOCK_SIZE; x += 1u) {
                if kk + x >= stride {
                    break;
                }
                let aa = mat4x4<f32>(
                    sa[t.x][x],
                    sa[t.x + 1u][x],
                    sa[t.x + 2u][x],
                    sa[t.x + 3u][x],
                );
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

    if all(u < vec2<u32>(ra.y, rb.y)) {
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

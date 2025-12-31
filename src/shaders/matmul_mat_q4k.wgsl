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

// Shared memory for input tile (N dimension x K tile)
// TILE_SIZE tokens x 64 vec4 (256 elements)
#ifdef IN_FP16
var<workgroup> sb: array<array<vec2<u32>, 64>, TILE_SIZE>;
#else
var<workgroup> sb: array<array<vec4<f32>, 64>, TILE_SIZE>;
#endif

// Shared memory for dequantized weight tile
// TILE_SIZE rows (M) x 64 vec4 (256 K elements)
// Dequantize once, reuse across all N tokens in the tile
var<workgroup> sa: array<array<vec4<f32>, 64>, TILE_SIZE>;

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
    // Branchless: compute both paths and select
    let b_j = get_scale_byte(scales_u32, j);
    let b_j4 = get_scale_byte(scales_u32, j + 4u);
    
    // Path for j < 4: sc = b_j & 63, m = b_j4 & 63
    let sc_lo = b_j & 63u;
    let m_lo = b_j4 & 63u;
    
    // Path for j >= 4: need j-4 byte as well
    // Use saturating subtract to avoid underflow when j < 4
    let j_minus_4 = select(0u, j - 4u, j >= 4u);
    let b_jm4 = get_scale_byte(scales_u32, j_minus_4);
    let sc_hi = (b_j4 & 0xFu) | ((b_jm4 >> 6u) << 4u);
    let m_hi = (b_j4 >> 4u) | ((b_j >> 6u) << 4u);
    
    let is_lo = j < 4u;
    let sc = select(sc_hi, sc_lo, is_lo);
    let m = select(m_hi, m_lo, is_lo);
    return vec2<f32>(f32(sc), f32(m));
}

@compute @workgroup_size(BLOCK_SIZE, BLOCK_SIZE, 1)
fn matmul(in: Input) {
    let k = va.shape.x;
    let m = va.shape.y;
    let b = in.bid.xy * TILE_SIZE;
    let u = in.uid.xy * 4u;
    let t = in.tid.xy * 4u;
    let rb = vec2<u32>(vb.shape.x / 4u, vb.shape.y);
    
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;

    var local_sum: mat4x4<f32>;
    
    // Process K dimension in super-block sized tiles (256 elements = 64 vec4)
    for (var sb_k = 0u; sb_k < num_super_blocks_k; sb_k++) {
        let k_offset = sb_k * Q4K_BLOCK_SIZE;
        
        // Cooperatively load input tile (TILE_SIZE tokens x 64 vec4)
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            for (var i = in.tid.x; i < 64u; i += BLOCK_SIZE) {
                let y = b.y + j;
                let x = k_offset / 4u + i;
                if y < rb.y && x < rb.x {
#ifdef IN_FP16
                    sb[j][i] = xb[compute_index(vb, in.uid.z, y, x)];
#else
                    sb[j][i] = xb[compute_index(vb, in.uid.z, y, x)];
#endif
                } else {
#ifdef IN_FP16
                    sb[j][i] = vec2<u32>(0u);
#else
                    sb[j][i] = vec4<f32>(0.0);
#endif
                }
            }
        }
        
        // Cooperatively dequantize weight tile (TILE_SIZE rows x 256 elements)
        // Each thread handles multiple (row, k_element) pairs
        for (var j = in.tid.y; j < TILE_SIZE; j += BLOCK_SIZE) {
            let row = b.x + j;
            if row >= m {
                for (var i = in.tid.x; i < 64u; i += BLOCK_SIZE) {
                    sa[j][i] = vec4<f32>(0.0);
                }
                continue;
            }
            
            let block_u32_base = (row * num_super_blocks_k + sb_k) * Q4K_BLOCK_U32;
            let d_dmin = unpack2x16float(matrix[block_u32_base]);
            let d = d_dmin.x;
            let dmin = d_dmin.y;
            
            let scales_u32 = array<u32, 3>(
                matrix[block_u32_base + 1u],
                matrix[block_u32_base + 2u],
                matrix[block_u32_base + 3u]
            );
            
            // Each thread dequantizes elements at stride BLOCK_SIZE
            for (var i = in.tid.x; i < 32u; i += BLOCK_SIZE) {
                let j64 = i / 8u;
                let l = i % 8u;
                
                let is = j64 * 2u;
                let sm0 = get_scale_min_k4(is, scales_u32);
                let sm1 = get_scale_min_k4(is + 1u, scales_u32);
                let d1 = d * sm0.x;
                let m1 = dmin * sm0.y;
                let d2 = d * sm1.x;
                let m2 = dmin * sm1.y;
                
                let qs_packed = matrix[block_u32_base + 4u + j64 * 8u + l];
                
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
                
                sa[j][j64 * 16u + l] = w_lo;
                sa[j][j64 * 16u + 8u + l] = w_hi;
            }
        }
        workgroupBarrier();
        
        // Compute 4x4 output tile using cached weights and inputs
        if all(u < vec2<u32>(m, rb.y)) {
            for (var ki = 0u; ki < 64u; ki++) {
                let aa = mat4x4<f32>(
                    sa[t.x][ki],
                    sa[t.x + 1u][ki],
                    sa[t.x + 2u][ki],
                    sa[t.x + 3u][ki],
                );
#ifdef IN_FP16
                let bb = mat4x4<f32>(
                    unpack4x16float(sb[t.y][ki]),
                    unpack4x16float(sb[t.y + 1u][ki]),
                    unpack4x16float(sb[t.y + 2u][ki]),
                    unpack4x16float(sb[t.y + 3u][ki]),
                );
#else
                let bb = mat4x4<f32>(
                    sb[t.y][ki],
                    sb[t.y + 1u][ki],
                    sb[t.y + 2u][ki],
                    sb[t.y + 3u][ki],
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

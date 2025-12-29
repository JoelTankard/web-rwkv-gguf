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

// Cache for input vector - 256 elements per super-block = 64 vec4
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

// Compute dot product of one Q4_K super-block with cached input
// Reads 36 u32s (144 bytes) of Q4K data, uses cached input (no extra reads)
fn dot_q4k_cached(block_u32_base: u32) -> f32 {
    let d_dmin = unpack2x16float(matrix[block_u32_base]);
    let d = d_dmin.x;
    let dmin = d_dmin.y;
    
    let scales_u32 = array<u32, 3>(
        matrix[block_u32_base + 1u],
        matrix[block_u32_base + 2u],
        matrix[block_u32_base + 3u]
    );
    
    var sum = 0.0f;
    
    // Process 4 groups of 64 elements (j64 = 0,1,2,3)
    for (var j64 = 0u; j64 < 4u; j64++) {
        let sm0 = get_scale_min_k4(j64 * 2u, scales_u32);
        let sm1 = get_scale_min_k4(j64 * 2u + 1u, scales_u32);
        let d1 = d * sm0.x;
        let m1 = dmin * sm0.y;
        let d2 = d * sm1.x;
        let m2 = dmin * sm1.y;
        
        let qs_base = block_u32_base + 4u + j64 * 8u;
        let cache_base = j64 * 16u;
        
        // Process 8 u32s
        for (var l = 0u; l < 8u; l++) {
            let qs = matrix[qs_base + l];
            let x_lo = input_cache[cache_base + l];
            let x_hi = input_cache[cache_base + 8u + l];
            
            // Low nibbles
            sum += fma(d1, f32(qs & 0xFu), -m1) * x_lo.x;
            sum += fma(d1, f32((qs >> 8u) & 0xFu), -m1) * x_lo.y;
            sum += fma(d1, f32((qs >> 16u) & 0xFu), -m1) * x_lo.z;
            sum += fma(d1, f32((qs >> 24u) & 0xFu), -m1) * x_lo.w;
            
            // High nibbles
            sum += fma(d2, f32((qs >> 4u) & 0xFu), -m2) * x_hi.x;
            sum += fma(d2, f32((qs >> 12u) & 0xFu), -m2) * x_hi.y;
            sum += fma(d2, f32((qs >> 20u) & 0xFu), -m2) * x_hi.z;
            sum += fma(d2, f32((qs >> 28u) & 0xFu), -m2) * x_hi.w;
        }
    }
    
    return sum;
}

// Each workgroup processes BLOCK_SIZE output rows
// Each thread handles one output row, iterating over all K super-blocks
// Input is cached in shared memory and reused across all rows
@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn matmul(@builtin(global_invocation_id) invocation_id: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let k = va.shape.x;
    let m = va.shape.y;
    let token = invocation_id.y;
    let batch = invocation_id.z;
    
    // Each thread handles one output row (4 elements packed as vec4)
    let row = invocation_id.x * 4u;
    
    let input_base = compute_index(source, batch, token, 0u);
    let num_super_blocks_k = k / Q4K_BLOCK_SIZE;
    
    var local_sum = vec4<f32>(0.0);
    
    // Process all super-blocks
    // Key: cache input once per super-block, reuse for all BLOCK_SIZE rows
    for (var sb = 0u; sb < num_super_blocks_k; sb++) {
        // Cooperatively load input for this super-block into shared memory
        // 64 vec4s = 256 elements, BLOCK_SIZE threads load them
        if tid < 64u {
            let idx = input_base + sb * 64u + tid;
#ifdef IN_FP16
            input_cache[tid] = unpack4x16float(input[idx]);
#else
            input_cache[tid] = input[idx];
#endif
        }
        workgroupBarrier();
        
        // Each thread processes its own row using the cached input
        // This is where we save bandwidth: read Q4K (144 bytes) instead of F16 (512 bytes)
        // Input is read once and shared across BLOCK_SIZE rows
        if row < m {
            for (var r = 0u; r < 4u; r++) {
                if row + r < m {
                    let block_u32_base = ((row + r) * num_super_blocks_k + sb) * Q4K_BLOCK_U32;
                    local_sum[r] += dot_q4k_cached(block_u32_base);
                }
            }
        }
        
        workgroupBarrier();
    }
    
    // Write output directly - no reduction needed since each thread handles one row
    if row < m {
        let out_idx = compute_index(destination, batch, token, invocation_id.x);
        let out = ACT(local_sum);
#ifdef OUT_FP16
        output[out_idx] = pack4x16float(out);
#else
        output[out_idx] = out;
#endif
    }
}

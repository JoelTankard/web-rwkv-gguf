struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> va: View;                                // [K, M, B] logical matrix shape
@group(0) @binding(1) var<uniform> source: View;                            // [K, T, B] input
@group(0) @binding(2) var<uniform> destination: View;                       // [M, T, B] output

@group(0) @binding(3) var<storage, read> matrix: array<u32>;                // Q8_0 blocks (34 bytes per 32 elements)

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

const Q8_0_BLOCK_SIZE: u32 = 32u;
const Q8_0_BLOCK_BYTES: u32 = 34u;

var<workgroup> sketch: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> input_cache: array<vec4<f32>, 32u>;

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

fn unpack4x8snorm_custom(x: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(i32(x << 24u) >> 24),
        f32(i32(x << 16u) >> 24),
        f32(i32(x << 8u) >> 24),
        f32(i32(x) >> 24)
    );
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
    let num_blocks_k = k / Q8_0_BLOCK_SIZE;
    let row = channel * 4u;
    
    var local_sum = vec4<f32>(0.0);
    
    for (var blk = index; blk < num_blocks_k; blk += BLOCK_SIZE) {
        let k_offset = blk * Q8_0_BLOCK_SIZE;
        
        if index < 8u {
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
            
            let block_idx = current_row * num_blocks_k + blk;
            let block_u32_base = block_idx * 9u;
            
            let scale_packed = matrix[block_u32_base];
            let scale = unpack2x16float(scale_packed).x;
            
            var row_sum = 0.0f;
            
            for (var l = 0u; l < 8u; l++) {
                let qs_packed = matrix[block_u32_base + 1u + l];
                let q = unpack4x8snorm_custom(qs_packed);
                let w = scale * q;
                let x = input_cache[l];
                row_sum += dot(w, x);
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

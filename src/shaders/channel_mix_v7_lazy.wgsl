struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, 1]
@group(0) @binding(1) var<uniform> view: View;                              // [C, 1, B]
@group(0) @binding(2) var<uniform> batch_info: vec4<u32>;                   // [batch_index, token_offset, num_tokens, 0]
@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>;     // (B, C)

#ifdef FP16
@group(0) @binding(5) var<storage, read> v: array<vec2<u32>>;               // (T, C)
@group(0) @binding(6) var<storage, read_write> x: array<vec2<u32>>;         // (T, C)
#else
@group(0) @binding(5) var<storage, read> v: array<vec4<f32>>;               // (T, C)
@group(0) @binding(6) var<storage, read_write> x: array<vec4<f32>>;         // (T, C)
#endif

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
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

fn load_v(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(v[index]);
#else
    return v[index];
#endif
}

fn load_x(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(x[index]);
#else
    return x[index];
#endif
}

fn store_x(index: u32, value: vec4<f32>) {
#ifdef FP16
    x[index] = pack4x16float(value);
#else
    x[index] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn channel_mix(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = batch_info.x;
    let num_tokens = batch_info.z;

    let ti = token * stride + index;

    if index >= stride {
        return;
    }

    if token + 1u == num_tokens {
        state[compute_index(batch, 0u, index)] = load_x(ti);
    }

    store_x(ti, load_v(ti));
}

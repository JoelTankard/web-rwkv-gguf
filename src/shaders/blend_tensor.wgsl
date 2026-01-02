struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> vx: View;                                // [C, T, B]
@group(0) @binding(1) var<uniform> vf: View;                                // [C, T?, B?] or [C, 1, 1]

#ifdef IN_FP16
@group(0) @binding(2) var<storage, read> lhs: array<vec2<u32>>;             // (B, T, C)
@group(0) @binding(3) var<storage, read> rhs: array<vec2<u32>>;             // (B, T, C)
#else
@group(0) @binding(2) var<storage, read> lhs: array<vec4<f32>>;             // (B, T, C)
@group(0) @binding(3) var<storage, read> rhs: array<vec4<f32>>;             // (B, T, C)
#endif

#ifdef FACTOR_FP16
@group(0) @binding(4) var<storage, read> factor: array<vec2<u32>>;          // (B?, T?, C)
#else
@group(0) @binding(4) var<storage, read> factor: array<vec4<f32>>;          // (B?, T?, C)
#endif

#ifdef OUT_FP16
@group(0) @binding(5) var<storage, read_write> output: array<vec2<u32>>;    // (B, T, C)
#else
@group(0) @binding(5) var<storage, read_write> output: array<vec4<f32>>;    // (B, T, C)
#endif

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

fn load_factor(batch: u32, token: u32, index: u32) -> vec4<f32> {
    let b = select(batch, 0u, vf.shape.z == 1u);
    let t = select(token, 0u, vf.shape.y == 1u);
#ifdef FACTOR_FP16
    return unpack4x16float(factor[compute_index(vf, b, t, index)]);
#else
    return factor[compute_index(vf, b, t, index)];
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn blend_tensor(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = vec3<u32>(vx.shape.x >> 2u, vx.shape.y, vx.shape.z);
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if any(vec3<u32>(index, token, batch) >= stride) {
        return;
    }

    let idx = compute_index(vx, batch, token, index);
    let f = load_factor(batch, token, index);

#ifdef IN_FP16
    let a = unpack4x16float(lhs[idx]);
    let b = unpack4x16float(rhs[idx]);
#else
    let a = lhs[idx];
    let b = rhs[idx];
#endif

    let result = a * f + b * (vec4<f32>(1.0) - f);

#ifdef OUT_FP16
    output[idx] = pack4x16float(result);
#else
    output[idx] = result;
#endif
}

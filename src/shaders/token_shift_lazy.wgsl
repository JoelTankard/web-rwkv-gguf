struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

@group(0) @binding(0) var<uniform> vx: View;                                // [C, T, B]
@group(0) @binding(1) var<uniform> vm: View;                                // [C, T?, 1] or [C, 1, 1]

#ifdef MIX_FP16
@group(0) @binding(2) var<storage, read> time_mix: array<vec2<u32>>;        // (1, T?, C)
#else
@group(0) @binding(2) var<storage, read> time_mix: array<vec4<f32>>;        // (1, T?, C)
#endif

@group(0) @binding(3) var<storage, read> state: array<vec4<f32>>;           // (B, 1, C)

#ifdef IN_FP16
@group(0) @binding(4) var<storage, read> x: array<vec2<u32>>;               // (B, T, C)
#else
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;               // (B, T, C)
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

fn load_mix(token: u32, index: u32) -> vec4<f32> {
    let t = select(token, 0u, vm.shape.y == 1u);
#ifdef MIX_FP16
    return unpack4x16float(time_mix[compute_index(vm, 0u, t, index)]);
#else
    return time_mix[compute_index(vm, 0u, t, index)];
#endif
}

fn load_input(batch: u32, token: u32, index: u32) -> vec4<f32> {
#ifdef IN_FP16
    return unpack4x16float(x[compute_index(vx, batch, token, index)]);
#else
    return x[compute_index(vx, batch, token, index)];
#endif
}

fn load_state(batch: u32, index: u32) -> vec4<f32> {
    let stride = vx.shape.x >> 2u;
    return state[batch * stride + index];
}

fn store_output(batch: u32, token: u32, index: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    output[compute_index(vx, batch, token, index)] = pack4x16float(value);
#else
    output[compute_index(vx, batch, token, index)] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn token_shift_lazy(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = vec3<u32>(vx.shape.x >> 2u, vx.shape.y, vx.shape.z);
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    if any(vec3<u32>(index, token, batch) >= stride) {
        return;
    }

    let factor = load_mix(token, index);
    let current = load_input(batch, token, index);

#ifdef REVERSED
    if token == 0u {
        let prev = load_state(batch, index);
        let out = mix(current, prev, factor);
        store_output(batch, token, index, out);
    } else {
        let prev = load_input(batch, token - 1u, index);
        let out = mix(current, prev, factor);
        store_output(batch, token, index, out);
    }
#else
    if token == 0u {
        let prev = load_state(batch, index);
        let out = mix(prev, current, factor);
        store_output(batch, token, index, out);
    } else {
        let prev = load_input(batch, token - 1u, index);
        let out = mix(prev, current, factor);
        store_output(batch, token, index, out);
    }
#endif
}

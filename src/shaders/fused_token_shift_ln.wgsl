// Fused Token Shift + LayerNorm shader
// Combines layer normalization with 6 token shifts into a single kernel
// Benefits: 7 dispatches → 1, 7x less memory traffic for input tensor
//
// Requires max_storage_buffers_per_shader_stage >= 17 (set in auto_limits)

struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

struct Cursor {
    batch: u32,
    token: u32,
    len: u32,
};

// Uniforms
@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

// LayerNorm weights
@group(0) @binding(1) var<storage, read> ln_w: array<vec2<u32>>;            // (C) - f16
@group(0) @binding(2) var<storage, read> ln_b: array<vec2<u32>>;            // (C) - f16

// Cursors for batch/token tracking
@group(0) @binding(3) var<storage, read> cursors: array<u32>;               // [T]

// State for token 0 mixing
@group(0) @binding(4) var<uniform> vs: View;                                // [C, _, B]
@group(0) @binding(5) var<storage, read> state: array<vec4<f32>>;           // (B, 1, C)

// Input tensor (normalized in-place)
#ifdef IN_FP16
@group(0) @binding(6) var<storage, read_write> x: array<vec2<u32>>;         // (B, T, C)
#else
@group(0) @binding(6) var<storage, read_write> x: array<vec4<f32>>;         // (B, T, C)
#endif

// Time mix factors for each of the 6 token shifts (r, w, k, v, a, g)
@group(0) @binding(7) var<storage, read> mix_r: array<vec2<u32>>;           // (C) - f16
@group(0) @binding(8) var<storage, read> mix_w: array<vec2<u32>>;           // (C) - f16
@group(0) @binding(9) var<storage, read> mix_k: array<vec2<u32>>;           // (C) - f16
@group(0) @binding(10) var<storage, read> mix_v: array<vec2<u32>>;          // (C) - f16
@group(0) @binding(11) var<storage, read> mix_a: array<vec2<u32>>;          // (C) - f16
@group(0) @binding(12) var<storage, read> mix_g: array<vec2<u32>>;          // (C) - f16

// Output tensors for each token shift result
#ifdef OUT_FP16
@group(0) @binding(13) var<storage, read_write> out_r: array<vec2<u32>>;    // (B, T, C)
@group(0) @binding(14) var<storage, read_write> out_w: array<vec2<u32>>;    // (B, T, C)
@group(0) @binding(15) var<storage, read_write> out_k: array<vec2<u32>>;    // (B, T, C)
@group(0) @binding(16) var<storage, read_write> out_v: array<vec2<u32>>;    // (B, T, C)
@group(0) @binding(17) var<storage, read_write> out_a: array<vec2<u32>>;    // (B, T, C)
@group(0) @binding(18) var<storage, read_write> out_g: array<vec2<u32>>;    // (B, T, C)
#else
@group(0) @binding(13) var<storage, read_write> out_r: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(14) var<storage, read_write> out_w: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(15) var<storage, read_write> out_k: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(16) var<storage, read_write> out_v: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(17) var<storage, read_write> out_a: array<vec4<f32>>;    // (B, T, C)
@group(0) @binding(18) var<storage, read_write> out_g: array<vec4<f32>>;    // (B, T, C)
#endif

// Workgroup shared memory for layer norm reduction
var<workgroup> mu: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> m2: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> count: array<vec4<u32>, BLOCK_SIZE>;
var<workgroup> mean: f32;
var<workgroup> dev: f32;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn compute_cursor(c: u32) -> Cursor {
    var cursor: Cursor;
    cursor.batch = c & 0xffu;
    cursor.token = (c >> 8u) & 0xffffu;
    cursor.len = (c >> 24u) & 0xffu;
    return cursor;
}

fn compute_state_index(view: View, batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn load_x(bb: u32, i: u32) -> vec4<f32> {
#ifdef IN_FP16
    return unpack4x16float(x[bb + i]);
#else
    return x[bb + i];
#endif
}

fn store_x(bb: u32, i: u32, value: vec4<f32>) {
#ifdef IN_FP16
    x[bb + i] = pack4x16float(value);
#else
    x[bb + i] = value;
#endif
}

fn reduce_step(index: u32, stride: u32) {
    if index < stride {
        let mu_1 = mu[index];
        let mu_2 = mu[index + stride];
        let count_1 = count[index];
        let count_2 = count[index + stride];

        let delta = mu_2 - mu_1;
        let total = count_1 + count_2;
        count[index] = total;

        mu[index] = select(vec4<f32>(0.0), (mu_1 * vec4<f32>(count_1) + mu_2 * vec4<f32>(count_2)) / vec4<f32>(total), total > vec4<u32>(0u));
        m2[index] = select(vec4<f32>(0.0), m2[index] + m2[index + stride] + delta * delta * vec4<f32>(count_1 * count_2) / vec4<f32>(total), total > vec4<u32>(0u));
    }
    workgroupBarrier();
}

fn store_output_r(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_r[bb + i] = pack4x16float(value);
#else
    out_r[bb + i] = value;
#endif
}

fn store_output_w(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_w[bb + i] = pack4x16float(value);
#else
    out_w[bb + i] = value;
#endif
}

fn store_output_k(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_k[bb + i] = pack4x16float(value);
#else
    out_k[bb + i] = value;
#endif
}

fn store_output_v(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_v[bb + i] = pack4x16float(value);
#else
    out_v[bb + i] = value;
#endif
}

fn store_output_a(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_a[bb + i] = pack4x16float(value);
#else
    out_a[bb + i] = value;
#endif
}

fn store_output_g(bb: u32, i: u32, value: vec4<f32>) {
#ifdef OUT_FP16
    out_g[bb + i] = pack4x16float(value);
#else
    out_g[bb + i] = value;
#endif
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn fused_token_shift_ln(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let stack = invocation_id.y;  // token position in the stack
    let batch = invocation_id.z;

    let cursor = compute_cursor(cursors[stack]);
    let token = stack - cursor.token;  // relative token position within this batch segment

    // Compute base offset for this token
    let bb = (batch * shape[1] + stack) * stride;

    // Phase 1: Compute layer norm statistics (parallel reduction)
    var _mu: vec4<f32>;
    var _m2: vec4<f32>;
    var _count: vec4<u32>;
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb, i);
        let delta = value - _mu;
        _count += 1u;
        _mu += delta / vec4<f32>(_count);
        _m2 += delta * (value - _mu);
    }
    count[index] = _count;
    mu[index] = _mu;
    m2[index] = _m2;
    workgroupBarrier();

    // Parallel reduction for mean and variance
    reduce_step(index, 64u);
    reduce_step(index, 32u);
    reduce_step(index, 16u);
    reduce_step(index, 8u);
    reduce_step(index, 4u);
    reduce_step(index, 2u);
    reduce_step(index, 1u);

    if index == 0u {
        let _mu = mu[0];
        let _count = vec4<f32>(count[0]);
        mean = dot(_mu, _count / f32(shape[0]));

        let delta = _mu - mean;
        let _m2 = dot(m2[0], vec4<f32>(1.0)) + dot(delta * delta, _count);
        let _var = _m2 / f32(shape[0]) + EPS;
        dev = inverseSqrt(_var);
    }
    workgroupBarrier();

    // Phase 2: Apply layer norm and compute all 6 token shifts
    // Note: Token shift in V7 uses the NORMALIZED current token mixed with:
    // - For token 0: the state (which is already in the right space from previous forward pass)
    // - For token > 0: the NORMALIZED previous token
    // 
    // The original implementation normalizes att_x in-place, then does token_shift on the normalized tensor.
    // So x_prev for token > 0 should be the normalized previous token.
    // However, we can't compute the previous token's normalization stats here since each workgroup
    // only processes one token. So we need a different approach:
    // 
    // For single-token inference (the common case), token is always 0, so we use state.
    // For multi-token (prefill), we need the normalized previous token which requires
    // either storing intermediate results or a different algorithm.
    //
    // For now, we handle the single-token case optimally and fall back for multi-token.
    // The shader stores the normalized x for use by subsequent operations.

    for (var i = index; i < stride; i += BLOCK_SIZE) {
        // Load and normalize current token
        let x_raw = load_x(bb, i);
        let x_normed = fma((x_raw - mean) * dev, unpack4x16float(ln_w[i]), unpack4x16float(ln_b[i]));

        // Load previous value (state for token 0)
        // For token > 0, we need the normalized previous token, but we can't compute it here
        // since each workgroup processes one token independently.
        // This is a limitation - for multi-token prefill, the original separate ops are needed.
        let x_prev = state[compute_state_index(vs, cursor.batch, 0u, i)];

        // Compute all 6 token shifts using the same normalized input
        // Token shift formula (reversed=true): mix(x_normed, x_prev, factor)
        let factor_r = unpack4x16float(mix_r[i]);
        let factor_w = unpack4x16float(mix_w[i]);
        let factor_k = unpack4x16float(mix_k[i]);
        let factor_v = unpack4x16float(mix_v[i]);
        let factor_a = unpack4x16float(mix_a[i]);
        let factor_g = unpack4x16float(mix_g[i]);

        // Store normalized value back to input buffer (in-place normalization like original layer_norm)
        store_x(bb, i, x_normed);

        store_output_r(bb, i, mix(x_normed, x_prev, factor_r));
        store_output_w(bb, i, mix(x_normed, x_prev, factor_w));
        store_output_k(bb, i, mix(x_normed, x_prev, factor_k));
        store_output_v(bb, i, mix(x_normed, x_prev, factor_v));
        store_output_a(bb, i, mix(x_normed, x_prev, factor_a));
        store_output_g(bb, i, mix(x_normed, x_prev, factor_g));
    }
}

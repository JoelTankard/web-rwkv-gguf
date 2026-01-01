struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
};

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                    // [C, T, 1, 1]

#ifdef FP16
@group(0) @binding(1) var<storage, read> k: array<vec2<u32>>;           // (T, C)
@group(0) @binding(2) var<storage, read> v: array<vec2<u32>>;           // (T, C)
@group(0) @binding(3) var<storage, read> a: array<vec2<u32>>;           // (T, C)
@group(0) @binding(4) var<storage, read> kk: array<vec2<u32>>;          // (T, C)
@group(0) @binding(5) var<storage, read_write> n: array<vec2<u32>>;     // (4, T, C)
#else
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;           // (T, C)
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;           // (T, C)
@group(0) @binding(3) var<storage, read> a: array<vec4<f32>>;           // (T, C)
@group(0) @binding(4) var<storage, read> kk: array<vec4<f32>>;          // (T, C)
@group(0) @binding(5) var<storage, read_write> n: array<vec4<f32>>;     // (4, T, C)
#endif

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn main(in: Input) {
    let stride = shape[0] * shape[1] / 4u;
    let index = in.uid.x;
    
    if index >= stride {
        return;
    }
    
    // Copy k, v, a, kk into packed n tensor
    // n layout: [k[0..stride], v[0..stride], a[0..stride], kk[0..stride]]
    n[index] = k[index];
    n[index + stride] = v[index];
    n[index + 2u * stride] = a[index];
    n[index + 3u * stride] = kk[index];
}

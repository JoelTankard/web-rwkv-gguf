// GPU-side argmax sampling shader
// Computes softmax and returns the index of the maximum value
// This avoids reading back 260KB of logits per token

@group(0) @binding(0) var<uniform> shape: vec4<u32>;                        // [C, T, B]

#ifdef FP16
@group(0) @binding(1) var<storage, read> x: array<vec2<u32>>;               // (B, T, C) - input logits
#else
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;               // (B, T, C) - input logits
#endif

@group(0) @binding(2) var<storage, read_write> output: array<u32>;          // (B, T) - output token indices

var<workgroup> sketch_max: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> sketch_idx: array<vec4<u32>, BLOCK_SIZE>;
var<workgroup> global_max: f32;
var<workgroup> global_idx: u32;

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn load_x(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(x[index]);
#else
    return x[index];
#endif
}

// Reduce to find max value and its index
fn reduce_max_idx(index: u32, stride: u32) {
    if index < stride {
        let other_max = sketch_max[index + stride];
        let other_idx = sketch_idx[index + stride];
        let my_max = sketch_max[index];
        let my_idx = sketch_idx[index];
        
        // Compare each component and keep the one with higher value
        var result_max = my_max;
        var result_idx = my_idx;
        
        if other_max.x > my_max.x {
            result_max.x = other_max.x;
            result_idx.x = other_idx.x;
        }
        if other_max.y > my_max.y {
            result_max.y = other_max.y;
            result_idx.y = other_idx.y;
        }
        if other_max.z > my_max.z {
            result_max.z = other_max.z;
            result_idx.z = other_idx.z;
        }
        if other_max.w > my_max.w {
            result_max.w = other_max.w;
            result_idx.w = other_idx.w;
        }
        
        sketch_max[index] = result_max;
        sketch_idx[index] = result_idx;
    }
    workgroupBarrier();
}

@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn sample_argmax(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let stride = shape[0] / 4u;
    let index = invocation_id.x;
    let token = invocation_id.y;
    let batch = invocation_id.z;

    let bb = (batch * shape[1] + token) * stride;

    // First pass: find local max and its index
    var local_max = vec4<f32>(-1.0e30);
    var local_idx = vec4<u32>(0u);
    
    for (var i = index; i < stride; i += BLOCK_SIZE) {
        let value = load_x(bb + i);
        let base_idx = i * 4u;
        
        if value.x > local_max.x {
            local_max.x = value.x;
            local_idx.x = base_idx;
        }
        if value.y > local_max.y {
            local_max.y = value.y;
            local_idx.y = base_idx + 1u;
        }
        if value.z > local_max.z {
            local_max.z = value.z;
            local_idx.z = base_idx + 2u;
        }
        if value.w > local_max.w {
            local_max.w = value.w;
            local_idx.w = base_idx + 3u;
        }
    }
    
    sketch_max[index] = local_max;
    sketch_idx[index] = local_idx;
    workgroupBarrier();

    // Reduce across workgroup
    reduce_max_idx(index, 64u);
    reduce_max_idx(index, 32u);
    reduce_max_idx(index, 16u);
    reduce_max_idx(index, 8u);
    reduce_max_idx(index, 4u);
    reduce_max_idx(index, 2u);
    reduce_max_idx(index, 1u);

    // Final reduction of the 4 components
    if index == 0u {
        let final_max = sketch_max[0];
        let final_idx = sketch_idx[0];
        
        var best_val = final_max.x;
        var best_idx = final_idx.x;
        
        if final_max.y > best_val {
            best_val = final_max.y;
            best_idx = final_idx.y;
        }
        if final_max.z > best_val {
            best_val = final_max.z;
            best_idx = final_idx.z;
        }
        if final_max.w > best_val {
            best_idx = final_idx.w;
        }
        
        output[batch * shape[1] + token] = best_idx;
    }
}

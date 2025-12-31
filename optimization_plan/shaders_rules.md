Below is a **practical, WebGPU-specific shader optimization checklist**, focused on **WGSL**, **Apple Silicon / Metal backend**, and **browser constraints**. This is not theoretical — it’s what actually moves perf.

---

## 1. Measure First (non-negotiable)

Before changing shaders, confirm **where the time is going**.

**What to check**

-   CPU → GPU submission time
-   GPU time per pass
-   Pipeline switches
-   Bind group changes

**Tools**

-   Chrome DevTools → Performance → WebGPU
-   Safari Web Inspector (Metal-backed, very useful on macOS)
-   Timestamp queries (`GPUQuerySet`) for GPU-side timing

```wgsl
// Use timestamp queries around passes (pseudo)
commandEncoder.writeTimestamp(querySet, 0);
// dispatch / draw
commandEncoder.writeTimestamp(querySet, 1);
```

If GPU time is low → **don’t touch shaders yet**.

---

## 2. Reduce Memory Bandwidth (biggest win)

WebGPU is often **memory-bound**, especially on Apple Silicon.

### Prefer smaller types

```wgsl
// BAD
var<storage> data: array<vec4<f32>>;

// BETTER (if precision allows)
var<storage> data: array<vec4<f16>>;
```

Use:

-   `f16` when possible
-   `vec2` over `vec4`
-   pack scalars

### Avoid random memory access

```wgsl
// BAD (cache-unfriendly)
let v = buffer[indexMap[i]];

// BETTER (linear)
let v = buffer[i];
```

If you _must_ scatter/gather → pre-sort on CPU or pre-pass compute.

---

## 3. Kill Branches (especially dynamic ones)

### Avoid divergent control flow

```wgsl
// BAD (divergent per-invocation)
if (input.x > threshold) {
  a = foo();
} else {
  a = bar();
}
```

### Replace with math

```wgsl
let t = step(threshold, input.x);
a = mix(bar(), foo(), t);
```

This is **huge** for fragment shaders.

---

## 4. Move Work Out of Fragment Shaders

Fragment shaders are the most expensive stage.

### Common wins

-   Precompute in **compute shaders**
-   Move math to **vertex**
-   Bake constants on CPU

```wgsl
// BAD: heavy math per fragment
let n = normalize(cross(dFdx(p), dFdy(p)));

// BETTER: compute per-vertex or pre-pass
```

Rule:

> If it doesn’t need per-pixel accuracy, don’t do it in fragment.

---

## 5. Use Workgroup Memory (Compute Shaders)

Shared memory is **orders of magnitude faster** than storage buffers.

```wgsl
var<workgroup> tile: array<f32, 256>;

tile[local_id.x] = input[global_id.x];
workgroupBarrier();

let v = tile[someIndex];
```

Use for:

-   Reductions
-   Filters
-   Matrix tiles
-   Prefix sums

---

## 6. Tune Workgroup Size (very important)

Bad defaults = wasted cores.

```wgsl
@compute @workgroup_size(64)
fn main(...)
```

### General guidance

-   Apple Silicon: **32–64**
-   Desktop GPUs: **64–256**
-   Test powers of two

Try:

```
32, 64, 128
```

Measure. Don’t guess.

---

## 7. Minimize Bind Group & Pipeline Changes

This is _not_ a shader change — but it dwarfs shader micro-opts.

### Bad

```js
for (...) {
  pass.setPipeline(pipelineA);
  pass.setBindGroup(0, groupA);
  pass.draw();
}
```

### Better

-   Batch by pipeline
-   Batch by bind group
-   Use **dynamic offsets**

```js
pass.setBindGroup(0, group, [offset]);
```

---

## 8. Inline & Remove Dead Code

WGSL compilers are decent — but not magic.

-   Inline small functions
-   Remove unused outputs
-   Avoid debug paths in production

```wgsl
// BAD
fn helper(x: f32) -> f32 {
  return x * 1.0;
}

// BETTER
let y = x;
```

Yes, this still matters in hot paths.

---

## 9. Avoid Expensive Built-ins

### Expensive

-   `pow`
-   `exp`
-   `log`
-   `normalize` (when repeated)
-   `length` (uses sqrt)

### Replace when possible

```wgsl
// BAD
let d = length(v);

// BETTER (if only comparison)
let d2 = dot(v, v);
```

Approximate when visuals allow.

---

## 10. Precision Discipline (Apple GPUs care)

Apple GPUs strongly benefit from:

-   `f16`
-   Fewer registers
-   Smaller structs

```wgsl
struct Params {
  a: f16,
  b: f16,
  c: f16,
  d: f16,
};
```

This can double occupancy.

---

## 11. Don’t Fight the Browser

WebGPU drivers already do:

-   Instruction reordering
-   Vectorization
-   Dead code elimination

Avoid:

-   Micro-unrolling loops manually
-   Obscure bit hacks

**High-level clarity often optimizes better.**

---

## 12. Real-World Order of Impact

If you want ROI, do this **in order**:

1. Reduce draw/dispatch count
2. Reduce bind group & pipeline switches
3. Reduce memory bandwidth
4. Kill divergent branches
5. Move work out of fragment shaders
6. Tune workgroup size
7. Math micro-optimizations

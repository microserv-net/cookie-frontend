// The orb.
//
// A single full-screen triangle; everything else is maths. The body is a
// sphere raymarched in screen space with a domain-warped FBM field standing
// in for the plasma, plus a rim shell and an outer bloom. There is no
// texture, no mesh and no imported asset: every frame is generated from the
// uniform block below, which is why the entity can react to a voice rather
// than replay an animation.
//
// Cost is deliberately modest — 18 march steps, 4 FBM octaves — so this runs
// on integrated graphics at 60 fps while a model is loading on the CPU.

struct Orb {
    // x: time, y: radius, z: wobble, w: turbulence
    a: vec4<f32>,
    // x: swirl, y: flow_speed, z: noise_scale, w: detail
    b: vec4<f32>,
    // x: glow, y: brightness, z: alpha, w: hue (degrees)
    c: vec4<f32>,
    // x: hue_spread, y: saturation, z: shell, w: core
    d: vec4<f32>,
    // x: distortion, y: offset.x, z: offset.y, w: spin
    e: vec4<f32>,
    // x: energy, y: low, z: mid, w: high
    f: vec4<f32>,
    // x: onset, y: seed, z: aspect, w: background alpha
    g: vec4<f32>,
    // x: 1 when the surface expects premultiplied colour, 0 when it does not
    h: vec4<f32>,
};

@group(0) @binding(0) var<uniform> orb: Orb;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// One oversized triangle covers the viewport with no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(index) - 1) * 2.0;
    let y = f32(i32(index & 1u) * 4 - 1) * 2.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3));
    let r = q * 17.0;
    return fract(r.x * r.y * r.z * (r.x + r.y + r.z));
}

// Value noise with smooth interpolation. Cheaper than gradient noise and the
// difference is invisible once it has been through four octaves of FBM and a
// domain warp.
fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash3(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash3(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(n000, n100, u.x);
    let x10 = mix(n010, n110, u.x);
    let x01 = mix(n001, n101, u.x);
    let x11 = mix(n011, n111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

fn fbm(p: vec3<f32>, detail: f32) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var total = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        let weight = amplitude * mix(1.0, f32(4 - i) / 4.0, 1.0 - detail);
        value = value + weight * noise3(p * frequency);
        total = total + weight;
        amplitude = amplitude * 0.55;
        frequency = frequency * 2.07;
    }
    return value / max(total, 0.0001);
}

fn rotate(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let s = sin(angle);
    let c = cos(angle);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

// HSV is the right space here: behaviours think in "hue, a bit more
// saturated, brighter", and the conversion is a handful of instructions.
fn hsv(h: f32, s: f32, v: f32) -> vec3<f32> {
    let hue = fract(h / 360.0) * 6.0;
    let i = floor(hue);
    let f = hue - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    if (i < 1.0) { return vec3<f32>(v, t, p); }
    if (i < 2.0) { return vec3<f32>(q, v, p); }
    if (i < 3.0) { return vec3<f32>(p, v, t); }
    if (i < 4.0) { return vec3<f32>(p, q, v); }
    if (i < 5.0) { return vec3<f32>(t, p, v); }
    return vec3<f32>(v, p, q);
}

// Density of the plasma at a point inside the unit sphere.
fn density(pos: vec3<f32>, radius: f32) -> f32 {
    let time = orb.a.x;
    let wobble = orb.a.z;
    let turbulence = orb.a.w;
    let swirl = orb.b.x;
    let flow = orb.b.y;
    let scale = orb.b.z;
    let detail = orb.b.w;
    let seed = orb.g.y * 37.0;

    var p = pos;
    // Swirl increases toward the centre, which is what makes the interior
    // look like it is being stirred rather than spun as a rigid body.
    let r = length(p);
    let twist = swirl * (1.0 - clamp(r, 0.0, 1.0)) * 1.6;
    let xz = rotate(p.xz, twist + time * 0.05);
    p = vec3<f32>(xz.x, p.y, xz.y);

    // Domain warp: noise displacing the sample point is what produces the
    // smoke-like filaments in the reference image.
    let warp = vec3<f32>(
        fbm(p * scale * 0.7 + vec3<f32>(time * flow * 0.6, seed, 0.0), detail),
        fbm(p * scale * 0.7 + vec3<f32>(0.0, time * flow * 0.5 + seed, 1.7), detail),
        fbm(p * scale * 0.7 + vec3<f32>(2.3, 0.0, time * flow * 0.45 + seed), detail)
    ) - 0.5;
    p = p + warp * (0.55 + turbulence * 0.7);

    var field = fbm(p * scale + vec3<f32>(0.0, time * flow * 0.35, time * flow * 0.2), detail);

    // Shape the field into a shell: bright near the surface, hollow in the
    // middle, exactly like the reference.
    let shell = orb.d.z;
    let core = orb.d.w;
    let surface = radius * (1.0 + wobble * (field - 0.5) * 1.2);
    let shell_mask = 1.0 - smoothstep(surface - shell * 0.55, surface, r);
    let hollow = smoothstep(0.0, max(core, 0.001) * surface, r);

    let body = shell_mask * hollow;
    return clamp(body * (0.35 + field * 1.25) * (0.6 + turbulence * 0.8), 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = orb.g.z;
    var uv = vec2<f32>(in.uv.x * aspect, in.uv.y);
    uv = uv - vec2<f32>(orb.e.y, orb.e.z);
    uv = rotate(uv, orb.e.w);

    // The body occupies half the window; the rest is room for the glow to
    // fade out inside, so nothing is ever clipped at the window edge — a clip
    // is a straight line, and a straight line is a visible boundary.
    let radius = orb.a.y * 0.44;
    let distortion = orb.e.x;
    let energy = orb.f.x;
    let onset = orb.g.x;

    // Silhouette distortion: a low-frequency angular ripple, so the outline
    // breathes without the body changing size.
    let angle = atan2(uv.y, uv.x);
    let ripple = sin(angle * 3.0 + orb.a.x * 0.8 + orb.g.y * 6.28) * 0.5
        + sin(angle * 5.0 - orb.a.x * 1.3) * 0.3;
    let edge = radius * (1.0 + distortion * ripple * 0.12);

    let dist = length(uv);
    var accum = 0.0;
    var weighted_depth = 0.0;

    // March only the span the sphere can occupy.
    if (dist < edge * 1.25) {
        let half_chord = sqrt(max(edge * edge * 1.55 - dist * dist, 0.0));
        let steps = 18;
        let step_size = (half_chord * 2.0) / f32(steps);
        var z = -half_chord;
        for (var i = 0; i < steps; i = i + 1) {
            let pos = vec3<f32>(uv.x, uv.y, z) / max(edge, 0.0001);
            let d = density(pos, 1.0);
            accum = accum + d * step_size * 2.6;
            weighted_depth = weighted_depth + d * (0.5 + 0.5 * (z / max(half_chord, 0.0001)));
            z = z + step_size;
        }
    }
    accum = clamp(accum, 0.0, 1.0);

    // Outer bloom, independent of the marched body so the glow survives even
    // when the body is thin. Tight on purpose: a wide bloom on a small orb
    // reads as a halo drawn around it rather than as light coming off it.
    let glow_amount = orb.c.x * (0.35 + energy * 0.4 + onset * 0.3);
    let bloom = exp(-max(dist - edge * 0.9, 0.0) * 16.0) * glow_amount;

    // Colour: hue travels from the core outward, so the entity has depth
    // rather than being one flat purple.
    let depth_t = clamp(weighted_depth * 0.35, 0.0, 1.0);
    let hue = orb.c.w + orb.d.x * (depth_t - 0.35) + orb.f.w * 6.0;
    let saturation = clamp(orb.d.y * (1.05 - depth_t * 0.25), 0.0, 1.0);
    let value = orb.c.y * (0.35 + accum * 0.9 + onset * 0.25);

    var colour = hsv(hue, saturation, value);
    // Rim light: the bright edge that reads as a membrane.
    let rim = smoothstep(edge * 1.02, edge * 0.72, dist) - smoothstep(edge * 0.72, edge * 0.2, dist);
    colour = colour + hsv(hue + orb.d.x * 0.6, saturation * 0.8, 1.0)
        * max(rim, 0.0) * (0.18 + energy * 0.35) * accum;
    colour = colour + hsv(hue - 8.0, saturation * 0.6, 1.0) * bloom * 0.55;

    var alpha = clamp(accum * 1.15 + bloom * 0.6, 0.0, 1.0) * orb.c.z;

    // The glow decays exponentially and never reaches zero, so a
    // window-sized rectangle of alpha 0.01 would be a visible square on a
    // dark desktop. Two previous attempts to remove it both drew a ring
    // instead: cutting at a fixed radius puts an edge at that radius, and
    // smoothstepping the alpha puts an edge wherever the alpha crosses the
    // threshold. Any threshold has a contour.
    //
    // So: no threshold. Cubing is monotonic and has no knee — 0.9 stays 0.73,
    // 0.05 becomes 0.000125, which is nothing — and a gaussian keeps the far
    // corners at exactly zero without a boundary of its own.
    alpha = alpha * alpha * alpha;
    alpha = alpha * exp(-dist * dist * 3.0);

    // Keep a whisper of the background tint when the window is opaque.
    alpha = max(alpha, orb.g.w);

    if (orb.h.x > 0.5) {
        return vec4<f32>(colour * alpha, alpha);
    }
    return vec4<f32>(colour, alpha);
}

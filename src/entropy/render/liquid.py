"""The liquid pass (SPEC 7.2): screen-space water from the liquid's particles (#52).

It reads the scene already drawn in linear light (an HDR target with its depth),
and writes the scene with the water over it into another HDR target, with the
water's depth, so later passes (glass) are hidden behind it.

1. **Depth.** Each particle is a sphere of `radius_m`, drawn as a point sprite in
   targets of half the output's size (ADR 0027). The nearest sphere surface per pixel
   gives the water's eye depth (m). A sphere is dropped where it is behind the scene's
   solids at every output pixel its sprite pixel covers.
2. **Thickness.** Each sphere adds its chord, 2·√(r² - d²), along the view ray.
   The sum is scaled by the particle's volume over its sphere's volume, so that in
   bulk water the thickness is the length of water crossed.
3. **Smoothing.** A narrow-range filter (Truong & Yuksel 2018): a separable
   Gaussian over the eye depth, in which neighbours far behind are skipped
   (another surface) and neighbours far in front are clamped. It turns the bumps
   of the spheres into a smooth surface while keeping silhouettes.
4. **Shading**, at full resolution. The depth and the normals (from the smoothed depth)
   are interpolated between the four nearest half-size pixels where they are one
   surface; at a silhouette, those of the nearest. Water behind an output pixel's solid
   is not drawn there. The surface reflects the sky
   by Fresnel's law (Schlick, F0 from the refractive index) and lets the rest
   through. What it lets through is the scene behind, shifted by refraction and
   absorbed by Beer-Lambert's law, exp(-a·thickness) per primary, a from the
   water's material file.

Simplifications, named (CLAUDE.md rule 7):
- spheres are drawn as camera-facing sprites (their depth is exact, their outline
  is the sprite's disc);
- refraction bends the view ray once at the surface (Snell's law) and follows it
  for the water's thickness; the image behind shifts by how far that lands from
  the straight ray. It is not traced further, and it never brings in what lies in
  front of the water;
- only the sky is reflected, not the scene;
- absorption uses one wavelength per sRGB primary (`water.toml`);
- a sprite is at least one pixel; a sphere smaller than that has its chords weighted by
  d² (d its diameter in pixels), so it adds its own volume on average and far water
  keeps its thickness on average, not pixel by pixel;
- glass behind the water is never seen through it (the water writes its depth);
- the spheres, their depth and its smoothing are drawn at half the output's resolution
  (ADR 0027): silhouettes are two output pixels coarse, detail under two output pixels
  is lost, and the water shimmers more as the camera turns (measured in ADR 0027);
- around a solid inside the water, the thickness blends across the solid's outline
  over one or two output pixels (ADR 0027);
- at the outline of water in front of other water, the smoothing pulls the far
  surface forward, by at most its passes times its range (the narrow-range filter);
- no foam, spray or caustics (later passes, SPEC 7.2).
"""

from __future__ import annotations

import math

import moderngl
import numpy as np

from entropy.render.camera import CameraView, gl_bytes
from entropy.render.frame import HdrTarget
from entropy.render.interop import SharedBuffer
from entropy.render.scene import HORIZON, SKY_GLSL, SUN_DIRECTION
from entropy.render.world_view import fresnel_f0
from entropy.world.materials import Material

__all__ = ["LiquidRenderer", "fresnel_f0"]

FAR_M = 1.0e6
"""Eye depth meaning "no water here"."""
SMOOTHING_RADIUS_PARTICLES = 4.0
"""The smoothing filter's radius, in particle radii."""
SMOOTHING_ITERATIONS = 3
SMOOTHING_RANGE_PARTICLES = 8.0
"""Neighbours further than this (in particle radii) in depth are another surface."""
MAX_FILTER_PX = 24
"""The smoothing filter's largest radius, in pixels of the half-size targets (48 output
pixels): water nearer than about 1.3 m at 1600 x 900 is smoothed less than its radius
asks (ADR 0027)."""
SPRITE_DOWNSCALE = 2
"""The spheres are drawn at 1/2 of the output's resolution (#57): their cost is per
fragment, and a million of them seen up close took 4.6 ms at 1600 x 900."""

_SPRITE_VERTEX = """
#version 450 core
uniform mat4 view;
uniform mat4 projection;
uniform float radius_m;
uniform float viewport_height_px;
in vec3 position_m;
out vec3 centre_view;
out float coverage;
void main() {
    vec4 v = view * vec4(position_m, 1.0);
    centre_view = v.xyz;
    gl_Position = projection * v;
    float diameter_px = radius_m * projection[1][1] * viewport_height_px / max(-v.z, 1e-4);
    gl_PointSize = max(diameter_px, 1.0);
    // A sphere smaller than its one-pixel sprite: the sprite still integrates a disc's
    // chords, but over a one-pixel disc, d^-2 times the sphere's own area (d its true
    // diameter in pixels). Weighted by d^2, it adds the sphere's volume, on average.
    coverage = min(1.0, diameter_px * diameter_px);
}
"""

_SPRITE_COMMON = """
uniform mat4 projection;
uniform float radius_m;
uniform sampler2D scene_depth;
uniform vec2 scene_per_sprite_px;  // scene pixels per sprite-target pixel (about 2)
in vec3 centre_view;
in float coverage;

// The sphere's point under this fragment, in view space; false outside the disc.
bool sphere_point(out vec3 p, out float nz) {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    c.y = -c.y;
    float r2 = dot(c, c);
    if (r2 > 1.0) return false;
    nz = sqrt(1.0 - r2);
    p = centre_view + radius_m * vec3(c, nz);
    return true;
}

float window_depth(vec3 p) {
    vec4 clip = projection * vec4(p, 1.0);
    return clip.z / clip.w * 0.5 + 0.5;
}

// Behind every scene pixel this sprite-target pixel covers: the farthest of the 2 x 2
// around its centre. Kept where any of them sees it; the composite tests each pixel.
bool behind_scene(float depth) {
    vec2 uv = gl_FragCoord.xy * scene_per_sprite_px / vec2(textureSize(scene_depth, 0));
    vec4 d = textureGather(scene_depth, uv, 0);
    return depth >= max(max(d.x, d.y), max(d.z, d.w));
}
"""

_DEPTH_FRAGMENT = (
    "#version 450 core\n"
    + _SPRITE_COMMON
    + """
out float eye_depth;
void main() {
    vec3 p;
    float nz;
    if (!sphere_point(p, nz)) discard;
    float depth = window_depth(p);
    if (behind_scene(depth)) discard;
    gl_FragDepth = depth;
    eye_depth = -p.z;
}
"""
)

_THICKNESS_FRAGMENT = (
    "#version 450 core\n"
    + _SPRITE_COMMON
    + """
uniform float thickness_scale;
out float thickness;
void main() {
    vec3 p;
    float nz;
    if (!sphere_point(p, nz)) discard;
    if (behind_scene(window_depth(p))) discard;
    thickness = 2.0 * nz * radius_m * thickness_scale * coverage;
}
"""
)

_FULL_SCREEN = """
#version 450 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_SMOOTH_FRAGMENT = f"""
#version 450 core
uniform sampler2D depth_in;
uniform ivec2 direction;
uniform float projection_y;
uniform float viewport_height_px;
uniform float filter_radius_m;
uniform float range_m;
out float depth_out;
const float FAR = {FAR_M:.1f};
void main() {{
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 last = textureSize(depth_in, 0) - 1;
    float z = texelFetch(depth_in, p, 0).r;
    if (z >= FAR) {{ depth_out = z; return; }}
    float radius_px = filter_radius_m * projection_y * 0.5 * viewport_height_px / z;
    radius_px = clamp(radius_px, 1.0, {MAX_FILTER_PX:.1f});
    float sigma = radius_px / 2.0;
    int n = int(ceil(radius_px));
    float sum = 0.0;
    float weights = 0.0;
    for (int i = -n; i <= n; ++i) {{
        float zq = texelFetch(depth_in, clamp(p + direction * i, ivec2(0), last), 0).r;
        if (zq >= FAR || zq > z + range_m) continue;
        zq = max(zq, z - range_m);
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += w * zq;
        weights += w;
    }}
    depth_out = sum / weights;
}}
"""

_COMPOSITE_FRAGMENT = (
    f"""
#version 450 core
const float FAR = {FAR_M:.1f};
"""
    + SKY_GLSL
    + """
uniform sampler2D scene_colour;
uniform sampler2D scene_depth;
uniform sampler2D fluid_depth;
uniform sampler2D fluid_thickness;
uniform mat4 projection;
uniform mat3 view_to_world;
uniform float f0;
uniform float eta;
uniform vec3 absorption_per_m;
uniform float range_m;
out vec4 colour;

vec2 size_px() { return vec2(textureSize(fluid_depth, 0)); }  // the water's targets
vec2 scene_px() { return vec2(textureSize(scene_colour, 0)); }

vec3 view_position(ivec2 p, float z) {
    vec2 ndc = (vec2(p) + 0.5) / size_px() * 2.0 - 1.0;
    return vec3(ndc.x * z / projection[0][0], ndc.y * z / projection[1][1], -z);
}

// The neighbour's view position along `step`, or the opposite one, whichever is
// nearer in depth: a silhouette does not bend the normal.
vec3 tangent(ivec2 p, vec3 centre, ivec2 step) {
    ivec2 last = textureSize(fluid_depth, 0) - 1;
    ivec2 a = clamp(p + step, ivec2(0), last);
    ivec2 b = clamp(p - step, ivec2(0), last);
    float za = texelFetch(fluid_depth, a, 0).r;
    float zb = texelFetch(fluid_depth, b, 0).r;
    vec3 forward = za < FAR ? view_position(a, za) - centre : vec3(0.0);
    vec3 backward = zb < FAR ? centre - view_position(b, zb) : vec3(0.0);
    if (za >= FAR) return backward;
    if (zb >= FAR) return forward;
    return abs(forward.z) < abs(backward.z) ? forward : backward;
}

float window_depth(vec3 p) {
    vec4 clip = projection * vec4(p, 1.0);
    return clip.z / clip.w * 0.5 + 0.5;
}

// The surface's normal at water pixel p (depth z), in view space; unoriented.
vec3 normal_at(ivec2 p, float z) {
    vec3 centre = view_position(p, z);
    vec3 n = cross(tangent(p, centre, ivec2(1, 0)), tangent(p, centre, ivec2(0, 1)));
    return dot(n, n) > 0.0 ? normalize(n) : vec3(0.0, 0.0, 1.0);
}

float eye_depth(float window) {
    return projection[3][2] / ((2.0 * window - 1.0) + projection[2][2]);
}

void main() {
    ivec2 s = ivec2(gl_FragCoord.xy);  // the scene's pixel
    // Where this scene pixel's centre falls among the water's pixels (centres at integers).
    vec2 f = (vec2(s) + 0.5) * size_px() / scene_px() - 0.5;
    ivec2 last = ivec2(size_px()) - 1;
    ivec2 p = clamp(ivec2(floor(f + 0.5)), ivec2(0), last);
    float behind_depth = texelFetch(scene_depth, s, 0).r;
    vec3 behind = texelFetch(scene_colour, s, 0).rgb;
    float z = texelFetch(fluid_depth, p, 0).r;
    // Depth and normal interpolated between the four nearest water pixels when they are
    // one surface; otherwise (a silhouette) those of the nearest.
    ivec2 a = clamp(ivec2(floor(f)), ivec2(0), last);
    ivec2 b = min(a + 1, last);
    vec2 t = clamp(f - vec2(a), 0.0, 1.0);
    float z00 = texelFetch(fluid_depth, a, 0).r;
    float z10 = texelFetch(fluid_depth, ivec2(b.x, a.y), 0).r;
    float z01 = texelFetch(fluid_depth, ivec2(a.x, b.y), 0).r;
    float z11 = texelFetch(fluid_depth, b, 0).r;
    float near = min(min(z00, z10), min(z01, z11));
    float far = max(max(z00, z10), max(z01, z11));
    vec3 n;
    if (far < FAR && far - near <= range_m) {
        z = mix(mix(z00, z10, t.x), mix(z01, z11, t.x), t.y);
        n = mix(
            mix(normal_at(a, z00), normal_at(ivec2(b.x, a.y), z10), t.x),
            mix(normal_at(ivec2(a.x, b.y), z01), normal_at(b, z11), t.x),
            t.y
        );
        n = dot(n, n) > 0.0 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    } else {
        n = z < FAR ? normal_at(p, z) : vec3(0.0, 0.0, 1.0);
    }
    // No water here, or the water lies behind this pixel's solid (the spheres were kept
    // where any of a block's pixels sees them). No allowance: none was visibly needed,
    // and one of a few sphere radii lets water show over a solid just in front of it.
    if (z >= FAR || z > eye_depth(behind_depth)) {
        colour = vec4(behind, 1.0);
        gl_FragDepth = behind_depth;
        return;
    }
    vec2 ndc = (vec2(s) + 0.5) / scene_px() * 2.0 - 1.0;
    vec3 centre = vec3(ndc.x * z / projection[0][0], ndc.y * z / projection[1][1], -z);
    vec3 v = normalize(-centre);
    n = dot(n, v) < 0.0 ? -n : n;
    float c = clamp(dot(n, v), 0.0, 1.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - c, 5.0);
    vec3 reflected = sky(normalize(view_to_world * reflect(-v, n)));

    float thickness = texture(fluid_thickness, (vec2(s) + 0.5) / scene_px()).r;
    float depth = window_depth(centre);
    vec3 straight = -v;
    vec3 bent = refract(straight, n, eta);
    vec2 shift_m = (bent - straight).xy * thickness;
    vec2 shift_px = shift_m * projection[1][1] * 0.5 * scene_px().y / z;
    vec2 at = vec2(s) + 0.5 + shift_px;
    ivec2 q = clamp(ivec2(at), ivec2(0), textureSize(scene_colour, 0) - 1);
    if (texelFetch(scene_depth, q, 0).r < depth) {
        at = vec2(s) + 0.5;  // never refract what is in front
    }
    vec3 behind_bent = texture(scene_colour, at / scene_px()).rgb;
    vec3 transmitted = behind_bent * exp(-absorption_per_m * thickness);

    colour = vec4(fresnel * reflected + (1.0 - fresnel) * transmitted, 1.0);
    gl_FragDepth = depth;
}
"""
)


class LiquidRenderer:
    """Draws the liquid's particles from a SharedBuffer as water (see the module doc)."""

    def __init__(
        self,
        ctx: moderngl.Context,
        particles: SharedBuffer,
        *,
        radius_m: float,
        particle_volume_m3: float,
        water: Material,
    ) -> None:
        if water.refractive_index is None or water.absorption_per_m_rgb is None:
            raise ValueError(f"{water.name} needs a refractive index and an absorption")
        if not (radius_m > 0 and particle_volume_m3 > 0):
            raise ValueError("radius_m and particle_volume_m3 must be positive")
        self.ctx = ctx
        self.particles = particles
        self.radius_m = radius_m
        self.refractive_index = water.refractive_index
        self.absorption_per_m_rgb = water.absorption_per_m_rgb
        self.thickness_scale = particle_volume_m3 / (4.0 / 3.0 * math.pi * radius_m**3)
        self._depth = ctx.program(vertex_shader=_SPRITE_VERTEX, fragment_shader=_DEPTH_FRAGMENT)
        self._thickness = ctx.program(
            vertex_shader=_SPRITE_VERTEX, fragment_shader=_THICKNESS_FRAGMENT
        )
        self._smooth = ctx.program(vertex_shader=_FULL_SCREEN, fragment_shader=_SMOOTH_FRAGMENT)
        self._composite = ctx.program(
            vertex_shader=_FULL_SCREEN, fragment_shader=_COMPOSITE_FRAGMENT
        )
        self._depth_vao = ctx.vertex_array(self._depth, [(particles.buffer, "3f", "position_m")])
        self._thickness_vao = ctx.vertex_array(
            self._thickness, [(particles.buffer, "3f", "position_m")]
        )
        self._smooth_vao = ctx.vertex_array(self._smooth, [])
        self._composite_vao = ctx.vertex_array(self._composite, [])
        for program in (self._depth, self._thickness):
            program["radius_m"].value = radius_m  # type: ignore[union-attr]
            program["scene_depth"].value = 0  # type: ignore[union-attr]
        self._thickness["thickness_scale"].value = self.thickness_scale  # type: ignore[union-attr]
        self._smooth["depth_in"].value = 0  # type: ignore[union-attr]
        composite = self._composite
        for name, unit in (
            ("scene_colour", 0),
            ("scene_depth", 1),
            ("fluid_depth", 2),
            ("fluid_thickness", 3),
        ):
            composite[name].value = unit  # type: ignore[union-attr]
        composite["sun_direction"].value = SUN_DIRECTION  # type: ignore[union-attr]
        composite["horizon"].value = HORIZON  # type: ignore[union-attr]
        composite["f0"].value = fresnel_f0(self.refractive_index)  # type: ignore[union-attr]
        composite["eta"].value = 1.0 / self.refractive_index  # type: ignore[union-attr]
        composite["absorption_per_m"].value = self.absorption_per_m_rgb  # type: ignore[union-attr]
        composite["range_m"].value = SMOOTHING_RANGE_PARTICLES * radius_m  # type: ignore[union-attr]
        self._size: tuple[int, int] | None = None
        self.sprite_size_px: tuple[int, int] | None = None
        """The size of the targets the spheres are drawn in: half the output's."""
        self._textures: list[moderngl.Texture] = []
        self._framebuffers: list[moderngl.Framebuffer] = []
        self._depth_buffer: moderngl.Renderbuffer | None = None

    def _resize(self, size: tuple[int, int]) -> None:
        if size == self._size:
            return
        self._release_targets()
        ctx = self.ctx
        small = (-(-size[0] // SPRITE_DOWNSCALE), -(-size[1] // SPRITE_DOWNSCALE))  # rounded up
        self._textures = [ctx.texture(small, 1, dtype="f4") for _ in range(3)]
        for texture in self._textures:
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._textures[2].filter = (moderngl.LINEAR, moderngl.LINEAR)  # thickness: smooth
        self._depth_buffer = ctx.depth_renderbuffer(small)
        self.sprite_size_px = small
        depth_a, depth_b, thickness = self._textures
        self._framebuffers = [
            ctx.framebuffer([depth_a], self._depth_buffer),
            ctx.framebuffer([depth_b]),
            ctx.framebuffer([thickness]),
            ctx.framebuffer([depth_a]),
        ]
        self._size = size

    def render(self, out: HdrTarget, scene: HdrTarget, view: CameraView, *, count: int) -> None:
        """Draw the first `count` particles as water over `scene` into `out`."""
        if out.size_px != scene.size_px:
            raise ValueError(f"out is {out.size_px}, scene is {scene.size_px}")
        if not 0 <= count <= self.particles.count:
            raise ValueError(f"count must be between 0 and {self.particles.count}, got {count}")
        width, height = size = scene.size_px
        if width <= 0 or height <= 0:
            return
        self._resize(size)
        ctx = self.ctx
        assert self.sprite_size_px is not None
        small_w, small_h = self.sprite_size_px
        # Each target's `use()` sets its own viewport; this pass sets none (#78's review).
        projection = view.projection(width / height).astype(np.float64)
        camera = view.view().astype(np.float64)
        depth_a, _, thickness = self._textures
        sprites_fb, _, thickness_fb, _ = self._framebuffers
        scene.depth.compare_func = ""
        sprites_fb.clear(FAR_M, 0.0, 0.0, 0.0, depth=1.0)
        thickness_fb.clear(0.0, 0.0, 0.0, 0.0)
        if count:
            for program in (self._depth, self._thickness):
                program["view"].write(gl_bytes(camera))  # type: ignore[union-attr]
                program["projection"].write(gl_bytes(projection))  # type: ignore[union-attr]
                program["viewport_height_px"].value = float(small_h)  # type: ignore[union-attr]
                ratio = (width / small_w, height / small_h)
                program["scene_per_sprite_px"].value = ratio  # type: ignore[union-attr]
            scene.depth.use(location=0)
            ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST)
            sprites_fb.use()
            self._depth_vao.render(moderngl.POINTS, vertices=count)
            ctx.disable(moderngl.DEPTH_TEST)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            thickness_fb.use()
            self._thickness_vao.render(moderngl.POINTS, vertices=count)
            ctx.disable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
            self._smooth_depth(projection, small_h)

        composite = self._composite
        composite["projection"].write(gl_bytes(projection))  # type: ignore[union-attr]
        rotation = np.ascontiguousarray(camera[:3, :3], dtype=np.float32)  # view_to_world = Rᵀ
        composite["view_to_world"].write(rotation.tobytes())  # type: ignore[union-attr]
        scene.colour.use(location=0)
        scene.depth.use(location=1)
        depth_a.use(location=2)
        thickness.use(location=3)
        out.framebuffer.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "1"  # always: the pass writes every pixel's depth
        try:
            self._composite_vao.render(moderngl.TRIANGLES, vertices=3)
        finally:
            ctx.depth_func = "<"
            ctx.disable(moderngl.DEPTH_TEST)

    def _smooth_depth(self, projection: np.ndarray, height: int) -> None:
        depth_a, depth_b, _ = self._textures
        _, smooth_b_fb, _, smooth_a_fb = self._framebuffers
        program = self._smooth
        program["projection_y"].value = float(projection[1, 1])  # type: ignore[union-attr]
        program["viewport_height_px"].value = float(height)  # type: ignore[union-attr]
        program["filter_radius_m"].value = SMOOTHING_RADIUS_PARTICLES * self.radius_m  # type: ignore[union-attr]
        program["range_m"].value = SMOOTHING_RANGE_PARTICLES * self.radius_m  # type: ignore[union-attr]
        for _ in range(SMOOTHING_ITERATIONS):
            for source, target, direction in (
                (depth_a, smooth_b_fb, (1, 0)),
                (depth_b, smooth_a_fb, (0, 1)),
            ):
                program["direction"].value = direction  # type: ignore[union-attr]
                source.use(location=0)
                target.use()
                self._smooth_vao.render(moderngl.TRIANGLES, vertices=3)

    def _release_targets(self) -> None:
        for resource in (*self._framebuffers, *self._textures):
            resource.release()
        if self._depth_buffer is not None:
            self._depth_buffer.release()
        self._framebuffers, self._textures, self._depth_buffer = [], [], None

    def release(self) -> None:
        self._release_targets()
        for resource in (
            self._depth_vao,
            self._thickness_vao,
            self._smooth_vao,
            self._composite_vao,
            self._depth,
            self._thickness,
            self._smooth,
            self._composite,
        ):
            resource.release()

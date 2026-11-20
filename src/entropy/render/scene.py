"""The empty lit scene of M0: sky, ground with a metre grid, sun, reference cubes.

Lighting is computed in linear RGB: Lambert diffuse from one directional sun
plus a sky term that is brighter from above (a hemisphere approximation).
The result is scaled by an exposure, tone-mapped with the ACES fit of
Narkowicz (2015) and encoded as sRGB. Values are relative (the sun irradiance
is 3, the sky 0.6); physically calibrated light units come with the HDR
pipeline of SPEC 7.2. Shaders are GLSL 4.50 (ADR 0009).
"""

from __future__ import annotations

import moderngl
import numpy as np

from entropy.render.camera import CameraView, gl_bytes

SUN_DIRECTION = tuple(float(c) for c in np.array([0.4, 0.3, 0.8]) / np.linalg.norm([0.4, 0.3, 0.8]))
"""Unit vector towards the sun: high in the sky, from +X and +Y."""

GROUND_HALF_SIZE_M = 500.0

REFERENCE_CUBES = (
    # (centre of the base in m, edge in m, linear albedo)
    ((0.0, 0.0, 0.0), 1.0, (0.60, 0.60, 0.60)),
    ((3.0, 2.0, 0.0), 1.0, (0.65, 0.20, 0.15)),
    ((3.0, 2.0, 1.0), 1.0, (0.65, 0.20, 0.15)),
    ((-2.0, 3.0, 0.0), 1.0, (0.15, 0.25, 0.60)),
    ((-4.0, -3.0, 0.0), 2.0, (0.50, 0.50, 0.45)),
)
"""Cubes of known size, so scale and lighting can be judged."""


def srgb_from_linear(value: float) -> float:
    """The sRGB transfer function (IEC 61966-2-1), linear [0, 1] to encoded [0, 1]."""
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


_DISPLAY = """
uniform float exposure;

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 srgb(vec3 c) {
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

vec3 display(vec3 radiance) {
    return srgb(aces(radiance * exposure));
}
"""

_SKY = """
uniform vec3 sun_direction;

vec3 sky(vec3 d) {
    vec3 zenith = vec3(0.10, 0.25, 0.70);
    vec3 horizon = vec3(0.60, 0.72, 0.90);
    vec3 colour = mix(horizon, zenith, pow(clamp(d.z, 0.0, 1.0), 0.5));
    if (d.z < 0.0) {
        colour = mix(horizon * 0.7, vec3(0.25, 0.24, 0.22), clamp(-d.z * 4.0, 0.0, 1.0));
    }
    float sun = smoothstep(0.9995, 0.99985, dot(d, sun_direction));
    return colour + sun * vec3(30.0, 28.0, 24.0);
}
"""

_LIGHTING = """
uniform vec3 sun_direction;
uniform vec3 sun_irradiance;
uniform vec3 sky_irradiance;

vec3 lit(vec3 albedo, vec3 normal) {
    float direct = max(dot(normal, sun_direction), 0.0);
    float hemisphere = 0.5 + 0.5 * normal.z;
    return albedo * (sun_irradiance * direct + sky_irradiance * hemisphere);
}
"""

_SKY_PROGRAM = (
    """
#version 450 core
uniform mat4 inverse_view_projection;
out vec3 ray;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    vec4 near = inverse_view_projection * vec4(p, -1.0, 1.0);
    vec4 far = inverse_view_projection * vec4(p, 1.0, 1.0);
    ray = far.xyz / far.w - near.xyz / near.w;
    gl_Position = vec4(p, 1.0, 1.0);
}
""",
    "#version 450 core\n"
    + _DISPLAY
    + _SKY
    + """
in vec3 ray;
out vec4 colour;
void main() {
    colour = vec4(display(sky(normalize(ray))), 1.0);
}
""",
)

_GROUND_PROGRAM = (
    """
#version 450 core
uniform mat4 view_projection;
in vec3 position_m;
out vec3 world_m;
void main() {
    world_m = position_m;
    gl_Position = view_projection * vec4(position_m, 1.0);
}
""",
    "#version 450 core\n"
    + _DISPLAY
    + _LIGHTING
    + """
uniform vec3 eye_m;
in vec3 world_m;
out vec4 colour;

float grid(vec2 p, float spacing) {
    vec2 q = p / spacing;
    vec2 distance = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(distance.x, distance.y), 1.0);
}

void main() {
    float lines = max(0.5 * grid(world_m.xy, 1.0), grid(world_m.xy, 10.0));
    vec3 albedo = mix(vec3(0.22), vec3(0.08), lines);
    vec3 radiance = lit(albedo, vec3(0.0, 0.0, 1.0));
    float haze = 1.0 - exp(-length(world_m - eye_m) / 300.0);
    radiance = mix(radiance, vec3(0.60, 0.72, 0.90), haze);
    colour = vec4(display(radiance), 1.0);
}
""",
)

_CUBE_PROGRAM = (
    """
#version 450 core
uniform mat4 view_projection;
uniform vec3 base_m;
uniform float edge_m;
in vec3 corner;
in vec3 normal;
out vec3 face_normal;
void main() {
    face_normal = normal;
    gl_Position = view_projection * vec4(base_m + corner * edge_m, 1.0);
}
""",
    "#version 450 core\n"
    + _DISPLAY
    + _LIGHTING
    + """
uniform vec3 albedo;
in vec3 face_normal;
out vec4 colour;
void main() {
    colour = vec4(display(lit(albedo, normalize(face_normal))), 1.0);
}
""",
)


def _cube_vertices() -> np.ndarray:
    """36 vertices (corner, normal) of a unit cube standing on z = 0, counter-clockwise."""
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            u, v = np.roll(np.eye(3), -axis - 1, axis=0)[:2]
            if sign < 0:
                u, v = v, u
            centre = np.array([0.0, 0.0, 0.5]) + 0.5 * normal
            quad = [centre + 0.5 * (a * u + b * v) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for index in (0, 1, 2, 0, 2, 3):
                faces.append(np.concatenate([quad[index], normal]))
    return np.array(faces, dtype=np.float32)


class SceneRenderer:
    """Draws the M0 scene into a framebuffer for a camera view."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.exposure = 0.5
        self._sky = ctx.program(vertex_shader=_SKY_PROGRAM[0], fragment_shader=_SKY_PROGRAM[1])
        self._ground = ctx.program(
            vertex_shader=_GROUND_PROGRAM[0], fragment_shader=_GROUND_PROGRAM[1]
        )
        self._cube = ctx.program(vertex_shader=_CUBE_PROGRAM[0], fragment_shader=_CUBE_PROGRAM[1])
        self._sky_vao = ctx.vertex_array(self._sky, [])

        h = GROUND_HALF_SIZE_M
        ground = np.array(
            [[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, -h, 0], [h, h, 0], [-h, h, 0]], dtype="f4"
        )
        self._ground_buffer = ctx.buffer(ground.tobytes())
        self._ground_vao = ctx.vertex_array(
            self._ground, [(self._ground_buffer, "3f", "position_m")]
        )
        self._cube_buffer = ctx.buffer(_cube_vertices().tobytes())
        self._cube_vao = ctx.vertex_array(
            self._cube, [(self._cube_buffer, "3f 3f", "corner", "normal")]
        )
        for program in (self._sky, self._ground, self._cube):
            program["sun_direction"].value = SUN_DIRECTION  # type: ignore[union-attr]
            program["exposure"].value = self.exposure  # type: ignore[union-attr]
        for program in (self._ground, self._cube):
            program["sun_irradiance"].value = (3.0, 2.9, 2.7)  # type: ignore[union-attr]
            program["sky_irradiance"].value = (0.45, 0.55, 0.75)  # type: ignore[union-attr]

    def render(self, target: moderngl.Framebuffer, view: CameraView) -> None:
        ctx = self.ctx
        target.use()
        width, height = target.size
        ctx.viewport = (0, 0, width, height)
        view_projection = view.projection(width / height) @ view.view()
        matrix = gl_bytes(view_projection)

        target.clear(depth=1.0)
        # The sky covers the whole frame; with the depth test off it writes no depth.
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._sky["inverse_view_projection"].write(  # type: ignore[union-attr]
            gl_bytes(np.linalg.inv(view_projection.astype(np.float64)).astype(np.float32))
        )
        self._sky_vao.render(moderngl.TRIANGLES, vertices=3)

        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._ground["view_projection"].write(matrix)  # type: ignore[union-attr]
        self._ground["eye_m"].value = view.position_m  # type: ignore[union-attr]
        self._ground_vao.render(moderngl.TRIANGLES)

        self._cube["view_projection"].write(matrix)  # type: ignore[union-attr]
        for base_m, edge_m, albedo in REFERENCE_CUBES:
            self._cube["base_m"].value = base_m  # type: ignore[union-attr]
            self._cube["edge_m"].value = edge_m  # type: ignore[union-attr]
            self._cube["albedo"].value = albedo  # type: ignore[union-attr]
            self._cube_vao.render(moderngl.TRIANGLES)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def release(self) -> None:
        for resource in (
            self._sky_vao,
            self._ground_vao,
            self._cube_vao,
            self._ground_buffer,
            self._cube_buffer,
            self._sky,
            self._ground,
            self._cube,
        ):
            resource.release()

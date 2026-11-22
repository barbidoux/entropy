"""The lit scene: sky, ground with a metre grid, sun, and opaque boxes.

M0 drew five reference cubes (the default); a world gives its own boxes and
ground height (`entropy.render.world_view`, #51).

A simple placeholder renderer until the passes of SPEC 7.2. Lighting is
computed in linear RGB: Lambert diffuse from one directional sun plus a sky
term that is brighter from above (a hemisphere approximation). It draws linear
radiance into an HDR target; `entropy.render.frame` tone-maps it for the display.
Shaders are GLSL 4.50 (ADR 0009).

Simplifications, named (CLAUDE.md rule 7):
- light values are relative, not calibrated physical units (SPEC 7.2 brings HDR);
- no shadows, and no light bounced from the ground, so faces turned down are black;
- distance haze: exponential over 300 m towards the horizon colour;
- the sun disk is drawn about 1° in radius (the real sun: 0.27°), to be visible;
- the ground is a one-sided 20 km square; seen from below it is invisible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import moderngl
import numpy as np

from entropy.render.camera import CameraView, gl_bytes

SUN_DIRECTION = tuple(float(c) for c in np.array([0.4, 0.3, 0.8]) / np.linalg.norm([0.4, 0.3, 0.8]))
"""Unit vector towards the sun: high in the sky, from +X and +Y."""
SUN_IRRADIANCE = (3.0, 2.9, 2.7)
"""Relative irradiance from the sun on a surface facing it, linear RGB."""
SKY_IRRADIANCE = (0.45, 0.55, 0.75)
"""Relative irradiance from the sky on a surface facing up, linear RGB."""
HORIZON = (0.60, 0.72, 0.90)
"""Sky colour at the horizon, also the colour distant ground fades to."""

GROUND_HALF_SIZE_M = 10_000.0

REFERENCE_CUBES = (
    # (centre of the base in m, edge in m, linear albedo)
    ((0.0, 0.0, 0.0), 1.0, (0.60, 0.60, 0.60)),
    ((3.0, 2.0, 0.0), 1.0, (0.65, 0.20, 0.15)),
    ((3.0, 2.0, 1.0), 1.0, (0.65, 0.20, 0.15)),
    ((-2.0, 3.0, 0.0), 1.0, (0.15, 0.25, 0.60)),
    ((-4.0, -3.0, 0.0), 2.0, (0.50, 0.50, 0.45)),
)
"""Cubes of known size, so scale and lighting can be judged."""


@dataclass(frozen=True)
class SceneBox:
    """An opaque axis-aligned box: centre and half sizes in metres, linear albedo."""

    centre_m: tuple[float, float, float]
    half_size_m: tuple[float, float, float]
    albedo_linear_rgb: tuple[float, float, float]
    grid_m: float = 0.0
    """Spacing of grid lines drawn on its top face (m); 0 draws none."""


M0_BOXES = tuple(
    SceneBox(
        (base[0], base[1], base[2] + edge / 2),
        (edge / 2, edge / 2, edge / 2),
        albedo,
    )
    for base, edge, albedo in REFERENCE_CUBES
)
"""The M0 scene's reference cubes as boxes."""


SKY_GLSL = """
uniform vec3 sun_direction;
uniform vec3 horizon;

vec3 sky(vec3 d) {
    vec3 zenith = vec3(0.10, 0.25, 0.70);
    // Above the horizon a gradient; below it, the hazed ground far away: continuous at 0.
    vec3 colour = mix(horizon, zenith, pow(clamp(d.z, 0.0, 1.0), 0.5));
    float sun = smoothstep(0.9995, 0.99985, dot(d, sun_direction));
    return colour + sun * vec3(30.0, 28.0, 24.0);
}
"""

LIGHTING_GLSL = """
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
uniform mat4 inverse_rotation_projection;
out vec3 ray;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    // Inverse of projection x rotation only: rays do not depend on where the camera is.
    vec4 far = inverse_rotation_projection * vec4(p, 1.0, 1.0);
    ray = far.xyz / far.w;
    gl_Position = vec4(p, 1.0, 1.0);
}
""",
    "#version 450 core\n"
    + SKY_GLSL
    + """
in vec3 ray;
out vec4 colour;
void main() {
    colour = vec4(sky(normalize(ray)), 1.0);
}
""",
)

GRID_GLSL = """
float grid(vec2 p, float spacing) {
    vec2 q = p / spacing;
    vec2 width = fwidth(q);
    vec2 distance = abs(fract(q - 0.5) - 0.5) / width;
    float line = 1.0 - min(min(distance.x, distance.y), 1.0);
    // Fade lines out where they get thinner than a pixel can show, to avoid moire.
    return line * (1.0 - smoothstep(0.1, 0.5, max(width.x, width.y)));
}
"""

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
    + LIGHTING_GLSL
    + GRID_GLSL
    + """
uniform vec3 eye_m;
uniform vec3 horizon;
in vec3 world_m;
out vec4 colour;

void main() {
    float lines = max(0.5 * grid(world_m.xy, 1.0), grid(world_m.xy, 10.0));
    vec3 albedo = mix(vec3(0.22), vec3(0.08), lines);
    vec3 radiance = lit(albedo, vec3(0.0, 0.0, 1.0));
    float haze = 1.0 - exp(-length(world_m - eye_m) / 300.0);
    radiance = mix(radiance, horizon, haze);
    colour = vec4(radiance, 1.0);
}
""",
)

_CUBE_PROGRAM = (
    """
#version 450 core
uniform mat4 view_projection;
uniform vec3 base_m;
uniform vec3 size_m;
in vec3 corner;
in vec3 normal;
out vec3 face_normal;
out vec3 world_m;
void main() {
    face_normal = normal;
    world_m = base_m + corner * size_m;
    gl_Position = view_projection * vec4(world_m, 1.0);
}
""",
    "#version 450 core\n"
    + LIGHTING_GLSL
    + GRID_GLSL
    + """
uniform vec3 albedo;
uniform float grid_m;
in vec3 face_normal;
in vec3 world_m;
out vec4 colour;
void main() {
    vec3 n = normalize(face_normal);
    float lines = grid_m > 0.0 && n.z > 0.5 ? grid(world_m.xy, grid_m) : 0.0;
    colour = vec4(lit(mix(albedo, 0.35 * albedo, lines), n), 1.0);
}
""",
)


def cube_vertices() -> np.ndarray:
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
    """Draws the sky, the ground at `ground_z_m` and `boxes` for a camera view."""

    def __init__(
        self,
        ctx: moderngl.Context,
        boxes: Sequence[SceneBox] = M0_BOXES,
        *,
        ground_z_m: float = 0.0,
    ) -> None:
        self.ctx = ctx
        self.boxes = tuple(boxes)
        self._sky = ctx.program(vertex_shader=_SKY_PROGRAM[0], fragment_shader=_SKY_PROGRAM[1])
        self._ground = ctx.program(
            vertex_shader=_GROUND_PROGRAM[0], fragment_shader=_GROUND_PROGRAM[1]
        )
        self._cube = ctx.program(vertex_shader=_CUBE_PROGRAM[0], fragment_shader=_CUBE_PROGRAM[1])
        self._sky_vao = ctx.vertex_array(self._sky, [])

        h, z = GROUND_HALF_SIZE_M, ground_z_m
        ground = np.array(
            [[-h, -h, z], [h, -h, z], [h, h, z], [-h, -h, z], [h, h, z], [-h, h, z]], dtype="f4"
        )
        self._ground_buffer = ctx.buffer(ground.tobytes())
        self._ground_vao = ctx.vertex_array(
            self._ground, [(self._ground_buffer, "3f", "position_m")]
        )
        self._cube_buffer = ctx.buffer(cube_vertices().tobytes())
        self._cube_vao = ctx.vertex_array(
            self._cube, [(self._cube_buffer, "3f 3f", "corner", "normal")]
        )
        for program in (self._sky, self._ground, self._cube):
            program["sun_direction"].value = SUN_DIRECTION  # type: ignore[union-attr]
        for program in (self._sky, self._ground):
            program["horizon"].value = HORIZON  # type: ignore[union-attr]
        for program in (self._ground, self._cube):
            program["sun_irradiance"].value = SUN_IRRADIANCE  # type: ignore[union-attr]
            program["sky_irradiance"].value = SKY_IRRADIANCE  # type: ignore[union-attr]

    def render(
        self,
        target: moderngl.Framebuffer,
        view: CameraView,
        size_px: tuple[int, int] | None = None,
    ) -> None:
        """Draw into `target` over `size_px` (default: the target's size).

        Pass the window's framebuffer size for the screen: moderngl keeps the
        screen's size from when the context was created. A zero size (a
        minimised window) draws nothing.
        """
        width, height = size_px or target.size
        if width <= 0 or height <= 0:
            return
        ctx = self.ctx
        target.use()
        ctx.viewport = (0, 0, width, height)
        projection = view.projection(width / height).astype(np.float64)
        camera = view.view().astype(np.float64)
        rotation = camera.copy()
        rotation[:3, 3] = 0.0
        matrix = gl_bytes(projection @ camera)

        target.clear(depth=1.0, viewport=(0, 0, width, height))
        # The sky covers the whole frame; with the depth test off it writes no depth.
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._sky["inverse_rotation_projection"].write(  # type: ignore[union-attr]
            gl_bytes(np.linalg.inv(projection @ rotation))
        )
        self._sky_vao.render(moderngl.TRIANGLES, vertices=3)

        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self._ground["view_projection"].write(matrix)  # type: ignore[union-attr]
        self._ground["eye_m"].value = view.position_m  # type: ignore[union-attr]
        self._ground_vao.render(moderngl.TRIANGLES)

        self._cube["view_projection"].write(matrix)  # type: ignore[union-attr]
        for box in self.boxes:
            (cx, cy, cz), (hx, hy, hz) = box.centre_m, box.half_size_m
            self._cube["base_m"].value = (cx, cy, cz - hz)  # type: ignore[union-attr]
            self._cube["size_m"].value = (2 * hx, 2 * hy, 2 * hz)  # type: ignore[union-attr]
            self._cube["albedo"].value = box.albedo_linear_rgb  # type: ignore[union-attr]
            self._cube["grid_m"].value = box.grid_m  # type: ignore[union-attr]
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

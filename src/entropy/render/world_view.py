"""The world's static solids as the renderer draws them (#51).

- A box is opaque, with its material's albedo (`albedo_linear_rgb`). A material
  without one is drawn neutral grey.
- A tank (open box) is a floor and four walls. Made of a material with a
  refractive index, such as glass, they are transparent panels; otherwise they are
  opaque boxes.
- The ground plane lies under the lowest solid, so it never fights with a floor.

Glass shading: each surface reflects the sky by Fresnel's law (Schlick's
approximation, F0 = ((n - 1)/(n + 1))², 4 % square on for n = 1.5) and lets the rest
through. Both surfaces of a panel are drawn, as a real pane has two.

Simplifications, named (CLAUDE.md rule 7):
- no refraction: what lies behind the glass is not shifted;
- no absorption or tint in the glass (2 cm of clear glass absorbs a few per cent);
- only the sky is reflected, not the scene;
- blending happens after tone mapping, not in linear light;
- half-spaces are not drawn (the Test Tank has none).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import moderngl
import numpy as np

from entropy.render.camera import CameraView, gl_bytes
from entropy.render.scene import (
    DISPLAY_GLSL,
    EXPOSURE,
    HORIZON,
    SKY_GLSL,
    SUN_DIRECTION,
    SceneBox,
    cube_vertices,
)
from entropy.world.materials import Material
from entropy.world.sdf import Box, OpenBox
from entropy.world.world import World

NEUTRAL_GREY = (0.5, 0.5, 0.5)
"""Albedo of a solid whose material gives none."""

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class GlassPanel:
    """A transparent axis-aligned box: centre and half sizes (m), refractive index."""

    centre_m: Vec3
    half_size_m: Vec3
    refractive_index: float


def tank_panels(tank: OpenBox) -> tuple[tuple[Vec3, Vec3], ...]:
    """(centre, half size) of the floor and the four walls, without overlaps: the floor
    covers the whole footprint, the x walls run the full outer depth, the y walls fit
    between them."""
    (x0, y0, z0), (x1, y1, z1), w = tank.inner_min_m, tank.inner_max_m, tank.wall_m
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    hx, hy, hz = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    return (
        ((cx, cy, z0 - w / 2), (hx + w, hy + w, w / 2)),
        ((x0 - w / 2, cy, cz), (w / 2, hy + w, hz)),
        ((x1 + w / 2, cy, cz), (w / 2, hy + w, hz)),
        ((cx, y0 - w / 2, cz), (hx, w / 2, hz)),
        ((cx, y1 + w / 2, cz), (hx, w / 2, hz)),
    )


def _albedo(material: Material) -> Vec3:
    return material.albedo_linear_rgb or NEUTRAL_GREY


def opaque_boxes(world: World) -> tuple[SceneBox, ...]:
    """Boxes, and the panels of tanks that are not transparent."""
    boxes: list[SceneBox] = []
    for shape, material in world.solids:
        if isinstance(shape, Box):
            boxes.append(SceneBox(shape.centre_m, shape.half_size_m, _albedo(material)))
        elif isinstance(shape, OpenBox) and material.refractive_index is None:
            boxes.extend(SceneBox(c, h, _albedo(material)) for c, h in tank_panels(shape))
    return tuple(boxes)


def glass_panels(world: World) -> tuple[GlassPanel, ...]:
    """The panels of tanks made of a transparent material."""
    panels: list[GlassPanel] = []
    for shape, material in world.solids:
        if isinstance(shape, OpenBox) and material.refractive_index is not None:
            n = material.refractive_index
            panels.extend(GlassPanel(c, h, n) for c, h in tank_panels(shape))
    return tuple(panels)


def ground_height_m(world: World) -> float:
    """The bottom of the lowest box or tank (m); 0 without one."""
    bottoms = [
        shape.centre_m[2] - shape.half_size_m[2]
        for shape, _ in world.solids
        if isinstance(shape, Box)
    ] + [
        shape.inner_min_m[2] - shape.wall_m
        for shape, _ in world.solids
        if isinstance(shape, OpenBox)
    ]
    return min(bottoms, default=0.0)


_GLASS_PROGRAM = (
    """
#version 450 core
uniform mat4 view_projection;
uniform vec3 base_m;
uniform vec3 size_m;
in vec3 corner;
in vec3 normal;
out vec3 world_m;
out vec3 face_normal;
void main() {
    world_m = base_m + corner * size_m;
    face_normal = normal;
    gl_Position = view_projection * vec4(world_m, 1.0);
}
""",
    "#version 450 core\n"
    + DISPLAY_GLSL
    + SKY_GLSL
    + """
uniform vec3 eye_m;
uniform float f0;
in vec3 world_m;
in vec3 face_normal;
out vec4 colour;
void main() {
    vec3 v = normalize(eye_m - world_m);
    vec3 n = normalize(face_normal);
    n = dot(n, v) < 0.0 ? -n : n;  // a back face is the pane's other surface
    float c = clamp(dot(n, v), 0.0, 1.0);
    float fresnel = f0 + (1.0 - f0) * pow(1.0 - c, 5.0);
    colour = vec4(display(sky(reflect(-v, n))), fresnel);
}
""",
)


def fresnel_f0(refractive_index: float) -> float:
    """Reflectance square on, from air: ((n - 1)/(n + 1))²."""
    return ((refractive_index - 1.0) / (refractive_index + 1.0)) ** 2


class GlassRenderer:
    """Draws transparent panels over what is already in the framebuffer."""

    def __init__(self, ctx: moderngl.Context, panels: Sequence[GlassPanel]) -> None:
        self.ctx = ctx
        self.panels = tuple(panels)
        self._program = ctx.program(
            vertex_shader=_GLASS_PROGRAM[0], fragment_shader=_GLASS_PROGRAM[1]
        )
        self._buffer = ctx.buffer(cube_vertices().tobytes())
        self._vao = ctx.vertex_array(self._program, [(self._buffer, "3f 3f", "corner", "normal")])
        self._program["sun_direction"].value = SUN_DIRECTION  # type: ignore[union-attr]
        self._program["horizon"].value = HORIZON  # type: ignore[union-attr]
        self._program["exposure"].value = EXPOSURE  # type: ignore[union-attr]

    def render(
        self,
        target: moderngl.Framebuffer,
        view: CameraView,
        size_px: tuple[int, int] | None = None,
    ) -> None:
        """Blend the panels over `target` (see `SceneRenderer.render` for `size_px`)."""
        width, height = size_px or target.size
        if width <= 0 or height <= 0 or not self.panels:
            return
        ctx = self.ctx
        target.use()
        ctx.viewport = (0, 0, width, height)
        matrix = view.projection(width / height).astype(np.float64) @ view.view()
        self._program["view_projection"].write(gl_bytes(matrix))  # type: ignore[union-attr]
        self._program["eye_m"].value = view.position_m  # type: ignore[union-attr]
        ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        # moderngl's stubs omit depth_mask, which it has.
        ctx.depth_mask = False  # pyright: ignore[reportAttributeAccessIssue]
        try:
            for panel in self.panels:
                (cx, cy, cz), (hx, hy, hz) = panel.centre_m, panel.half_size_m
                self._program["base_m"].value = (cx, cy, cz - hz)  # type: ignore[union-attr]
                self._program["size_m"].value = (2 * hx, 2 * hy, 2 * hz)  # type: ignore[union-attr]
                self._program["f0"].value = fresnel_f0(panel.refractive_index)  # type: ignore[union-attr]
                self._vao.render(moderngl.TRIANGLES)
        finally:
            ctx.depth_mask = True  # pyright: ignore[reportAttributeAccessIssue]
            ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)

    def release(self) -> None:
        for resource in (self._vao, self._buffer, self._program):
            resource.release()

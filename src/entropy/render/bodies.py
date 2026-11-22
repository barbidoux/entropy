"""Rigid bodies in the solids pass (SPEC 7.2, #91).

The bodies are drawn after the scene's sky, ground and static boxes, into the same
target and depth, with the same lighting (`scene.LIGHTING_GLSL`): a body box looks like
a static box. Each frame a kernel writes one instance per body into a buffer shared
with OpenGL (`SharedBuffer`, zero copy on CUDA) from the solver's arrays: its position,
orientation (unit quaternion, body to world), shape, size and albedo. The renderer
reads the solver's arrays and never writes them (SPEC 8.1).

Meshes, in the body's frame:
- box: a cube with its face normals, scaled by the half sizes;
- sphere: a UV sphere of `SPHERE_STACKS` by `SPHERE_SLICES`, scaled by the radius;
- capsule: the same sphere split at its equator, the halves moved ±h along z and joined
  by a cylinder, scaled by the radius.

Simplifications, named (CLAUDE.md rule 7):
- every mesh is drawn for every body, and a body keeps only its shape's: the others
  are clipped in the vertex shader. Three draws of all bodies, the vertex work three
  times over (about 900 vertices a body), instead of sorting the bodies by shape;
- spheres and capsules are faceted: 24 slices make a silhouette at most 0.9 % of the
  radius inside the true one.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

import moderngl
import numpy as np
import warp as wp

from entropy.render._body_kernels import write_instances
from entropy.render.camera import CameraView, gl_bytes
from entropy.render.interop import SharedBuffer
from entropy.render.scene import (
    LIGHTING_GLSL,
    SKY_IRRADIANCE,
    SUN_DIRECTION,
    SUN_IRRADIANCE,
    cube_vertices,
)
from entropy.solvers.rigid import Shape

ROWS_PER_BODY = 4
"""vec4s of instance data per body: position and shape, orientation, size, albedo."""
SPHERE_STACKS = 16
SPHERE_SLICES = 24


_VERTEX = """
#version 450 core
uniform mat4 view_projection;
uniform int mesh_shape;
in vec3 local;
in vec3 normal;
in float side;
in vec4 position_shape;
in vec4 orientation;
in vec4 size;
in vec4 albedo_in;
out vec3 world_normal;
out vec3 albedo;
vec3 rotate(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
void main() {
    if (int(position_shape.w + 0.5) != mesh_shape) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);  // clipped: another shape's mesh
        world_normal = vec3(0.0);
        albedo = vec3(0.0);
        return;
    }
    vec3 p;
    if (mesh_shape == 0) {
        p = local * size.xyz;
    } else if (mesh_shape == 1) {
        p = local * size.x;
    } else {
        p = local * size.x + vec3(0.0, 0.0, side * size.y);
    }
    world_normal = rotate(orientation, normal);
    albedo = albedo_in.rgb;
    gl_Position = view_projection * vec4(position_shape.xyz + rotate(orientation, p), 1.0);
}
"""

_FRAGMENT = (
    "#version 450 core\n"
    + LIGHTING_GLSL
    + """
in vec3 world_normal;
in vec3 albedo;
out vec4 colour;
void main() {
    colour = vec4(lit(albedo, normalize(world_normal)), 1.0);
}
"""
)


def box_mesh() -> np.ndarray:
    """(local, normal, side) of a cube with corners at ±1, counter-clockwise outside."""
    cube = cube_vertices().astype(np.float64)
    local = (cube[:, :3] - (0.0, 0.0, 0.5)) * 2.0
    return np.column_stack([local, cube[:, 3:], np.zeros(len(cube))]).astype(np.float32)


def _sphere_rings(capsule: bool) -> list[tuple[float, float]]:
    """(latitude in radians, side) of each ring from the north pole to the south pole; a
    capsule's equator is two rings, one per half."""
    rings = []
    for k in range(SPHERE_STACKS + 1):
        latitude = math.pi / 2.0 - math.pi * k / SPHERE_STACKS
        if capsule and k == SPHERE_STACKS // 2:
            rings += [(latitude, 1.0), (latitude, -1.0)]
        else:
            rings.append((latitude, (1.0 if latitude > 0 else -1.0) if capsule else 0.0))
    return rings


def round_mesh(capsule: bool) -> np.ndarray:
    """(local, normal, side) of a unit sphere, or of a capsule's template (`side` moves a
    vertex ±h along z), counter-clockwise outside."""
    rings = _sphere_rings(capsule)
    points = []
    for latitude, side in rings:
        ring = []
        for j in range(SPHERE_SLICES + 1):
            longitude = 2.0 * math.pi * j / SPHERE_SLICES
            p = (
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            )
            ring.append((*p, *p, side))
        points.append(ring)
    triangles = []
    for upper, lower in itertools.pairwise(points):
        for j in range(SPHERE_SLICES):
            a, b, c, d = upper[j], lower[j], lower[j + 1], upper[j + 1]
            triangles += [a, b, c, a, c, d]
    return np.asarray(triangles, dtype=np.float32)


class BodyRenderer:
    """Draws up to `capacity` rigid bodies into a target that already holds the scene."""

    def __init__(self, ctx: moderngl.Context, capacity: int, device: str) -> None:
        self.ctx = ctx
        self.capacity = capacity
        self.count = 0
        self.instances = SharedBuffer(ctx, capacity * ROWS_PER_BODY, wp.vec4, device)
        self._program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self._program["sun_direction"].value = SUN_DIRECTION  # type: ignore[union-attr]
        self._program["sun_irradiance"].value = SUN_IRRADIANCE  # type: ignore[union-attr]
        self._program["sky_irradiance"].value = SKY_IRRADIANCE  # type: ignore[union-attr]
        self._meshes = []
        for shape, mesh in (
            (Shape.BOX, box_mesh()),
            (Shape.SPHERE, round_mesh(capsule=False)),
            (Shape.CAPSULE, round_mesh(capsule=True)),
        ):
            buffer = ctx.buffer(mesh.tobytes())
            vao = ctx.vertex_array(
                self._program,
                [
                    (buffer, "3f 3f 1f", "local", "normal", "side"),
                    (
                        self.instances.buffer,
                        "4f 4f 4f 4f/i",
                        "position_shape",
                        "orientation",
                        "size",
                        "albedo_in",
                    ),
                ],
            )
            self._meshes.append((int(shape), len(mesh), buffer, vao))

    def update(
        self, fields: Mapping[str, wp.array], count: int, albedo_linear_rgb: wp.array
    ) -> None:
        """Write the instances of the first `count` bodies from the rigid solver's
        `fields()`, and their linear albedo (one vec3 per body)."""
        if not 0 <= count <= self.capacity:
            raise ValueError(f"count must be between 0 and {self.capacity}, got {count}")
        self.count = count
        if not count:
            return
        with self.instances.writable(count * ROWS_PER_BODY) as out:
            wp.launch(
                write_instances,
                dim=count,
                inputs=[
                    fields["positions_m"],
                    fields["orientations"],
                    fields["shapes"],
                    fields["sizes_m"],
                    albedo_linear_rgb,
                    out,
                ],
                device=out.device,
            )

    def render(
        self, target: moderngl.Framebuffer, view: CameraView, size_px: tuple[int, int]
    ) -> None:
        """Draw the bodies over what `target` holds, depth-tested against it."""
        width, height = size_px
        if not self.count or width <= 0 or height <= 0:
            return
        ctx = self.ctx
        target.use()
        projection = view.projection(width / height).astype(np.float64)
        self._program["view_projection"].write(  # type: ignore[union-attr]
            gl_bytes(projection @ view.view().astype(np.float64))
        )
        ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        for shape, vertices, _, vao in self._meshes:
            self._program["mesh_shape"].value = shape  # type: ignore[union-attr]
            vao.render(moderngl.TRIANGLES, vertices=vertices, instances=self.count)
        ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    def release(self) -> None:
        for _, _, buffer, vao in self._meshes:
            vao.release()
            buffer.release()
        self._program.release()
        self.instances.release()

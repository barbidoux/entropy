"""Points from a `SharedBuffer`, for checks and debugging (the game draws water with
the liquid pass, `entropy.render.liquid`).

`PointsRenderer` draws the first `count` `vec3` positions of a buffer as small
points of one colour.

`write_test_pattern` is M0's check of the CUDA-OpenGL path (ADR 0004), kept for the
interop tests: a fixed formula (a slowly turning spiral galaxy), not physics.
"""

from __future__ import annotations

from types import TracebackType

import moderngl
import warp as wp

from entropy.render.camera import CameraView, gl_bytes
from entropy.render.interop import SharedBuffer

PATTERN_CENTRE_M = (0.0, 0.0, 6.0)
PATTERN_RADIUS_M = 4.0
WATER_BLUE = (0.30, 0.62, 1.0)
"""Colour of the particles on screen (display sRGB); a look choice."""


@wp.kernel
def _spiral(points: wp.array(dtype=wp.vec3), time_s: float, radius_m: float, centre: wp.vec3):
    i = wp.tid()
    # The same scatter every frame: only time moves it. (1D tid is an int.)
    state = wp.rand_init(1234, i)  # pyright: ignore[reportArgumentType]
    u = wp.randf(state)
    arm = float(i % 3) * 2.0943951  # pyright: ignore[reportOperatorIssue]
    r = radius_m * wp.sqrt(u)
    spread = 0.35 * (wp.randf(state) - 0.5) * (1.0 + 2.0 * u)
    angle = arm + 2.5 * r / radius_m * 3.0 + spread + 0.25 * time_s * (1.5 - u)
    thickness = 0.25 * (wp.randf(state) - 0.5) * (1.0 - 0.8 * u)
    z = thickness + 0.15 * wp.sin(3.0 * angle + time_s) * u
    points[i] = centre + wp.vec3(r * wp.cos(angle), r * wp.sin(angle), z)


def write_test_pattern(points: wp.array, *, time_s: float) -> None:
    """Fill `points` (vec3, metres) with the spiral at simulated time `time_s` (s)."""
    wp.launch(
        _spiral,
        dim=points.shape[0],
        inputs=[points, time_s, PATTERN_RADIUS_M, wp.vec3(*PATTERN_CENTRE_M)],
        device=points.device,
    )


_VERTEX = """
#version 450 core
uniform mat4 view_projection;
in vec3 position_m;
void main() {
    gl_Position = view_projection * vec4(position_m, 1.0);
    gl_PointSize = 2.0;
}
"""

_FRAGMENT = """
#version 450 core
uniform vec3 point_colour;
out vec4 colour;
void main() {
    colour = vec4(point_colour, 1.0);
}
"""


class PointsRenderer:
    """Draws the positions of a SharedBuffer as points."""

    def __init__(
        self,
        ctx: moderngl.Context,
        shared: SharedBuffer,
        colour: tuple[float, float, float] = WATER_BLUE,
    ) -> None:
        self.ctx = ctx
        self.count = shared.count
        self._program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self._program["point_colour"].value = colour  # type: ignore[union-attr]
        self._vao = ctx.vertex_array(self._program, [(shared.buffer, "3f", "position_m")])

    def render(
        self,
        target: moderngl.Framebuffer,
        view: CameraView,
        size_px: tuple[int, int] | None = None,
        *,
        count: int | None = None,
    ) -> None:
        """Draw the first `count` points (default: all) over `size_px` of `target` (see
        `SceneRenderer.render`)."""
        width, height = size_px or target.size
        count = self.count if count is None else count
        if not 0 <= count <= self.count:
            raise ValueError(f"count must be between 0 and {self.count}, got {count}")
        if width <= 0 or height <= 0 or count == 0:
            return
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        view_projection = view.projection(width / height) @ view.view()
        self._program["view_projection"].write(gl_bytes(view_projection))  # type: ignore[union-attr]
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._vao.render(moderngl.POINTS, vertices=count)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)

    def release(self) -> None:
        self._vao.release()
        self._program.release()

    def __enter__(self) -> PointsRenderer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

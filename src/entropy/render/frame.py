"""The frame's HDR target and its last pass, tone mapping (SPEC 7.2: post).

Passes draw linear radiance (relative units: SPEC 7.2 brings calibration) into a
half-float target, so light brighter than white keeps its value and blending
happens in linear light. The tone-mapping pass turns it into what the display
shows: scaled by `EXPOSURE`, the ACES fit of Narkowicz (2015), then the sRGB
transfer function (IEC 61966-2-1). Shaders are GLSL 4.50 (ADR 0009).
"""

from __future__ import annotations

import moderngl
import numpy as np

EXPOSURE = 0.5

DISPLAY_GLSL = """
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


def srgb_from_linear(value: float) -> float:
    """The sRGB transfer function (IEC 61966-2-1), linear [0, 1] to encoded [0, 1]."""
    if value <= 0.0031308:
        return 12.92 * value
    return 1.055 * value ** (1.0 / 2.4) - 0.055


def display_reference(radiance: np.ndarray) -> np.ndarray:
    """The shader's `display`, in NumPy: the reference tests compare images with."""
    x = np.asarray(radiance, dtype=float) * EXPOSURE
    aces = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)
    return np.vectorize(srgb_from_linear)(aces)


class HdrTarget:
    """A half-float colour texture and a depth texture of one size, as a framebuffer."""

    def __init__(self, ctx: moderngl.Context, size_px: tuple[int, int]) -> None:
        self.size_px = size_px
        self.colour = ctx.texture(size_px, 4, dtype="f2")
        self.depth = ctx.depth_texture(size_px)
        self.framebuffer = ctx.framebuffer([self.colour], self.depth)

    def release(self) -> None:
        for resource in (self.framebuffer, self.colour, self.depth):
            resource.release()


_TONE_MAP = (
    """
#version 450 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
""",
    "#version 450 core\n"
    + DISPLAY_GLSL
    + """
uniform sampler2D hdr;
out vec4 colour;
void main() {
    colour = vec4(display(texelFetch(hdr, ivec2(gl_FragCoord.xy), 0).rgb), 1.0);
}
""",
)


class ToneMapper:
    """Draws an HDR target onto a display framebuffer."""

    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._program = ctx.program(vertex_shader=_TONE_MAP[0], fragment_shader=_TONE_MAP[1])
        self._program["exposure"].value = EXPOSURE  # type: ignore[union-attr]
        self._program["hdr"].value = 0  # type: ignore[union-attr]
        self._vao = ctx.vertex_array(self._program, [])

    def render(
        self, target: moderngl.Framebuffer, source: HdrTarget, size_px: tuple[int, int]
    ) -> None:
        width, height = size_px
        if width <= 0 or height <= 0:
            return
        target.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND)
        source.colour.use(location=0)
        self._vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self._vao.release()
        self._program.release()

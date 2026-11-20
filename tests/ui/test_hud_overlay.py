"""The Dear ImGui overlay of the performance HUD (`entropy.ui.hud.PerformanceHud`)."""

from collections.abc import Iterator

import moderngl
import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.ui.hud import PerformanceHud

pytestmark = pytest.mark.render

LINES = ["60.0 fps", "frame 16.67 ms", "interop zero-copy"]


@pytest.fixture
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _draw_into_black(window: GameWindow, hud: PerformanceHud) -> np.ndarray:
    """Draw two HUD frames (Dear ImGui hides an auto-sized window in its first) over black."""
    size = window.framebuffer_size_px()
    target = window.ctx.simple_framebuffer(size)
    try:
        for _ in range(2):
            target.use()
            target.clear(0.0, 0.0, 0.0, 1.0)
            hud.draw(LINES)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)
    finally:
        target.release()
    return image.reshape(size[1], size[0], 3)[::-1]


def test_hud_lights_the_top_left_corner_only(window: GameWindow) -> None:
    with PerformanceHud(window.handle) as hud:
        image = _draw_into_black(window, hud)

    height, width, _ = image.shape
    assert image[: height // 3, : width // 2].max() > 100
    assert image[-height // 4 :, -width // 4 :].max() == 0


def test_f3_toggles_the_hud_off_and_on(window: GameWindow) -> None:
    with PerformanceHud(window.handle) as hud:
        hud.toggle()
        hidden = _draw_into_black(window, hud)
        hud.toggle()
        shown = _draw_into_black(window, hud)

    assert hidden.max() == 0
    assert shown.max() > 100


def test_moderngl_draws_correctly_after_the_hud(window: GameWindow) -> None:
    """The ImGui backend must restore the OpenGL state moderngl relies on."""
    ctx = window.ctx
    program = ctx.program(
        vertex_shader="""
        #version 450 core
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
            gl_Position = vec4(p, 0.5, 1.0);
        }""",
        fragment_shader="""
        #version 450 core
        out vec4 colour;
        void main() { colour = vec4(0.0, 1.0, 0.0, 1.0); }""",
    )
    vao = ctx.vertex_array(program, [])
    size = window.framebuffer_size_px()
    target = ctx.simple_framebuffer(size)
    try:
        with PerformanceHud(window.handle) as hud:
            ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
            target.use()
            target.clear(0.0, 0.0, 0.0, 1.0)
            hud.draw(LINES)
            hud.draw(LINES)
            ctx.disable(moderngl.BLEND)
            target.use()
            ctx.viewport = (0, 0, *size)
            target.clear(1.0, 0.0, 0.0, 1.0)
            vao.render(moderngl.TRIANGLES, vertices=3)
            centre = np.frombuffer(
                target.read(viewport=(size[0] // 2, size[1] // 2, 1, 1), components=3),
                dtype=np.uint8,
            )
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    finally:
        target.release()
        vao.release()
        program.release()

    assert list(centre) == [0, 255, 0]


def test_hud_starts_visible_and_release_is_idempotent(window: GameWindow) -> None:
    hud = PerformanceHud(window.handle)

    assert hud.visible
    hud.release()
    hud.release()


def test_a_missing_window_is_refused_before_anything_is_created() -> None:
    with pytest.raises(ValueError, match="window"):
        PerformanceHud(None)

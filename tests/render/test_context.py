"""The game window and its OpenGL context (`entropy.render.context`), SPEC 7.1."""

import dataclasses

import glfw
import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow

pytestmark = pytest.mark.render

SMALL = WindowSettings(width_px=320, height_px=200, title="ENTROPY test", vsync=False)


def test_window_has_the_requested_core_context() -> None:
    with GameWindow(SMALL, visible=False) as window:
        major, minor = SMALL.gl_version

        assert window.ctx.version_code >= major * 100 + minor * 10
        assert SMALL.gl_version == (4, 6)


def test_context_renders_and_reads_back() -> None:
    with GameWindow(SMALL, visible=False) as window:
        target = window.ctx.simple_framebuffer((8, 4))
        target.use()
        target.clear(0.25, 0.5, 0.75, 1.0)

        pixels = np.frombuffer(target.read(components=4), dtype=np.uint8).reshape(4, 8, 4)

        np.testing.assert_allclose(pixels[..., :3], [[[64, 128, 191]] * 8] * 4, atol=1)


def test_framebuffer_size_matches_the_window() -> None:
    with GameWindow(SMALL, visible=False) as window:
        width, height = window.framebuffer_size_px()

        assert (width, height) == (SMALL.width_px, SMALL.height_px)


def test_windows_can_be_opened_one_after_another() -> None:
    for _ in range(2):
        with GameWindow(SMALL, visible=False) as window:
            assert not window.should_close()


def test_close_is_idempotent_and_releases_the_window() -> None:
    window = GameWindow(SMALL, visible=False)

    window.close()
    window.close()

    assert window.closed
    assert window.should_close()
    with pytest.raises(RuntimeError, match="closed"):
        window.framebuffer_size_px()


def test_frame_calls_and_mouse_capture() -> None:
    with GameWindow(SMALL, visible=False) as window:
        window.poll_events()
        window.swap_buffers()

        window.set_mouse_captured(True)
        captured = glfw.get_input_mode(window.handle, glfw.CURSOR)
        window.set_mouse_captured(False)
        released = glfw.get_input_mode(window.handle, glfw.CURSOR)

        assert (captured, released) == (glfw.CURSOR_DISABLED, glfw.CURSOR_NORMAL)


def test_an_impossible_gl_version_is_a_clear_error() -> None:
    with pytest.raises(RuntimeError, match=r"OpenGL 9\.9"):
        GameWindow(dataclasses.replace(SMALL, gl_version=(9, 9)), visible=False)


def test_no_display_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(glfw, "init", lambda: False)

    with pytest.raises(RuntimeError, match="GLFW could not start"):
        GameWindow(SMALL, visible=False)

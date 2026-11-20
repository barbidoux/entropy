"""The game window and its OpenGL context (`entropy.render.context`), SPEC 7.1."""

import dataclasses

import glfw
import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow

pytestmark = pytest.mark.render


def test_window_has_the_requested_core_context(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        major, minor = window_settings.gl_version

        assert window.ctx.version_code >= major * 100 + minor * 10


def test_context_renders_and_reads_back(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        target = window.ctx.simple_framebuffer((8, 4))
        target.use()
        target.clear(0.25, 0.5, 0.75, 1.0)

        pixels = np.frombuffer(target.read(components=4), dtype=np.uint8).reshape(4, 8, 4)

        np.testing.assert_allclose(pixels[..., :3], [[[64, 128, 191]] * 8] * 4, atol=1)


def test_framebuffer_size_matches_the_window(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        width, height = window.framebuffer_size_px()

        assert (width, height) == (window_settings.width_px, window_settings.height_px)


def test_windows_can_be_opened_one_after_another(window_settings: WindowSettings) -> None:
    for _ in range(2):
        with GameWindow(window_settings, visible=False) as window:
            assert not window.should_close()


def test_close_is_idempotent_and_releases_the_window(window_settings: WindowSettings) -> None:
    window = GameWindow(window_settings, visible=False)

    window.close()
    window.close()

    assert window.closed
    assert window.should_close()
    with pytest.raises(RuntimeError, match="closed"):
        window.framebuffer_size_px()


def test_frame_calls_and_mouse_capture(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        window.poll_events()
        window.swap_buffers()

        window.set_mouse_captured(True)
        captured = glfw.get_input_mode(window.handle, glfw.CURSOR)
        window.set_mouse_captured(False)
        released = glfw.get_input_mode(window.handle, glfw.CURSOR)

        assert (captured, released) == (glfw.CURSOR_DISABLED, glfw.CURSOR_NORMAL)


@pytest.mark.filterwarnings("error")
def test_an_impossible_gl_version_is_one_clear_error(window_settings: WindowSettings) -> None:
    """GLFW's own diagnosis goes into the error instead of a stray warning."""
    with pytest.raises(RuntimeError, match=r"no OpenGL 9\.9 core context.*\(GLFW: .+\)"):
        GameWindow(dataclasses.replace(window_settings, gl_version=(9, 9)), visible=False)


def test_no_display_is_a_clear_error(
    window_settings: WindowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glfw, "init", lambda: False)

    with pytest.raises(RuntimeError, match="GLFW could not start"):
        GameWindow(window_settings, visible=False)

"""The game window and its OpenGL context (`entropy.render.context`), SPEC 7.1."""

import dataclasses

import glfw
import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render import context
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


def _glfw_is_terminated() -> bool:
    """GLFW reports 'not initialized' for any call after `glfw.terminate()`."""
    with pytest.warns(glfw.GLFWError) as record:
        glfw.get_current_context()
    return any("not initialized" in str(w.message).lower() for w in record)


def test_closing_terminates_glfw(window_settings: WindowSettings) -> None:
    GameWindow(window_settings, visible=False).close()

    assert _glfw_is_terminated()


def test_a_failed_window_leaves_nothing_behind(window_settings: WindowSettings) -> None:
    with pytest.raises(RuntimeError):
        GameWindow(dataclasses.replace(window_settings, gl_version=(9, 9)), visible=False)

    assert _glfw_is_terminated()


def test_a_failure_after_the_window_exists_cleans_up(
    window_settings: WindowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(**kwargs: object) -> None:
        raise RuntimeError("context creation failed")

    monkeypatch.setattr(context.moderngl, "create_context", broken)

    with pytest.raises(RuntimeError, match="context creation failed"):
        GameWindow(window_settings, visible=False)
    assert _glfw_is_terminated()


def test_a_second_window_at_the_same_time_is_refused(window_settings: WindowSettings) -> None:
    """GLFW is process-wide: closing one window would terminate it under the other."""
    with GameWindow(window_settings, visible=False) as first:
        with pytest.raises(RuntimeError, match="one game window"):
            GameWindow(window_settings, visible=False)

        assert first.framebuffer_size_px() == (window_settings.width_px, window_settings.height_px)


def test_the_context_is_unusable_after_close(window_settings: WindowSettings) -> None:
    window = GameWindow(window_settings, visible=False)
    window.close()

    with pytest.raises(RuntimeError, match="closed"):
        _ = window.ctx


def test_raw_mouse_motion_when_captured(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        window.set_mouse_captured(True)

        if glfw.raw_mouse_motion_supported():
            assert glfw.get_input_mode(window.handle, glfw.RAW_MOUSE_MOTION) == 1


def test_no_display_is_a_clear_error(
    window_settings: WindowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(glfw, "init", lambda: False)

    with pytest.raises(RuntimeError, match="GLFW could not start"):
        GameWindow(window_settings, visible=False)

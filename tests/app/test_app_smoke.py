"""The assembled game runs (`entropy.app.main`): window, loop, scene, interop, HUD."""

import dataclasses

import glfw
import pytest

from entropy.app.main import GameStages, run
from entropy.core.config import Settings, WindowSettings
from entropy.core.profiler import FrameStats
from entropy.render.context import GameWindow

pytestmark = pytest.mark.render


def test_the_game_runs_a_few_frames_headless(window_settings: WindowSettings) -> None:
    settings = dataclasses.replace(Settings(), window=window_settings)

    stats = run(settings, visible=False, max_frames=5)

    summary = stats.summary()
    assert summary.frames == 5
    assert summary.frame_ms_mean > 0.0


def test_drawing_follows_a_resized_window(window_settings: WindowSettings) -> None:
    with GameWindow(window_settings, visible=False) as window:
        stages = GameStages(
            window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
        )
        try:
            glfw.set_window_size(window.handle, 400, 240)
            window.poll_events()
            stages.render()

            assert window.ctx.viewport == (0, 0, *window.framebuffer_size_px())
            assert window.framebuffer_size_px() == (400, 240)
        finally:
            stages.release()


def test_glfw_events_reach_god_mode(window_settings: WindowSettings) -> None:
    """The key, cursor and button callbacks feed God mode's input."""
    with GameWindow(window_settings, visible=False) as window:
        stages = GameStages(
            window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
        )
        try:
            on_key, on_cursor, on_button = stages.callbacks
            on_button(window.handle, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            on_key(window.handle, glfw.KEY_W, 0, glfw.PRESS, 0)
            on_cursor(window.handle, 10.0, 10.0)
            stages.update(0.1)

            assert stages.mode.input.mouse_captured
            assert glfw.get_input_mode(window.handle, glfw.CURSOR) == glfw.CURSOR_DISABLED
        finally:
            stages.release()

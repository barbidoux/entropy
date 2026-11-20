"""The assembled game runs (`entropy.app.main`): window, loop, scene, interop, HUD."""

import dataclasses

import glfw
import pytest

from entropy.app import main as app_main
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


def test_hud_notes_give_the_controls_and_any_interop_fallback(
    window_settings: WindowSettings,
) -> None:
    with GameWindow(window_settings, visible=False) as window:
        stages = GameStages(
            window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
        )
        try:
            stages.pattern.fallback_reason = "CUDA-OpenGL registration failed: test"
            notes = stages.hud_notes()

            assert any("Échap" in note for note in notes)
            assert "CUDA-OpenGL registration failed: test" in "\n".join(notes)
        finally:
            stages.release()


def test_a_failed_set_up_releases_what_it_made(
    window_settings: WindowSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resources made before a failure are released before the window closes."""
    released: list[str] = []
    original = app_main.SharedBuffer.release

    def counting_release(self: app_main.SharedBuffer) -> None:
        released.append("pattern")
        original(self)

    def broken_hud(*args: object, **kwargs: object) -> None:
        raise RuntimeError("HUD could not start")

    monkeypatch.setattr(app_main.SharedBuffer, "release", counting_release)
    monkeypatch.setattr(app_main, "PerformanceHud", broken_hud)
    with GameWindow(window_settings, visible=False) as window:
        with pytest.raises(RuntimeError, match="HUD could not start"):
            GameStages(
                window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
            )

        assert released == ["pattern"]


def test_glfw_events_reach_god_mode(window_settings: WindowSettings) -> None:
    """The key, cursor and button callbacks feed God mode's input."""
    with GameWindow(window_settings, visible=False) as window:
        stages = GameStages(
            window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
        )
        try:
            on_key, on_cursor, on_button = stages.callbacks
            start = stages.mode.pose
            on_button(window.handle, glfw.MOUSE_BUTTON_LEFT, glfw.PRESS, 0)
            on_cursor(window.handle, 0.0, 0.0)
            on_cursor(window.handle, 10.0, 0.0)
            stages.update(0.1)
            turned = stages.mode.pose
            forward_key = next(
                k for k, a in stages.mode.input.bindings.items() if a.value == "forward"
            )
            on_key(window.handle, forward_key, 0, glfw.PRESS, 0)
            stages.update(0.1)

            assert stages.mode.input.mouse_captured
            assert glfw.get_input_mode(window.handle, glfw.CURSOR) == glfw.CURSOR_DISABLED
            assert turned.yaw_rad != start.yaw_rad
            assert stages.mode.pose.position_m != turned.position_m
        finally:
            stages.release()

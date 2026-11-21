"""The assembled game runs (`entropy.app.main`): window, loop, scene, interop, HUD."""

import dataclasses
from collections.abc import Iterator
from pathlib import Path

import glfw
import numpy as np
import pytest
import warp as wp

from entropy.app import main as app_main
from entropy.app.main import GameStages, run
from entropy.core.config import Settings, WindowSettings
from entropy.core.profiler import FrameStats
from entropy.render.context import GameWindow
from entropy.solvers.liquid.solver import NonFiniteStateError, fill_box

pytestmark = pytest.mark.render


@pytest.fixture
def stages(window_settings: WindowSettings) -> Iterator[GameStages]:
    with GameWindow(window_settings, visible=False) as window:
        made = GameStages(
            window, dataclasses.replace(Settings(), window=window_settings), FrameStats()
        )
        try:
            yield made
        finally:
            made.release()


def test_the_game_starts_in_the_test_tank(stages: GameStages) -> None:
    """#51: the Test Tank's world, its liquid in the simulation, its floor and glass drawn."""
    assert stages.world.name == "Test Tank"
    assert stages.liquid is not None
    assert stages.simulation.solvers == (stages.liquid,)
    assert len(stages.frame.scene.boxes) == 1  # the concrete slab
    assert len(stages.frame.glass.panels) == 5  # the tank's floor and walls
    assert any("Test Tank" in note for note in stages.hud_notes())


def test_the_liquid_s_particles_are_drawn_as_water(stages: GameStages) -> None:
    assert stages.liquid is not None
    stages.liquid.add_particles(fill_box((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5), 0.0625, 8, seed=1))

    stages.step_simulation(1.0 / 60.0)
    stages.render()

    assert stages.particles_drawn == stages.liquid.diagnostics()["particles"] == 16 * 16 * 8 * 8
    assert any(f"{16 * 16 * 8 * 8:,}" in note for note in stages.hud_notes())


def test_a_non_finite_liquid_freezes_the_world_and_says_so(
    stages: GameStages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC 5.9: the solver dumps its state; the game stops stepping and alerts."""
    assert stages.liquid is not None

    def broken_step(dt_s: float, ctx: object = None) -> None:
        raise NonFiniteStateError(Path("out/nan_dumps/liquid-test.npz"))

    monkeypatch.setattr(stages.liquid, "step", broken_step)

    assert stages.step_simulation(1.0 / 60.0) == 0.0
    assert stages.step_simulation(1.0 / 60.0) == 0.0  # stays frozen
    notes = "\n".join(stages.hud_notes())
    assert "liquid-test.npz" in notes
    assert np.isfinite(stages.loop_settings.time_scale)


def test_the_game_runs_a_few_frames_headless(window_settings: WindowSettings) -> None:
    settings = dataclasses.replace(Settings(), window=window_settings)

    stats = run(settings, visible=False, max_frames=5)

    summary = stats.summary()
    assert summary.frames == 5
    assert summary.frame_ms_mean > 0.0


def test_the_game_records_gpu_time_per_stage(window_settings: WindowSettings) -> None:
    """Render passes get OpenGL times, and the Warp test pattern CUDA times on a GPU."""
    settings = dataclasses.replace(Settings(), window=window_settings)

    stats = run(settings, visible=False, max_frames=10)

    gpu = stats.summary().gpu_ms_mean
    passes = {"scene", "liquid", "glass", "tone map", "HUD"}
    assert {f"{name} (OpenGL)" for name in passes} <= gpu.keys()
    assert ("simulation (CUDA)" in gpu) == wp.is_cuda_available()
    assert all(ms >= 0.0 for ms in gpu.values())


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
            stages.particles.fallback_reason = "CUDA-OpenGL registration failed: test"
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
        released.append("particles")
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

        assert released == ["particles"]


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

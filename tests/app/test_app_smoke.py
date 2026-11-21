"""The assembled game runs (`entropy.app.main`): window, loop, scene, interop, HUD."""

import dataclasses
import math
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
from entropy.game.free_camera import CameraPose
from entropy.game.input import Action, FrameInput
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
    assert "MONDE ARRÊTÉ" in notes and "divergé" in notes and "liquid-test.npz" in notes
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
            text = "\n".join(notes)
            assert "CUDA-OpenGL registration failed: test" in text
            assert "pause" in text and "F5" in text and "F9" in text  # the time keys
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
            on_key, on_cursor, on_button, on_scroll = stages.callbacks
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

            on_scroll(window.handle, 0.0, 2.0)
            stages.update(0.1)
            assert stages.toolbox.current.values["flow_l_per_s"] == 22.0
            assert stages.mode.input.mouse_captured
            assert glfw.get_input_mode(window.handle, glfw.CURSOR) == glfw.CURSOR_DISABLED
            assert turned.yaw_rad != start.yaw_rad
            assert stages.mode.pose.position_m != turned.position_m
        finally:
            stages.release()


def test_the_hose_fills_the_tank(stages: GameStages) -> None:
    """#53: holding the tool button with the hose in hand pours water into the tank; the
    jet in flight is drawn with the water."""
    from entropy.game.free_camera import CameraPose
    from entropy.game.input import FrameInput

    assert stages.liquid is not None
    stages.mode.pose = CameraPose((0.0, -1.0, 2.8), 1.5708, -0.9)  # above the rim, aiming in
    hold = FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_held=True, tool_pressed=True)
    stages.toolbox.handle(hold)
    for _ in range(20):
        stages.step_simulation(1.0 / 60.0)
        stages.toolbox.handle(FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_held=True))
    stages.render()

    in_tank = stages.liquid.diagnostics()["particles"]
    assert in_tank > 0
    assert stages.particles_drawn == in_tank + len(stages.toolbox.jet)
    assert any("Tuyau" in note for note in stages.hud_notes())


def _press(stages: GameStages, key: int) -> None:
    on_key = stages.callbacks[0]
    on_key(stages.window.handle, key, 0, glfw.PRESS, 0)
    on_key(stages.window.handle, key, 0, glfw.RELEASE, 0)
    stages.update(1.0 / 60.0)


def test_time_keys_pause_slow_and_step(stages: GameStages) -> None:
    """#55, SPEC 4.5: P pauses, [ and ] cycle the scale, the period steps; the HUD says so."""
    _press(stages, glfw.KEY_P)
    assert stages.time_control.paused
    assert any("EN PAUSE" in note for note in stages.hud_notes())
    _press(stages, glfw.KEY_LEFT_BRACKET)
    assert stages.time_control.scale == 0.5
    _press(stages, glfw.KEY_PERIOD)
    assert stages.time_control.step_requested
    _press(stages, glfw.KEY_RIGHT_BRACKET)
    assert stages.time_control.scale == 1.0


def test_f5_saves_and_f9_restores_the_water(stages: GameStages) -> None:
    """#55: a checkpoint of the liquid, and back to it."""
    assert stages.liquid is not None
    stages.liquid.add_particles(fill_box((-0.5, -0.5, 0.0), (0.5, 0.5, 0.25), 0.0625, 8, seed=2))
    stages.step_simulation(1.0 / 60.0)
    _press(stages, glfw.KEY_F5)
    saved = stages.liquid.particle_positions().copy()
    stages.liquid.add_particles(fill_box((1.0, 1.0, 1.0), (1.5, 1.5, 1.5), 0.0625, 8, seed=3))
    for _ in range(5):
        stages.step_simulation(1.0 / 60.0)

    stages.toolbox.jet.launch(np.array([[0.0, 0.0, 9.0]]), np.zeros((1, 3)))  # in flight

    _press(stages, glfw.KEY_F9)

    np.testing.assert_allclose(stages.liquid.particle_positions(), saved)
    assert len(stages.toolbox.jet) == 0  # water in flight is not saved (ADR 0024)


def test_the_hud_counts_the_water(stages: GameStages) -> None:
    """#56: particles, litres, cells of water, substeps and pressure iterations."""
    assert stages.liquid is not None
    stages.liquid.add_particles(fill_box((-0.5, -0.5, 0.0), (0.5, 0.5, 0.5), 0.0625, 8, seed=4))
    stages.step_simulation(1.0 / 60.0)
    stages.step_simulation(1.0 / 60.0)

    text = "\n".join(stages.hud_notes())

    litres = 16 * 16 * 8 * 8 * 0.0625**3 / 8 * 1e3  # the particles' own volume: 125 L
    assert f"{litres:,.0f} L" in text
    assert "particules" in text and "cellules" in text
    assert "sous-pas" in text and "itérations" in text


def test_a_frozen_world_saves_no_checkpoint_and_keeps_the_last_good_one(
    stages: GameStages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC 5.9, 4.6: F5 while frozen would save the non-finite state, and so would one
    saved just before the freeze was seen; F9 must lead back to finite water."""
    assert stages.liquid is not None
    stages.liquid.add_particles(fill_box((-0.5, -0.5, 0.0), (0.5, 0.5, 0.25), 0.0625, 8, seed=5))
    stages.step_simulation(1.0 / 60.0)
    _press(stages, glfw.KEY_F5)  # checked by the next step
    good_s = stages.rewind.ring.clock_s
    stages.step_simulation(1.0 / 60.0)
    _press(stages, glfw.KEY_F5)  # saved after the step that goes non-finite

    def found_non_finite() -> float:
        raise NonFiniteStateError(Path("out/nan_dumps/liquid-test.npz"))

    monkeypatch.setattr(stages.liquid, "max_advance_s", found_non_finite)
    assert stages.step_simulation(1.0 / 60.0) == 0.0
    _press(stages, glfw.KEY_F5)  # refused while frozen

    assert stages.rewind.ring.saved_times_s() == [0.0, good_s]  # and the one at the start
    _press(stages, glfw.KEY_F9)
    assert stages.simulation.frozen_by is None


def test_a_click_while_paused_is_not_kept_for_later(stages: GameStages) -> None:
    """Paused, no tool acts; the click must not drop the bucket when play resumes
    (#74's review)."""
    assert stages.liquid is not None
    looking_in = CameraPose((0.0, -3.0, 3.0), math.radians(90.0), math.radians(-37.0))
    stages.mode.pose = looking_in  # from above the rim, down into the tank
    click = FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_pressed=True, tool_released=True)
    stages.toolbox.handle(FrameInput(frozenset(), frozenset({Action.TOOL_2}), 0.0, 0.0))
    stages.toolbox.handle(click)
    stages.step_simulation(1.0 / 60.0)
    dropped = stages.liquid.diagnostics()["particles"]
    assert dropped > 0  # running, a click drops the bucket where the camera aims

    stages.time_control.toggle_pause()
    stages.toolbox.handle(click)
    stages.update(1.0 / 60.0)
    stages.time_control.toggle_pause()
    stages.step_simulation(1.0 / 60.0)

    assert stages.liquid.diagnostics()["particles"] == dropped

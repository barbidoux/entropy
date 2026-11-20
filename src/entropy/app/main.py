"""`uv run entropy`: open the window and run the M0 scene (SPEC 10.5).

Wiring only (ADR 0005, ADR 0010): GLFW callbacks feed God mode's input, the
frame loop calls the stages below, the renderer draws the scene and the
interop test pattern, and the HUD shows the frame statistics.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import glfw
import warp as wp

from entropy.core.config import Settings, default_settings_path, load_settings
from entropy.core.profiler import FrameStats, gpu_memory
from entropy.game.god_mode import GodMode
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.interop import SharedBuffer
from entropy.render.points import PointsRenderer, write_test_pattern
from entropy.render.scene import SceneRenderer
from entropy.sim.loop import FrameLoop, LoopSettings
from entropy.ui.hud import PerformanceHud, hud_lines

_log = logging.getLogger(__name__)

PATTERN_POINTS = 200_000


class GameStages:
    """The frame loop's stages for the M0 scene."""

    def __init__(self, window: GameWindow, settings: Settings, stats: FrameStats) -> None:
        self.window = window
        self.settings = settings
        self.stats = stats
        self.loop_settings = LoopSettings()
        self.mode = GodMode.from_settings(settings.camera)
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self._mouse_captured = False
        self._pattern_time_s = 0.0

        ctx = window.ctx
        self.scene = SceneRenderer(ctx)
        self.pattern = SharedBuffer(ctx, PATTERN_POINTS, wp.vec3, self.device)
        self.points = PointsRenderer(ctx, self.pattern)
        self.hud = PerformanceHud(window.handle)
        self._install_callbacks()

    def _install_callbacks(self) -> None:
        handle = self.window.handle
        mode = self.mode

        def on_key(_window: Any, key: int, _scancode: int, action: int, _mods: int) -> None:
            mode.input.on_key(key, action)

        def on_cursor(_window: Any, x: float, y: float) -> None:
            mode.input.on_mouse_move(x, y)

        def on_button(_window: Any, _button: int, action: int, _mods: int) -> None:
            mode.on_mouse_button(pressed=action == glfw.PRESS)

        self.callbacks = (on_key, on_cursor, on_button)  # GLFW keeps no reference
        glfw.set_key_callback(handle, on_key)
        glfw.set_cursor_pos_callback(handle, on_cursor)
        glfw.set_mouse_button_callback(handle, on_button)

    def poll_input(self) -> None:
        self.window.poll_events()

    def update(self, frame_dt_s: float) -> None:
        self.mode.update(frame_dt_s)
        captured = self.mode.input.mouse_captured
        if captured != self._mouse_captured:
            self.window.set_mouse_captured(captured)
            self._mouse_captured = captured
        self.hud.visible = self.mode.hud_visible

    def step_simulation(self, dt_s: float) -> None:
        self._pattern_time_s += dt_s

    def render(self) -> None:
        width, height = self.window.framebuffer_size_px()
        if width == 0 or height == 0:
            return  # minimised
        with self.pattern.writable() as positions:
            write_test_pattern(positions, time_s=self._pattern_time_s)
        camera = self.settings.camera
        pose = self.mode.pose
        forward = pose.forward()
        view = CameraView(
            pose.position_m,
            (float(forward[0]), float(forward[1]), float(forward[2])),
            fov_y_rad=math.radians(camera.fov_y_deg),
            near_m=camera.near_m,
            far_m=camera.far_m,
        )
        screen = self.window.ctx.screen
        self.scene.render(screen, view, (width, height))
        self.points.render(screen, view, (width, height))
        lines = hud_lines(
            self.stats.summary(),
            gpu_memory(self.device),
            interop=self.pattern.mode.value,
            time_scale=self.loop_settings.time_scale,
            notes=[f"{PATTERN_POINTS:,} points by Warp (interop test pattern)"],
        )
        self.hud.draw(lines)

    def present(self) -> None:
        self.window.swap_buffers()

    def release(self) -> None:
        self.hud.release()
        self.points.release()
        self.pattern.release()
        self.scene.release()


def run(settings: Settings, *, visible: bool = True, max_frames: int | None = None) -> FrameStats:
    """Open the window and run frames until it is closed (or `max_frames` ran)."""
    stats = FrameStats()
    with GameWindow(settings.window, visible=visible) as window:
        stages = GameStages(window, settings, stats)
        loop = FrameLoop(stages, settings=stages.loop_settings, stats=stats)
        frames = 0

        def should_close() -> bool:
            nonlocal frames
            frames += 1
            return window.should_close() or (max_frames is not None and frames > max_frames)

        try:
            loop.run(should_close)
        finally:
            stages.release()
    return stats


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(load_settings(default_settings_path()))
    return 0

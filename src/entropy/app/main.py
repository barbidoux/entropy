"""`uv run entropy`: open the window in the Test Tank (SPEC 4.4, 10.5).

Wiring only (ADR 0005, ADR 0010): the Test Tank's map gives the world; the world
gives the liquid's setup and what the renderer draws (floor, glass). GLFW
callbacks feed God mode's input, the frame loop calls the stages below, the
liquid's particles reach OpenGL through the interop buffer, and the HUD shows the
frame statistics.

NaN guard (SPEC 5.9): when the liquid finds a non-finite state it dumps it and
raises; the game then stops stepping the world, keeps drawing it, and says where
the dump is.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import glfw
import warp as wp

from entropy.core.config import Settings, default_settings_path, load_settings
from entropy.core.paths import content_dir, out_dir
from entropy.core.profiler import FrameStats, StageTimer, cuda_stage_timer, gpu_memory
from entropy.game.god_mode import GodMode
from entropy.game.input import bindings_from_settings
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.gl_timer import GlQueries
from entropy.render.interop import SharedBuffer
from entropy.render.world_frame import WorldFrame
from entropy.sim.loop import FrameLoop, LoopSettings
from entropy.sim.setup import liquid_setup
from entropy.sim.simulation import Simulation
from entropy.solvers.liquid.solver import LiquidSolver
from entropy.ui.hud import PerformanceHud, hud_lines
from entropy.world.maps import load_map
from entropy.world.materials import load_material
from entropy.world.world import build_world

_log = logging.getLogger(__name__)

START_MAP = "test_tank"
"""The map the game opens (SPEC 4.4)."""
DEFAULT_LOG_PATH = out_dir() / "entropy.log"
"""Where the game logs, next to the clone (`out/` is ignored by git)."""
CONTROLS_HINT = "clic : regarder   Échap : libérer la souris   F3 : HUD"


class GameStages:
    """The frame loop's stages for the Test Tank."""

    def __init__(self, window: GameWindow, settings: Settings, stats: FrameStats) -> None:
        self.window = window
        self.settings = settings
        self.stats = stats
        self.loop_settings = LoopSettings()
        bindings = bindings_from_settings(settings.controls)  # needs GLFW running
        self.mode = GodMode.from_settings(settings.camera, bindings)
        self.device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self._mouse_captured = False
        self.world = build_world(load_map(content_dir() / "maps" / START_MAP))
        setup = liquid_setup(self.world)
        self.liquid = LiquidSolver.create(setup, self.device) if setup is not None else None
        self.simulation = Simulation([self.liquid] if self.liquid is not None else [])
        self.particles_drawn = 0
        # GPU time per stage for the HUD (SPEC 4.7): CUDA events around Warp work,
        # OpenGL timer queries around render passes.
        self.cuda_timer = cuda_stage_timer(self.device)
        self.gl_timer = StageTimer(GlQueries(window.ctx))

        ctx = window.ctx
        # Everything made here is released in reverse order, also if a later step fails,
        # while the window and its OpenGL context still exist.
        with ExitStack() as resources:
            capacity = setup.max_particles if setup is not None else 1
            self.particles = resources.enter_context(
                SharedBuffer(ctx, capacity, wp.vec3, self.device)
            )
            # Without a liquid nothing is drawn as water; any positive volume will do.
            volume_m3 = setup.particle_volume_m3 if setup is not None else 1.0
            self.frame = WorldFrame(
                ctx,
                self.world,
                self.particles,
                particle_volume_m3=volume_m3,
                water=load_material("water"),
            )
            resources.callback(self.frame.release)
            self.hud = resources.enter_context(PerformanceHud(window.handle))
            self._install_callbacks()
            resources.callback(self._remove_callbacks)
            self._resources = resources.pop_all()

    def _install_callbacks(self) -> None:
        handle = self.window.handle
        mode = self.mode

        def on_key(_window: Any, key: int, _scancode: int, action: int, _mods: int) -> None:
            mode.input.on_key(key, action)

        def on_cursor(_window: Any, x: float, y: float) -> None:
            mode.input.on_mouse_move(x, y)

        def on_button(_window: Any, _button: int, action: int, _mods: int) -> None:
            mode.on_mouse_button(pressed=action == glfw.PRESS)

        self.callbacks = (on_key, on_cursor, on_button)
        glfw.set_key_callback(handle, on_key)
        glfw.set_cursor_pos_callback(handle, on_cursor)
        glfw.set_mouse_button_callback(handle, on_button)

    def _remove_callbacks(self) -> None:
        handle = self.window.handle
        # pyGLFW accepts None to remove a callback; its stubs do not say so.
        glfw.set_key_callback(handle, None)  # pyright: ignore[reportArgumentType]
        glfw.set_cursor_pos_callback(handle, None)  # pyright: ignore[reportArgumentType]
        glfw.set_mouse_button_callback(handle, None)  # pyright: ignore[reportArgumentType]

    def _particle_count(self) -> int:
        return int(self.liquid.diagnostics()["particles"]) if self.liquid is not None else 0

    def hud_notes(self) -> list[str]:
        """Extra HUD lines: an alert if the world froze, the controls, the map and its
        water, and why interop copies, if it does."""
        notes = []
        if self.simulation.frozen_because:
            notes.append(f"MONDE ARRÊTÉ : {self.simulation.frozen_because}")
        notes += [CONTROLS_HINT, f"{self.world.name} : {self._particle_count():,} particules d'eau"]
        if self.particles.fallback_reason:
            notes.append(f"host copy because: {self.particles.fallback_reason}")
        return notes

    def poll_input(self) -> None:
        self.window.poll_events()

    def update(self, frame_dt_s: float) -> None:
        self.mode.update(frame_dt_s)
        captured = self.mode.input.mouse_captured
        if captured != self._mouse_captured:
            self.window.set_mouse_captured(captured)
            self._mouse_captured = captured
        self.hud.visible = self.mode.hud_visible

    def step_simulation(self, dt_s: float) -> float:
        with self.cuda_timer.stage("simulation (CUDA)"):
            return self.simulation.step(dt_s)

    def render(self) -> None:
        width, height = self.window.framebuffer_size_px()
        if width == 0 or height == 0:
            return  # minimised
        count = self._particle_count()
        if count and self.liquid is not None:
            with self.particles.writable(count) as positions:
                wp.copy(positions, self.liquid.fields()["positions_m"], count=count)
        self.particles_drawn = count
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
        self.frame.render(screen, view, (width, height), count=count, timer=self.gl_timer)
        lines = hud_lines(
            self.stats.summary(),
            gpu_memory(self.device),
            interop=self.particles.mode.value,
            time_scale=self.loop_settings.time_scale,
            notes=self.hud_notes(),
        )
        with self.gl_timer.stage("HUD (OpenGL)"):
            self.hud.draw(lines)
        for timer in (self.cuda_timer, self.gl_timer):
            timer.end_frame()
            for stages_s in timer.collect():
                self.stats.record_gpu(stages_s)

    def present(self) -> None:
        self.window.swap_buffers()

    def release(self) -> None:
        self._resources.close()


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


def _show_error(title: str, text: str) -> None:
    """A message box on Windows (the console of the shortcut is minimised), stderr elsewhere."""
    if sys.platform == "win32":
        mb_iconerror, mb_setforeground, mb_topmost = 0x10, 0x10000, 0x40000
        flags = mb_iconerror | mb_setforeground | mb_topmost
        ctypes.WinDLL("user32").MessageBoxW(None, text, title, flags)
    else:
        print(f"{title}: {text}", file=sys.stderr)


def main(log_path: Path = DEFAULT_LOG_PATH) -> int:
    """Run the game; on a failure, log the details and tell Matt in French."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    level_before = root.level
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        run(load_settings(default_settings_path()))
    except Exception as error:
        _log.exception("ENTROPY stopped on an error")
        _show_error(
            "ENTROPY",
            f"ENTROPY n'a pas pu démarrer ou s'est arrêté :\n\n{error}\n\nDétails : {log_path}",
        )
        return 1
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(level_before)
    return 0

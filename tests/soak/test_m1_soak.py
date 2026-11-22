"""M1 gate soak test (#58): the water tank runs for minutes without NaN, leak or crash.

`uv run pytest -m soak` on the laptop, with the GPU otherwise idle. The game runs in a
hidden window for `ENTROPY_SOAK_S` seconds (default 600) with water in the tank, as a
player would use it: the camera above the rim, and every few seconds one action at
random:
- the hose, the flood source or the eraser held for a while;
- the bucket in either shape;
- pause and resume, a slower or faster time scale, a single step;
- F5 and F9.

The amount of water is held between 480,000 and 600,000 particles: the eraser is used
when there is more, the other tools when there is less. Checkpoints' host buffers grow
by a quarter over what they hold (ADR 0024), so within that band they never grow
again. The baseline is taken after the settling minute, once the checkpoint ring is
full and every action has run once (its kernels loaded). Growth after it means a leak.
It is measured on the process, ENTROPY's Warp pool and the whole GPU, with the same
64 MiB limit and the same sampling as M0's soak.

V-LIQ-02 over 10 minutes (SPEC 9.4, ADR 0019, ADR 0028): the last `QUIET_S` of every
`QUIET_EVERY_S` start no action, so the water settles. In the last 2 s of each such
window, with no tool held and no water in flight, the liquid's own measure of its
water's volume (from the level set, as on the HUD) is averaged against its particles'
volume.

At the end:
- the world is not frozen by the NaN guard;
- every frame completed;
- the particles are finite;
- V-LIQ-02's two criteria: the first settled window's volume within 1 % of its
  particles' volume, and every later one, as a share of its particles' volume, within
  1 % of the first. (A 150 s trial read 0.9905 at first, then up to +0.4 % from it.)
"""

import math
import os
import random
import time

import glfw
import numpy as np
import pytest

from entropy.app.main import GameStages
from entropy.core.config import default_settings_path, load_settings
from entropy.core.profiler import FrameStats
from entropy.game.free_camera import CameraPose
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop
from entropy.solvers.liquid.solver import fill_box
from tests.soak.test_m0_soak import LEAK_LIMIT_BYTES, SETTLE_S, _sample

pytestmark = pytest.mark.soak

LOW, HIGH = 480_000, 600_000
QUIET_EVERY_S, QUIET_S = 30.0, 5.0
VOLUME_TOLERANCE = 0.01  # V-LIQ-02's 1 %
KINDS = {"hose", "flood", "bucket", "eraser", "time", "save", "load"}
ABOVE_THE_RIM = CameraPose((0.0, -3.2, 3.4), math.radians(90.0), math.radians(-38.0))
TIME_KEYS = (glfw.KEY_P, glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET, glfw.KEY_PERIOD)


def _tap(stages: GameStages, key: int) -> None:
    stages.mode.input.on_key(key, glfw.PRESS)
    stages.mode.input.on_key(key, glfw.RELEASE)


def test_the_water_tank_runs_for_minutes_without_nan_or_leak() -> None:
    duration_s = float(os.environ.get("ENTROPY_SOAK_S", 600))
    settings = load_settings(default_settings_path())
    rng = random.Random(1)
    stats = FrameStats()
    actions: dict[str, int] = {}
    with GameWindow(settings.window, visible=False) as window:
        stages = GameStages(window, settings, stats)
        liquid = stages.liquid
        assert liquid is not None
        liquid.add_particles(fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 1.0), 0.0625, 8, seed=3))
        loop = FrameLoop(
            stages, settings=stages.loop_settings, stats=stats, time_control=stages.time_control
        )
        stages.mode.on_mouse_button(pressed=True)  # captures the mouse: clicks use tools
        loop.reset_clock()
        start = time.perf_counter()
        next_action_s, release_at_s = 1.0, -1.0
        baseline = None
        frames = 0
        quiet: list[float] = []  # volume over particles' volume, this quiet window
        settled_volumes: list[float] = []
        try:
            while time.perf_counter() - start < duration_s:
                now_s = time.perf_counter() - start
                stages.mode.pose = ABOVE_THE_RIM
                if release_at_s >= 0.0 and now_s >= release_at_s:
                    stages.mode.input.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.RELEASE)
                    release_at_s = -1.0
                into_quiet_s = now_s % QUIET_EVERY_S - (QUIET_EVERY_S - QUIET_S)
                if now_s >= next_action_s and release_at_s < 0.0 and into_quiet_s < 0.0:
                    count = liquid.particle_count()
                    if stages.time_control.paused and rng.random() < 0.7:
                        name = "resume"
                        _tap(stages, glfw.KEY_P)
                    elif count > HIGH:
                        name = "eraser"
                    elif count < LOW:
                        name = rng.choice(["hose", "flood", "bucket"])
                    elif KINDS - set(actions):  # each action once, before the baseline
                        name = sorted(KINDS - set(actions))[0]
                    else:
                        name = rng.choice(
                            ["hose", "flood", "bucket", "eraser", "time", "time", "save", "load"]
                        )
                    if name in ("hose", "bucket", "flood", "eraser"):
                        number = {"hose": 1, "bucket": 2, "flood": 3, "eraser": 4}[name]
                        _tap(stages, glfw.KEY_0 + number)
                        if name == "bucket" and rng.random() < 0.3:
                            stages.mode.input.on_mouse_button(glfw.MOUSE_BUTTON_RIGHT, glfw.PRESS)
                        stages.mode.input.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS)
                        release_at_s = now_s + rng.uniform(0.2, 1.5)
                    elif name == "time":
                        _tap(stages, rng.choice(TIME_KEYS))
                    elif name == "save":
                        _tap(stages, glfw.KEY_F5)
                    elif name == "load":
                        _tap(stages, glfw.KEY_F9)
                    actions[name] = actions.get(name, 0) + 1
                    next_action_s = now_s + rng.uniform(1.0, 4.0)
                loop.run_frame()
                frames += 1
                if into_quiet_s >= QUIET_S - 2.0 and release_at_s < 0.0 and not stages.toolbox.jet:
                    d = liquid.diagnostics()
                    quiet.append(
                        d["water_volume_m3"] / (d["particles"] * stages.particle_volume_m3)
                    )
                elif into_quiet_s < 0.0 and quiet:
                    settled_volumes.append(float(np.mean(quiet)))
                    quiet = []
                ring = stages.rewind.ring
                settled = (
                    now_s >= min(SETTLE_S, duration_s / 2)
                    and len(ring.entries) == ring.entries.maxlen
                    and set(actions) >= KINDS
                )
                if baseline is None and settled:
                    baseline = _sample(loop, stages.device)
            final = _sample(loop, stages.device)
            elapsed_s = time.perf_counter() - start
            frozen = stages.simulation.frozen_by
            particles = liquid.particle_count()
            finite = bool(np.isfinite(liquid.particle_positions()).all())
            overflow_m3 = stages.toolbox.overflow_m3
        finally:
            stages.release()

    assert baseline is not None
    host, warp, device = (after - before for after, before in zip(final, baseline, strict=True))
    summary = stats.summary()
    print(
        f"\nM1 soak: {frames} frames in {elapsed_s:.0f} s ({frames / elapsed_s:.1f} fps), "
        f"{particles:,} particles at the end, actions {actions}, overflow {overflow_m3:.3f} m³, "
        f"world speed {summary.world_speed:.2f}; growth process {host / 2**20:.1f} MiB, Warp "
        f"pool {warp / 2**20:.1f} MiB, GPU-wide {device / 2**20:.1f} MiB; settled volume "
        f"over particles' volume {[round(v, 4) for v in settled_volumes]}"
    )
    assert frames > 0
    assert frozen is None, f"the NaN guard froze the world: {frozen}"
    assert finite
    assert host < LEAK_LIMIT_BYTES
    assert warp < LEAK_LIMIT_BYTES
    assert device < LEAK_LIMIT_BYTES
    assert len(settled_volumes) >= int(duration_s // QUIET_EVERY_S) - 1
    first = settled_volumes[0]
    assert abs(first - 1.0) <= VOLUME_TOLERANCE
    assert max(abs(v / first - 1.0) for v in settled_volumes) <= VOLUME_TOLERANCE

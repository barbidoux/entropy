"""M0 gate soak test: the assembled game runs for minutes without leaking or failing.

`uv run pytest -m soak` on the laptop, with the GPU otherwise idle. The game
runs in a hidden window with random keys and mouse motion for `ENTROPY_SOAK_S`
seconds (default 600). After a settling minute, the process's own memory, the
memory in ENTROPY's Warp pool and the video memory used on the whole GPU must
not grow by more than 64 MiB; each figure is the minimum of several samples,
so a passing spike elsewhere on the GPU does not count. Every frame must
complete.
"""

import os
import random
import time

import glfw
import pytest

from entropy.app.main import GameStages
from entropy.core.config import default_settings_path, load_settings
from entropy.core.profiler import FrameStats, gpu_memory, host_memory_bytes
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop

pytestmark = pytest.mark.soak

KEYS = (glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D, glfw.KEY_SPACE, glfw.KEY_LEFT_CONTROL)
LEAK_LIMIT_BYTES = 64 * 2**20
SETTLE_S = 60.0
SAMPLES = 5


def _memory(device: str) -> tuple[int, int, int]:
    """(process bytes, Warp pool bytes, GPU-wide used bytes); 0 where unknown."""
    gpu = gpu_memory(device)
    return (
        host_memory_bytes() or 0,
        (gpu.warp_bytes or 0) if gpu else 0,
        gpu.device_used_bytes if gpu else 0,
    )


def _sample(loop: FrameLoop, device: str) -> tuple[int, int, int]:
    """Minimum of each figure over a few frames."""
    samples = []
    for _ in range(SAMPLES):
        loop.run_frame()
        samples.append(_memory(device))
    return tuple(min(column) for column in zip(*samples, strict=True))  # type: ignore[return-value]


def test_the_game_runs_for_minutes_without_leaking() -> None:
    duration_s = float(os.environ.get("ENTROPY_SOAK_S", 600))
    settings = load_settings(default_settings_path())
    rng = random.Random(0)
    stats = FrameStats()
    with GameWindow(settings.window, visible=False) as window:
        stages = GameStages(window, settings, stats)
        loop = FrameLoop(stages, stats=stats)
        stages.mode.on_mouse_button(pressed=True)
        loop.reset_clock()
        start = time.perf_counter()
        baseline = None
        frames = 0
        try:
            while time.perf_counter() - start < duration_s:
                if rng.random() < 0.05:
                    key = rng.choice(KEYS)
                    stages.mode.input.on_key(key, rng.choice((glfw.PRESS, glfw.RELEASE)))
                if rng.random() < 0.01:
                    stages.mode.input.on_key(glfw.KEY_F3, glfw.PRESS)
                    stages.mode.input.on_key(glfw.KEY_F3, glfw.RELEASE)
                stages.mode.input.on_mouse_move(rng.uniform(0, 1600), rng.uniform(0, 900))
                loop.run_frame()
                frames += 1
                settled = time.perf_counter() - start >= min(SETTLE_S, duration_s / 2)
                if baseline is None and settled:
                    baseline = _sample(loop, stages.device)
            final = _sample(loop, stages.device)
            elapsed_s = time.perf_counter() - start
        finally:
            stages.release()

    assert baseline is not None
    host, warp, device = (after - before for after, before in zip(final, baseline, strict=True))
    print(
        f"\nsoak: {frames} frames in {elapsed_s:.0f} s ({frames / elapsed_s:.1f} fps); growth "
        f"process {host / 2**20:.1f} MiB, Warp pool {warp / 2**20:.1f} MiB, "
        f"GPU-wide {device / 2**20:.1f} MiB"
    )
    assert frames > 0
    assert host < LEAK_LIMIT_BYTES
    assert warp < LEAK_LIMIT_BYTES
    assert device < LEAK_LIMIT_BYTES

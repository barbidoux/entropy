"""M0 gate soak test: the assembled game runs for minutes without leaking or failing.

`uv run pytest -m soak` on the laptop. The game runs in a hidden window with
random keys and mouse motion for `ENTROPY_SOAK_S` seconds (default 600). After
a settling minute, video memory used on the GPU and the process's own memory
must not grow by more than a few tens of megabytes, and every frame must
complete.
"""

import ctypes
import os
import random
import sys
import time

import glfw
import pytest

from entropy.app.main import GameStages
from entropy.core.config import default_settings_path, load_settings
from entropy.core.profiler import FrameStats, gpu_memory
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop

pytestmark = pytest.mark.soak

KEYS = (glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D, glfw.KEY_SPACE, glfw.KEY_LEFT_CONTROL)
LEAK_LIMIT_BYTES = 64 * 2**20
SETTLE_S = 60.0


class _ProcessMemory(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set", ctypes.c_size_t),
        ("working_set", ctypes.c_size_t),
        ("quota_peak_paged_pool", ctypes.c_size_t),
        ("quota_paged_pool", ctypes.c_size_t),
        ("quota_peak_non_paged_pool", ctypes.c_size_t),
        ("quota_non_paged_pool", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    )


def _private_bytes() -> int:
    """Committed memory of this process (Windows); 0 elsewhere."""
    if sys.platform != "win32":
        return 0
    counters = _ProcessMemory()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.pagefile_usage)


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
                if baseline is None and time.perf_counter() - start >= min(
                    SETTLE_S, duration_s / 2
                ):
                    memory = gpu_memory(stages.device)
                    baseline = (memory.device_used_bytes if memory else 0, _private_bytes())
            memory = gpu_memory(stages.device)
            final = (memory.device_used_bytes if memory else 0, _private_bytes())
        finally:
            stages.release()

    assert baseline is not None
    vram_growth, host_growth = final[0] - baseline[0], final[1] - baseline[1]
    print(
        f"\nsoak: {frames} frames in {duration_s:.0f} s ({frames / duration_s:.1f} fps), "
        f"VRAM growth {vram_growth / 2**20:.1f} MiB, host growth {host_growth / 2**20:.1f} MiB"
    )
    assert frames > 0
    assert vram_growth < LEAK_LIMIT_BYTES
    assert host_growth < LEAK_LIMIT_BYTES

"""M1 measurement: does one hitch leave the game at 30 fps with vertical sync (#36)?

On a 60 Hz panel with vsync, a frame that misses a refresh could wait for the next one,
and a loop that then stays a refresh behind would run at 30 fps for good: it would
need hysteresis. This runs the frame loop (its 60 fps cap, ADR 0008) in a visible
window (vsync needs one) with a steady frame's work and one long frame, and counts the
frames over 25 ms after it. The window opens for about 7 s.

Measured on the laptop (2026-11-22, windowed, Windows' compositor): no frame over 25 ms
after a 25 or 40 ms hitch, with 8 to 17.5 ms of work per frame; a 17.5 ms frame runs at
17.5 ms, not 33 ms. So no hysteresis is added.
"""

import time

import numpy as np
import pytest

from bench.harness import BenchSession
from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop, LoopSettings

FRAMES = 400
HITCH_FRAME = 120


def _busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class _Stages:
    """The frame loop's stages: `work_s` of busy time a frame, one frame of `hitch_s`."""

    def __init__(self, window: GameWindow, work_s: float, hitch_s: float) -> None:
        self.window, self.work_s, self.hitch_s = window, work_s, hitch_s
        self.frame = 0

    def poll_input(self) -> None:
        self.window.poll_events()

    def update(self, frame_dt_s: float) -> None:
        pass

    def step_simulation(self, dt_s: float) -> float:
        _busy(self.hitch_s if self.frame == HITCH_FRAME else self.work_s)
        self.frame += 1
        return dt_s

    def render(self) -> None:
        self.window.ctx.clear(0.1, 0.1, 0.1)

    def present(self) -> None:
        self.window.swap_buffers()


@pytest.mark.parametrize(("work_ms", "hitch_ms"), [(8.0, 25.0), (15.0, 40.0)])
def test_one_hitch_does_not_leave_the_game_at_30_fps(
    bench_session: BenchSession, work_ms: float, hitch_ms: float
) -> None:
    settings = WindowSettings(width_px=640, height_px=360, title="frame pacing", vsync=True)
    name = f"frame_pacing_{work_ms:g}ms_hitch_{hitch_ms:g}ms"
    with GameWindow(settings) as window:
        loop = FrameLoop(_Stages(window, work_ms / 1e3, hitch_ms / 1e3), settings=LoopSettings())
        loop.reset_clock()
        bench_session.start(name)
        frames_ms = np.array([loop.run_frame().frame_s * 1e3 for _ in range(FRAMES)][1:])
    before, after = frames_ms[40 : HITCH_FRAME - 2], frames_ms[HITCH_FRAME + 3 :]
    metrics = {
        "work_ms": work_ms,
        "hitch_ms": hitch_ms,
        "median_before_ms": float(np.median(before)),
        "median_after_ms": float(np.median(after)),
        "frames_after": len(after),
        "frames_after_over_25_ms": int((after > 25.0).sum()),
        "around_the_hitch_ms": [
            round(float(f), 2) for f in frames_ms[HITCH_FRAME - 4 : HITCH_FRAME + 8]
        ],
    }
    path = bench_session.record(name, metrics)
    print(
        f"\n{name}: median {metrics['median_before_ms']:.2f} ms before, "
        f"{metrics['median_after_ms']:.2f} ms after, {metrics['frames_after_over_25_ms']} of "
        f"{len(after)} frames over 25 ms after the hitch -> {path}"
    )
    assert metrics["frames_after_over_25_ms"] == 0

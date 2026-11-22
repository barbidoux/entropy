"""M1 measurement: does one hitch leave the game at 30 fps with vertical sync (#36)?

With vsync on a 60 Hz display, a frame that misses a refresh could wait for the next one,
and a loop that then stays a refresh behind would run at 30 fps for good: it would need
hysteresis. This runs the frame loop (its 60 fps cap, ADR 0008) in a visible window
(vsync needs one), with a steady frame's work and one long frame, and looks at the
frames after it. The work is either a CPU busy-wait or a GPU fragment shader timed to
take that long; the window opens for about 7 s per case.

A game stuck a refresh behind shows as a sustained run of frames over 25 ms, so the
check is on runs and on the median, not on single frames (any unrelated stall makes
one). The display's mode and the GL renderer are recorded with each result: the
conclusion holds for what was measured.
"""

import time

import glfw
import moderngl
import numpy as np
import pytest

from bench.harness import BenchSession
from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop, LoopSettings

FRAMES = 400
HITCH_FRAME = 120
AFTER = HITCH_FRAME + 8
"""Frames before this are excluded: with GPU work the hitch shows up to 3 frames late."""
RUN = 5
"""Consecutive frames over 25 ms that would mean the loop stayed a refresh behind."""

_VERTEX = """
#version 450 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""
_BURN = """
#version 450 core
uniform int iterations;
out vec4 colour;
void main() {
    float x = gl_FragCoord.x * 1e-3;
    for (int i = 0; i < iterations; ++i) x = fract(x * 1.0001 + 0.37);
    colour = vec4(x, x, x, 1.0);
}
"""


def _busy(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class _GpuWork:
    """A full-screen pass whose fragment loop is sized to take given times, each measured
    near that time with a GL timer query (#77's re-review: sized from a 0.04 ms pass, a
    17.5 ms pass took 16.7 ms and a 40 ms one 32 to 39 ms)."""

    def __init__(self, ctx: moderngl.Context, *times_ms: float) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=_VERTEX, fragment_shader=_BURN)
        self.vao = ctx.vertex_array(self.program, [])
        self.iterations: dict[float, int] = {}
        self.measured_ms: dict[float, float] = {}
        for ms in times_ms:
            self._size(ms)

    def _time(self, iterations: int) -> float:
        """Median GPU time of three passes (ms)."""
        self.program["iterations"].value = iterations  # type: ignore[union-attr]
        times = []
        for _ in range(3):
            query = self.ctx.query(time=True)
            with query:
                self.vao.render(moderngl.TRIANGLES, vertices=3)
            self.ctx.finish()
            times.append(query.elapsed / 1e6)
        return float(np.median(times))

    def _size(self, ms: float) -> None:
        n = 400
        took = self._time(n)
        while took < 2.0:
            n *= 2
            took = self._time(n)
        for _ in range(2):
            n = max(1, round(n * ms / took))
            took = self._time(n)
        self.iterations[ms], self.measured_ms[ms] = n, took

    def run(self, ms: float) -> None:
        self.program["iterations"].value = self.iterations[ms]  # type: ignore[union-attr]
        self.vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        self.vao.release()
        self.program.release()


class _Stages:
    """The frame loop's stages: `work_ms` a frame on the CPU or the GPU, one of `hitch_ms`."""

    def __init__(self, window: GameWindow, work_ms: float, hitch_ms: float, gpu: _GpuWork | None):
        self.window, self.work_ms, self.hitch_ms, self.gpu = window, work_ms, hitch_ms, gpu
        self.frame = 0

    def poll_input(self) -> None:
        self.window.poll_events()

    def update(self, frame_dt_s: float) -> None:
        pass

    def step_simulation(self, dt_s: float) -> float:
        ms = self.hitch_ms if self.frame == HITCH_FRAME else self.work_ms
        if self.gpu is None:
            _busy(ms / 1e3)
        else:
            self.gpu.run(ms)
        self.frame += 1
        return dt_s

    def render(self) -> None:
        if self.gpu is None:
            self.window.ctx.clear(0.1, 0.1, 0.1)

    def present(self) -> None:
        self.window.swap_buffers()


def _longest_run(over: np.ndarray) -> int:
    longest = current = 0
    for flag in over:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


@pytest.mark.parametrize("where", ["cpu", "gpu"])
@pytest.mark.parametrize(("work_ms", "hitch_ms"), [(8.0, 25.0), (15.0, 40.0), (17.5, 25.0)])
def test_one_hitch_does_not_leave_the_game_at_30_fps(
    bench_session: BenchSession, work_ms: float, hitch_ms: float, where: str
) -> None:
    settings = WindowSettings(width_px=640, height_px=360, title="frame pacing", vsync=True)
    name = f"frame_pacing_{where}_{work_ms:g}ms_hitch_{hitch_ms:g}ms"
    with GameWindow(settings) as window:
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        display = {
            "primary_mode": f"{mode.size.width}x{mode.size.height} at {mode.refresh_rate} Hz",
            "gl_renderer": window.ctx.info["GL_RENDERER"],
        }
        gpu = _GpuWork(window.ctx, work_ms, hitch_ms) if where == "gpu" else None
        if gpu is not None:
            display["gpu_work_measured_ms"] = round(gpu.measured_ms[work_ms], 2)
            display["gpu_hitch_measured_ms"] = round(gpu.measured_ms[hitch_ms], 2)
        try:
            loop = FrameLoop(_Stages(window, work_ms, hitch_ms, gpu), settings=LoopSettings())
            loop.reset_clock()
            bench_session.start(name)
            frames_ms = np.array([loop.run_frame().frame_s * 1e3 for _ in range(FRAMES)][1:])
        finally:
            if gpu is not None:
                gpu.release()
    before, after = frames_ms[40 : HITCH_FRAME - 2], frames_ms[AFTER:]
    over = after > 25.0
    metrics = {
        "work": where,
        "work_ms": work_ms,
        "hitch_ms": hitch_ms,
        **display,
        "median_before_ms": float(np.median(before)),
        "median_after_ms": float(np.median(after)),
        "max_after_ms": float(after.max()),
        "frames_after": len(after),
        "frames_after_over_25_ms": int(over.sum()),
        "longest_run_over_25_ms": _longest_run(over),
        "over_25_ms_at": [int(i) + AFTER for i in np.nonzero(over)[0]],
        "around_the_hitch_ms": [
            round(float(f), 2) for f in frames_ms[HITCH_FRAME - 4 : HITCH_FRAME + 8]
        ],
    }
    path = bench_session.record(name, metrics)
    print(
        f"\n{name} on {display['primary_mode']}: median {metrics['median_before_ms']:.2f} ms "
        f"before, {metrics['median_after_ms']:.2f} ms after, longest run over 25 ms "
        f"{metrics['longest_run_over_25_ms']} -> {path}"
    )
    assert metrics["median_after_ms"] <= metrics["median_before_ms"] + 1.0
    assert metrics["longest_run_over_25_ms"] < RUN

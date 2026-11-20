"""M0 measurement: cost of handing 1 million particles to OpenGL each frame (issue #10).

A kernel writes 1 million positions (vec3, 12 MB) into an OpenGL buffer, either
through CUDA-OpenGL interop (map, write, unmap) or through a host copy (write a
Warp array, copy it to host memory, upload it). The time is wall time per
frame on the Python thread, from the start of the write until both CUDA and
OpenGL have finished it (`wp.synchronize_device` and `ctx.finish`).
"""

import time

import moderngl
import warp as wp

from bench.harness import BenchSession, summarize
from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.render.interop import InteropMode, SharedBuffer

DEVICE = "cuda:0"
PARTICLES = 1_000_000
REPEATS = 100


@wp.kernel
def _positions(points: wp.array(dtype=wp.vec3), t: float):
    i = wp.tid()
    x = float(i) * 1e-6  # pyright: ignore[reportArgumentType] (1D tid is an int)
    points[i] = wp.vec3(wp.sin(x + t), wp.cos(x + t), x)


def _frame_seconds(shared: SharedBuffer, ctx: moderngl.Context) -> list[float]:
    samples = []
    for frame in range(REPEATS + 3):
        wp.synchronize_device(DEVICE)
        begin = time.perf_counter()
        with shared.writable() as points:
            wp.launch(_positions, dim=PARTICLES, inputs=[points, float(frame)], device=DEVICE)
        wp.synchronize_device(DEVICE)
        ctx.finish()
        if frame >= 3:  # the first frames load the kernel and map the buffer
            samples.append(time.perf_counter() - begin)
    return samples


def test_interop_cost_for_a_million_particles(bench_session: BenchSession) -> None:
    settings = WindowSettings(width_px=64, height_px=64, vsync=False)
    with (
        GameWindow(settings, visible=False) as window,
        SharedBuffer(window.ctx, PARTICLES, wp.vec3, DEVICE) as zero_copy,
        SharedBuffer(
            window.ctx, PARTICLES, wp.vec3, DEVICE, mode=InteropMode.HOST_COPY
        ) as host_copy,
    ):
        assert zero_copy.mode is InteropMode.ZERO_COPY, zero_copy.fallback_reason

        bench_session.start("interop")
        zero = summarize(_frame_seconds(zero_copy, window.ctx))
        host = summarize(_frame_seconds(host_copy, window.ctx))

    metrics = {
        "particles": PARTICLES,
        "bytes": PARTICLES * 12,
        "zero_copy": zero.as_ms(),
        "host_copy": host.as_ms(),
        "host_copy_over_zero_copy": host.median_s / zero.median_s,
    }
    path = bench_session.record("interop", metrics)
    print(
        f"\ninterop, 1 M particles: zero-copy {zero.median_s * 1e3:.3f} ms, "
        f"host copy {host.median_s * 1e3:.3f} ms per frame -> {path}"
    )
    assert zero.median_s < host.median_s

"""Frame statistics and GPU memory for the performance HUD (SPEC 4.7).

Stage times are CPU wall times on the main thread. GPU work is enqueued
asynchronously, so its cost shows up wherever the CPU first waits for it (the
CUDA-OpenGL map, or presenting the frame), not in the stage that enqueued it;
GPU timing per solver comes with the solvers (issue #29).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class FrameSummary:
    """Statistics over the recent frames; times in milliseconds."""

    frames: int
    fps: float
    frame_ms_mean: float
    frame_ms_worst: float
    stages_ms_mean: Mapping[str, float]
    world_speed: float = 0.0
    """Simulated seconds per real second over the window (the effective time scale).

    It includes the chosen time scale: compare it with that scale to tell a slowdown
    under load from slow motion. After a stall it reads low for one window.
    """


@dataclass(frozen=True)
class _Frame:
    duration_s: float
    stages_s: Mapping[str, float]
    advance_s: float
    real_s: float


class FrameStats:
    """Rolling window of frame and stage durations and simulated time."""

    def __init__(self, window: int = 120) -> None:
        if window < 1:
            raise ValueError("the window must hold at least one frame")
        self._frames: deque[_Frame] = deque(maxlen=window)

    def record(
        self,
        frame_s: float,
        stages_s: Mapping[str, float],
        *,
        advance_s: float = 0.0,
        real_s: float | None = None,
    ) -> None:
        """One frame, in seconds: its duration, its stages, and the simulated time it
        advanced over `real_s` of real time (by default the frame's duration)."""
        self._frames.append(
            _Frame(frame_s, dict(stages_s), advance_s, frame_s if real_s is None else real_s)
        )

    def summary(self) -> FrameSummary:
        if not self._frames:
            return FrameSummary(0, 0.0, 0.0, 0.0, {})
        durations = [frame.duration_s for frame in self._frames]
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for frame in self._frames:
            for stage, seconds in frame.stages_s.items():
                totals[stage] = totals.get(stage, 0.0) + seconds
                counts[stage] = counts.get(stage, 0) + 1
        elapsed_s = sum(durations)
        real_s = sum(frame.real_s for frame in self._frames)
        simulated_s = sum(frame.advance_s for frame in self._frames)
        return FrameSummary(
            frames=len(durations),
            fps=len(durations) / elapsed_s if elapsed_s > 0 else 0.0,
            frame_ms_mean=elapsed_s / len(durations) * 1e3,
            frame_ms_worst=max(durations) * 1e3,
            stages_ms_mean={stage: totals[stage] / counts[stage] * 1e3 for stage in totals},
            world_speed=simulated_s / real_s if real_s > 0 else 0.0,
        )


@dataclass(frozen=True)
class GpuMemory:
    """Device memory in bytes: ENTROPY's Warp pool, and the whole GPU."""

    warp_bytes: int | None
    """Bytes in use in Warp's memory pool; None when the pool is disabled."""
    device_used_bytes: int
    device_total_bytes: int
    warp_high_bytes: int | None = None
    """Most bytes Warp's pool has held since it started (for VRAM budgets)."""


def gpu_memory(device: str) -> GpuMemory | None:
    """Memory use of a CUDA device, or None for the CPU.

    `warp_bytes` counts Warp's memory pool only; OpenGL buffers show up in the
    device-wide figure, which also includes other programs and the display.
    """
    wp.init()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    pooled = wp.is_mempool_supported(dev) and wp.is_mempool_enabled(dev)
    return GpuMemory(
        warp_bytes=int(wp.get_mempool_used_mem_current(dev)) if pooled else None,
        device_used_bytes=int(dev.total_memory - dev.free_memory),
        device_total_bytes=int(dev.total_memory),
        warp_high_bytes=int(wp.get_mempool_used_mem_high(dev)) if pooled else None,
    )

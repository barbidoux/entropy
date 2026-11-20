"""Frame statistics and GPU memory for the performance HUD (SPEC 4.7)."""

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


class FrameStats:
    """Rolling window of frame and stage durations."""

    def __init__(self, window: int = 120) -> None:
        if window < 1:
            raise ValueError("the window must hold at least one frame")
        self._frames: deque[tuple[float, Mapping[str, float]]] = deque(maxlen=window)

    def record(self, frame_s: float, stages_s: Mapping[str, float]) -> None:
        """One frame: its duration and the duration of each stage, in seconds."""
        self._frames.append((frame_s, dict(stages_s)))

    def summary(self) -> FrameSummary:
        if not self._frames:
            return FrameSummary(0, 0.0, 0.0, 0.0, {})
        durations = [frame_s for frame_s, _ in self._frames]
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for _, stages in self._frames:
            for stage, seconds in stages.items():
                totals[stage] = totals.get(stage, 0.0) + seconds
                counts[stage] = counts.get(stage, 0) + 1
        elapsed_s = sum(durations)
        return FrameSummary(
            frames=len(durations),
            fps=len(durations) / elapsed_s if elapsed_s > 0 else 0.0,
            frame_ms_mean=elapsed_s / len(durations) * 1e3,
            frame_ms_worst=max(durations) * 1e3,
            stages_ms_mean={stage: totals[stage] / counts[stage] * 1e3 for stage in totals},
        )


@dataclass(frozen=True)
class GpuMemory:
    """Device memory in bytes: held by ENTROPY's Warp pool, and used on the whole GPU."""

    warp_bytes: int
    device_used_bytes: int
    device_total_bytes: int


def gpu_memory(device: str) -> GpuMemory | None:
    """Memory use of a CUDA device, or None for the CPU.

    `warp_bytes` counts Warp's memory pool only; OpenGL buffers show up in the
    device-wide figure, which also includes other programs and the display.
    """
    wp.init()
    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    return GpuMemory(
        warp_bytes=int(wp.get_mempool_used_mem_current(dev)),
        device_used_bytes=int(dev.total_memory - dev.free_memory),
        device_total_bytes=int(dev.total_memory),
    )

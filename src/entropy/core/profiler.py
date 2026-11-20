"""Frame statistics and GPU memory for the performance HUD (SPEC 4.7).

Stage times are CPU wall times on the main thread. GPU work is enqueued
asynchronously, so its cost shows up wherever the CPU first waits for it (the
CUDA-OpenGL map, or presenting the frame), not in the stage that enqueued it.
GPU time per stage is measured separately by a `StageTimer` (issue #29): CUDA
events around Warp work here, OpenGL timer queries around render passes in
`entropy.render.gl_timer`. Its results arrive a frame or more late and are
read only when the GPU has finished, so the CPU never waits for them.
"""

from __future__ import annotations

import ctypes
import sys
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

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

    It includes the chosen time scale; to tell a slowdown under load, use `time_kept`.
    """
    time_kept: float = 1.0
    """Simulated time over the simulated time the frames asked for (frame time x time
    scale), over frames that asked for some: below 1 when the 1/30 s clamp bit."""
    gpu_ms_mean: Mapping[str, float] = field(default_factory=dict)
    """GPU time per stage, from `StageTimer` results (a frame or more late)."""


@dataclass(frozen=True)
class _Frame:
    duration_s: float
    stages_s: Mapping[str, float]
    advance_s: float
    real_s: float
    requested_s: float


class FrameStats:
    """Rolling window of frame and stage durations and simulated time."""

    def __init__(self, window: int = 120) -> None:
        if window < 1:
            raise ValueError("the window must hold at least one frame")
        self._frames: deque[_Frame] = deque(maxlen=window)
        self._gpu_frames: deque[Mapping[str, float]] = deque(maxlen=window)

    def record_gpu(self, stages_s: Mapping[str, float]) -> None:
        """GPU seconds per stage of one frame, as `StageTimer.collect` gives them."""
        self._gpu_frames.append(dict(stages_s))

    def record(
        self,
        frame_s: float,
        stages_s: Mapping[str, float],
        *,
        advance_s: float = 0.0,
        real_s: float | None = None,
        requested_s: float | None = None,
    ) -> None:
        """One frame, in seconds: its duration, its stages, and the simulated time it
        advanced over `real_s` of real time (by default the frame's duration), out of
        `requested_s` it asked for (by default all of it)."""
        self._frames.append(
            _Frame(
                frame_s,
                dict(stages_s),
                advance_s,
                frame_s if real_s is None else real_s,
                advance_s if requested_s is None else requested_s,
            )
        )

    def summary(self) -> FrameSummary:
        if not self._frames:
            return FrameSummary(0, 0.0, 0.0, 0.0, {})
        durations = [frame.duration_s for frame in self._frames]
        elapsed_s = sum(durations)
        real_s = sum(frame.real_s for frame in self._frames)
        simulated_s = sum(frame.advance_s for frame in self._frames)
        asked = [frame for frame in self._frames if frame.requested_s > 0]
        requested_s = sum(frame.requested_s for frame in asked)
        kept_s = sum(frame.advance_s for frame in asked)
        return FrameSummary(
            frames=len(durations),
            fps=len(durations) / elapsed_s if elapsed_s > 0 else 0.0,
            frame_ms_mean=elapsed_s / len(durations) * 1e3,
            frame_ms_worst=max(durations) * 1e3,
            stages_ms_mean=_mean_ms(frame.stages_s for frame in self._frames),
            world_speed=simulated_s / real_s if real_s > 0 else 0.0,
            time_kept=kept_s / requested_s if requested_s > 0 else 1.0,
            gpu_ms_mean=_mean_ms(self._gpu_frames),
        )


def _mean_ms(frames: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Mean milliseconds of each stage over the frames that have it."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for stages_s in frames:
        for stage, seconds in stages_s.items():
            totals[stage] = totals.get(stage, 0.0) + seconds
            counts[stage] = counts.get(stage, 0) + 1
    return {stage: totals[stage] / counts[stage] * 1e3 for stage in totals}


class TimerBackend(Protocol):
    """Starts and ends GPU time spans and reads them once the GPU has finished."""

    def begin(self) -> Any: ...

    def end(self, token: Any) -> None: ...

    def ready(self, token: Any, frames_ago: int) -> bool:
        """Whether the span's result can be read without waiting; `frames_ago` counts the
        frames ended since the span's own frame ended."""
        ...

    def take_elapsed_s(self, token: Any) -> float:
        """The span's GPU seconds; the backend may reuse the token afterwards."""
        ...


class StageTimer:
    """GPU time of named stages per frame (SPEC 4.7), read only once the GPU is done.

    Stages with the same name in one frame add up (a solver's substeps). With no
    backend (no GPU) nothing is timed.
    """

    def __init__(self, backend: TimerBackend | None) -> None:
        self._backend = backend
        self._current: list[tuple[str, Any]] = []
        self._pending: deque[tuple[int, list[tuple[str, Any]]]] = deque()
        self._frames_ended = 0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._backend is None:
            yield
            return
        token = self._backend.begin()
        try:
            yield
        finally:
            self._backend.end(token)
            self._current.append((name, token))

    def end_frame(self) -> None:
        if self._current:
            self._pending.append((self._frames_ended, self._current))
            self._current = []
        self._frames_ended += 1

    def collect(self) -> list[dict[str, float]]:
        """GPU seconds per stage of every frame the GPU has finished, oldest first."""
        backend = self._backend
        finished: list[dict[str, float]] = []
        while backend is not None and self._pending:
            frame, spans = self._pending[0]
            frames_ago = self._frames_ended - frame - 1
            if not all(backend.ready(token, frames_ago) for _, token in spans):
                break
            self._pending.popleft()
            times: dict[str, float] = {}
            for name, token in spans:
                times[name] = times.get(name, 0.0) + backend.take_elapsed_s(token)
            finished.append(times)
        return finished


@dataclass
class _CudaSpan:
    start: wp.Event
    end: wp.Event | None = None


class CudaEvents:
    """CUDA events with timing on the device's current stream; reused once read."""

    def __init__(self, device: str) -> None:
        self._device = wp.get_device(device)
        self._free: list[wp.Event] = []

    def _event(self) -> wp.Event:
        return self._free.pop() if self._free else wp.Event(self._device, enable_timing=True)

    def begin(self) -> _CudaSpan:
        span = _CudaSpan(self._event())
        self._device.stream.record_event(span.start)
        return span

    def end(self, token: _CudaSpan) -> None:
        token.end = self._event()
        self._device.stream.record_event(token.end)

    def ready(self, token: _CudaSpan, frames_ago: int) -> bool:
        return token.end is not None and token.end.is_complete

    def take_elapsed_s(self, token: _CudaSpan) -> float:
        assert token.end is not None
        elapsed_ms = wp.get_event_elapsed_time(token.start, token.end, synchronize=False)
        self._free += [token.start, token.end]
        return elapsed_ms / 1e3


def cuda_stage_timer(device: str) -> StageTimer:
    """A `StageTimer` on CUDA events for `device`; on the CPU it times nothing."""
    wp.init()
    return StageTimer(CudaEvents(device) if wp.get_device(device).is_cuda else None)


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


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_uint32),
        ("page_fault_count", ctypes.c_uint32),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    )


def host_memory_bytes() -> int | None:
    """Memory this process has committed (its private bytes), on Windows; None elsewhere.

    SPEC 3 budgets host RAM at 24 GB; the soak test watches this for leaks.
    """
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = kernel32.K32GetProcessMemoryInfo
    get_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    )
    get_info.restype = wintypes.BOOL
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    return int(counters.pagefile_usage)

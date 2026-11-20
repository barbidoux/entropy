"""Benchmark harness: conditions, warm-up, timing and result files.

SPEC section 3: budgets are measured on the laptop, on mains power, in
performance power mode, after a 5-minute warm-up, so thermal throttling is in
the numbers. A run that misses any of these still records its results, but
under `out/benchmarks/` (ignored by git) and marked unofficial; only official
runs are written to `docs/benchmarks/results/`.
"""

from __future__ import annotations

import ctypes
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

WARMUP_REQUIRED_S = 300.0
"""Warm-up before official measurements, in seconds (SPEC 3: 5 minutes)."""


@dataclass(frozen=True)
class Summary:
    """Statistics of repeated timings, in seconds."""

    n: int
    median_s: float
    p10_s: float
    p90_s: float
    min_s: float
    max_s: float

    def as_ms(self) -> dict[str, float]:
        """The same statistics in milliseconds, for reports."""
        return {
            "median_ms": self.median_s * 1e3,
            "p10_ms": self.p10_s * 1e3,
            "p90_ms": self.p90_s * 1e3,
            "min_ms": self.min_s * 1e3,
            "max_ms": self.max_s * 1e3,
            "n": self.n,
        }


def summarize(samples_s: Sequence[float]) -> Summary:
    """Median, 10th and 90th percentiles (linear interpolation) and extremes."""
    if len(samples_s) == 0:
        raise ValueError("no samples to summarize")
    values = np.asarray(samples_s, dtype=np.float64)
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    return Summary(
        n=len(values),
        median_s=float(median),
        p10_s=float(p10),
        p90_s=float(p90),
        min_s=float(values.min()),
        max_s=float(values.max()),
    )


class PowerMode(StrEnum):
    """Windows power mode, from the power plan and the power-mode overlay."""

    BEST_PERFORMANCE = "best_performance"
    HIGH_PERFORMANCE_PLAN = "high_performance_plan"
    BALANCED = "balanced"
    BETTER_PERFORMANCE = "better_performance"
    BEST_EFFICIENCY = "best_efficiency"
    UNKNOWN = "unknown"


PERFORMANCE_MODES = frozenset({PowerMode.BEST_PERFORMANCE, PowerMode.HIGH_PERFORMANCE_PLAN})

_PERFORMANCE_PLANS = {
    "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",  # High performance
    "e9a42b02-d5df-448d-aa00-03f14749eb61",  # Ultimate performance
}
_OVERLAYS = {
    "ded574b5-45a0-4f42-8737-46345c09c238": PowerMode.BEST_PERFORMANCE,
    "3af9b8d9-7c97-431d-ad78-34a8bfea439f": PowerMode.BETTER_PERFORMANCE,
    "00000000-0000-0000-0000-000000000000": PowerMode.BALANCED,
    "961cc777-2547-4f9d-8174-7d86181b8a7a": PowerMode.BEST_EFFICIENCY,
}


def power_mode_from_guids(*, plan: str | None, overlay: str | None) -> PowerMode:
    """Interpret the active power plan and overlay GUIDs.

    A High or Ultimate performance plan wins; otherwise the Windows 11 power-mode
    slider (the overlay) decides.
    """
    if plan is not None and plan.lower() in _PERFORMANCE_PLANS:
        return PowerMode.HIGH_PERFORMANCE_PLAN
    if overlay is None:
        return PowerMode.UNKNOWN
    return _OVERLAYS.get(overlay.lower(), PowerMode.UNKNOWN)


@dataclass(frozen=True)
class PowerState:
    on_ac: bool | None
    power_mode: PowerMode


class _Guid(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]

    def __str__(self) -> str:
        tail = bytes(self.data4).hex()
        return f"{self.data1:08x}-{self.data2:04x}-{self.data3:04x}-{tail[:4]}-{tail[4:]}"


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ac_line_status", ctypes.c_ubyte),
        ("battery_flag", ctypes.c_ubyte),
        ("battery_life_percent", ctypes.c_ubyte),
        ("system_status_flag", ctypes.c_ubyte),
        ("battery_life_time", ctypes.c_uint32),
        ("battery_full_life_time", ctypes.c_uint32),
    ]


def read_power_state() -> PowerState:
    """Mains power and power mode, read from the Windows power API.

    Returns unknown values on other systems or if a call fails.
    """
    if sys.platform != "win32":
        return PowerState(on_ac=None, power_mode=PowerMode.UNKNOWN)
    kernel32 = ctypes.WinDLL("kernel32")
    powrprof = ctypes.WinDLL("powrprof")

    status = _SystemPowerStatus()
    on_ac: bool | None = None
    if kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        on_ac = {0: False, 1: True}.get(status.ac_line_status)

    plan: str | None = None
    plan_ptr = ctypes.POINTER(_Guid)()
    if powrprof.PowerGetActiveScheme(None, ctypes.byref(plan_ptr)) == 0:
        plan = str(plan_ptr.contents)
        kernel32.LocalFree(plan_ptr)

    overlay: str | None = None
    overlay_guid = _Guid()
    get_overlay = getattr(powrprof, "PowerGetEffectiveOverlayScheme", None)
    if get_overlay is not None and get_overlay(ctypes.byref(overlay_guid)) == 0:
        overlay = str(overlay_guid)

    return PowerState(on_ac=on_ac, power_mode=power_mode_from_guids(plan=plan, overlay=overlay))


@dataclass(frozen=True)
class Conditions:
    """Conditions of a benchmark session."""

    on_ac: bool | None
    power_mode: PowerMode
    warmup_s: float

    def reasons_not_official(self) -> list[str]:
        """Why results of this session cannot be official; empty if they can."""
        reasons = []
        if self.on_ac is not True:
            reasons.append("not on mains power")
        if self.power_mode not in PERFORMANCE_MODES:
            reasons.append(f"power mode is {self.power_mode.value}, not performance")
        if self.warmup_s < WARMUP_REQUIRED_S:
            reasons.append(f"warm-up {self.warmup_s:.0f} s < {WARMUP_REQUIRED_S:.0f} s")
        return reasons

    @property
    def official(self) -> bool:
        return not self.reasons_not_official()

    def as_dict(self) -> dict[str, Any]:
        return {
            "on_ac": self.on_ac,
            "power_mode": self.power_mode.value,
            "warmup_s": self.warmup_s,
            "official": self.official,
            "reasons_not_official": self.reasons_not_official(),
        }


_NVIDIA_SMI_FIELDS = (
    ("gpu", "name", str),
    ("driver", "driver_version", str),
    ("sm_clock_mhz", "clocks.sm", float),
    ("memory_clock_mhz", "clocks.mem", float),
    ("temperature_c", "temperature.gpu", float),
    ("power_w", "power.draw", float),
    ("pstate", "pstate", str),
    ("throttle_reasons", "clocks_throttle_reasons.active", str),
)


def parse_nvidia_smi(line: str) -> dict[str, Any]:
    """Parse one CSV line of `nvidia-smi --query-gpu` (no header, no units)."""
    values = [value.strip() for value in line.split(",")]
    parsed: dict[str, Any] = {}
    for (key, _, convert), value in zip(_NVIDIA_SMI_FIELDS, values, strict=True):
        parsed[key] = None if "N/A" in value else convert(value)
    return parsed


def read_gpu_telemetry() -> dict[str, Any] | None:
    """Clocks, temperature and power of GPU 0, or None without nvidia-smi."""
    query = ",".join(name for _, name, _ in _NVIDIA_SMI_FIELDS)
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvidia_smi(output.strip().splitlines()[0])


def time_device(
    fn: Callable[[], object], *, repeats: int, device: str, warmup_calls: int = 3
) -> list[float]:
    """Duration of each of `repeats` calls of `fn`, in seconds.

    On CUDA the time is measured on the GPU with CUDA events around each call,
    so it is the GPU time of the work `fn` enqueues. On CPU it is wall time,
    with the device synchronized after each call.
    """
    wp.init()
    for _ in range(warmup_calls):
        fn()
    wp.synchronize_device(device)
    samples: list[float] = []
    if wp.get_device(device).is_cuda:
        events = [
            (wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True))
            for _ in range(repeats)
        ]
        for start, end in events:
            wp.record_event(start)
            fn()
            wp.record_event(end)
        wp.synchronize_device(device)
        samples = [wp.get_event_elapsed_time(start, end) * 1e-3 for start, end in events]
    else:
        for _ in range(repeats):
            begin = time.perf_counter()
            fn()
            wp.synchronize_device(device)
            samples.append(time.perf_counter() - begin)
    return samples


@wp.kernel
def _warm_up_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = wp.sin(x[i]) * 0.5 + y[i] * 0.5


def warm_up(
    seconds: float,
    *,
    device: str,
    clock: Callable[[], float] = time.perf_counter,
    elements: int = 1 << 26,
) -> float:
    """Keep `device` busy for at least `seconds`; return the time actually spent, in s.

    The load streams two arrays of `elements` floats (512 MiB with the default),
    which heats both the GPU and its memory.
    """
    x = wp.full(elements, 1.0, dtype=float, device=device)
    y = wp.zeros(elements, dtype=float, device=device)
    start = clock()
    now = start
    while now - start < seconds:
        for _ in range(16):
            wp.launch(_warm_up_kernel, dim=elements, inputs=[x, y], device=device)
        wp.synchronize_device(device)
        now = clock()
    return now - start


def result_path(name: str, conditions: Conditions, repo_root: Path, day: date) -> Path:
    """Where a result file goes: `docs/benchmarks/results/` if official, else `out/`."""
    folder = repo_root / "docs" / "benchmarks" / "results"
    if not conditions.official:
        folder = repo_root / "out" / "benchmarks"
    return folder / f"{day.isoformat()}-{name}.json"


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_result(
    name: str,
    metrics: Mapping[str, Any],
    conditions: Conditions,
    repo_root: Path,
    *,
    day: date | None = None,
    telemetry: Mapping[str, Any] | None = None,
) -> Path:
    """Write one benchmark result as JSON and return its path."""
    day = day or date.today()
    path = result_path(name, conditions, repo_root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "date": day.isoformat(),
        "metrics": dict(metrics),
        "conditions": conditions.as_dict(),
        "telemetry": dict(telemetry or {}),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.config.version,
            "platform": platform.platform(),
            "git_commit": _git_commit(repo_root),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@dataclass
class BenchSession:
    """One `-m perf` session: fixed conditions, telemetry before and after each benchmark."""

    conditions: Conditions
    repo_root: Path
    telemetry: Callable[[], Mapping[str, Any] | None] = read_gpu_telemetry
    day: date | None = None
    _before: dict[str, Any] = field(default_factory=dict)

    def start(self, name: str) -> None:
        """Call right before measuring `name`."""
        self._before[name] = self.telemetry()

    def record(self, name: str, metrics: Mapping[str, Any]) -> Path:
        """Write the result of `name` with telemetry from `start` and now."""
        telemetry = {"before": self._before.pop(name, None), "after": self.telemetry()}
        return write_result(
            name, metrics, self.conditions, self.repo_root, day=self.day, telemetry=telemetry
        )

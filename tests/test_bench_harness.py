"""The benchmark harness (`bench/harness.py`), tested on CPU.

The benchmarks themselves only run on the laptop (`-m perf`); the logic that
decides whether a measurement is official and how it is recorded runs here.
"""

import itertools
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from bench.harness import (
    WARMUP_REQUIRED_S,
    BenchSession,
    Conditions,
    PowerMode,
    parse_nvidia_smi,
    power_mode_from_guids,
    read_power_state,
    result_path,
    summarize,
    time_device,
    warm_up,
    write_result,
)

BALANCED_PLAN = "381b4222-f694-41f0-9685-ff5bb260df2e"
HIGH_PERFORMANCE_PLAN = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
BALANCED_OVERLAY = "00000000-0000-0000-0000-000000000000"
BEST_PERFORMANCE_OVERLAY = "ded574b5-45a0-4f42-8737-46345c09c238"


def test_summarize_reports_median_spread_and_extremes() -> None:
    summary = summarize([0.003, 0.001, 0.002, 0.004, 0.005])

    assert summary.n == 5
    assert summary.median_s == pytest.approx(0.003)
    assert summary.p10_s == pytest.approx(0.0014)
    assert summary.p90_s == pytest.approx(0.0046)
    assert (summary.min_s, summary.max_s) == (0.001, 0.005)


def test_summarize_rejects_no_samples() -> None:
    with pytest.raises(ValueError, match="no samples"):
        summarize([])


@pytest.mark.parametrize(
    ("plan", "overlay", "expected"),
    [
        (BALANCED_PLAN, BEST_PERFORMANCE_OVERLAY, PowerMode.BEST_PERFORMANCE),
        (BALANCED_PLAN, BALANCED_OVERLAY, PowerMode.BALANCED),
        (HIGH_PERFORMANCE_PLAN, BALANCED_OVERLAY, PowerMode.HIGH_PERFORMANCE_PLAN),
        (BALANCED_PLAN, "12345678-0000-0000-0000-000000000000", PowerMode.UNKNOWN),
        (None, None, PowerMode.UNKNOWN),
    ],
)
def test_power_mode_from_windows_guids(
    plan: str | None, overlay: str | None, expected: PowerMode
) -> None:
    assert power_mode_from_guids(plan=plan, overlay=overlay) is expected


@pytest.mark.parametrize(
    ("on_ac", "mode", "warmup_s", "official"),
    [
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, True),
        (True, PowerMode.HIGH_PERFORMANCE_PLAN, WARMUP_REQUIRED_S + 1, True),
        (False, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, False),
        (None, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, False),
        (True, PowerMode.BALANCED, WARMUP_REQUIRED_S, False),
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S - 1, False),
    ],
)
def test_results_are_official_only_on_mains_in_performance_mode_after_warm_up(
    on_ac: bool | None, mode: PowerMode, warmup_s: float, official: bool
) -> None:
    conditions = Conditions(on_ac=on_ac, power_mode=mode, warmup_s=warmup_s)

    assert conditions.official is official
    assert bool(conditions.reasons_not_official()) is not official


def test_warm_up_requirement_is_five_minutes() -> None:
    """SPEC section 3: budgets are measured after a 5-minute warm-up."""
    assert WARMUP_REQUIRED_S == 300.0


def test_official_results_go_to_docs_and_others_to_out(tmp_path: Path) -> None:
    official = Conditions(on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, warmup_s=300.0)
    trial = Conditions(on_ac=True, power_mode=PowerMode.BALANCED, warmup_s=10.0)
    day = date(2026, 11, 20)

    assert result_path("bandwidth", official, tmp_path, day) == (
        tmp_path / "docs" / "benchmarks" / "results" / "2026-11-20-bandwidth.json"
    )
    assert result_path("bandwidth", trial, tmp_path, day) == (
        tmp_path / "out" / "benchmarks" / "2026-11-20-bandwidth.json"
    )


def test_write_result_records_metrics_conditions_and_environment(tmp_path: Path) -> None:
    conditions = Conditions(on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, warmup_s=301.5)

    path = write_result(
        "bandwidth",
        {"copy_gb_per_s": 480.0},
        conditions,
        tmp_path,
        day=date(2026, 11, 20),
        telemetry={"before": {"temperature_c": 70.0}},
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["name"] == "bandwidth"
    assert data["date"] == "2026-11-20"
    assert data["metrics"] == {"copy_gb_per_s": 480.0}
    assert data["conditions"] == {
        "on_ac": True,
        "power_mode": "best_performance",
        "warmup_s": 301.5,
        "official": True,
        "reasons_not_official": [],
    }
    assert data["telemetry"] == {"before": {"temperature_c": 70.0}}
    assert data["environment"]["python"].startswith("3.12")
    assert "warp" in data["environment"]


def test_parse_nvidia_smi_query_line() -> None:
    line = (
        "NVIDIA GeForce RTX 4090 Laptop GPU, 596.71, 2040, 9001, 78, 145.20, P0, 0x0000000000000004"
    )

    assert parse_nvidia_smi(line) == {
        "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
        "driver": "596.71",
        "sm_clock_mhz": 2040.0,
        "memory_clock_mhz": 9001.0,
        "temperature_c": 78.0,
        "power_w": 145.2,
        "pstate": "P0",
        "throttle_reasons": "0x0000000000000004",
    }


def test_parse_nvidia_smi_keeps_unavailable_values_as_none() -> None:
    line = "GPU, 596.71, 210, 405, 46, [N/A], P8, 0x1"

    assert parse_nvidia_smi(line)["power_w"] is None


def test_time_device_returns_one_duration_per_repeat() -> None:
    calls: list[int] = []

    samples = time_device(lambda: calls.append(1), repeats=7, device="cpu", warmup_calls=2)

    assert len(samples) == 7
    assert len(calls) == 9
    assert all(sample >= 0.0 for sample in samples)


def test_warm_up_runs_until_the_requested_duration() -> None:
    ticks = itertools.count(start=0.0, step=0.1)

    elapsed = warm_up(0.35, device="cpu", clock=lambda: next(ticks), elements=1024)

    assert elapsed == pytest.approx(0.4)


def test_bench_session_records_telemetry_around_each_result(tmp_path: Path) -> None:
    readings = iter([{"temperature_c": 70.0}, {"temperature_c": 72.0}])
    session = BenchSession(
        conditions=Conditions(on_ac=True, power_mode=PowerMode.BALANCED, warmup_s=1.0),
        repo_root=tmp_path,
        telemetry=lambda: next(readings),
        day=date(2026, 11, 20),
    )

    session.start("stencil")
    path = session.record("stencil", {"ms_per_iteration": 0.5})
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path / "out" / "benchmarks"
    assert data["telemetry"] == {
        "before": {"temperature_c": 70.0},
        "after": {"temperature_c": 72.0},
    }
    assert data["conditions"]["official"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power API")
def test_read_power_state_on_windows() -> None:
    state = read_power_state()

    assert state.on_ac in (True, False)
    assert isinstance(state.power_mode, PowerMode)

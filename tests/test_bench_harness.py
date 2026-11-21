"""The benchmark harness (`bench/harness.py`), tested on CPU.

The benchmarks themselves only run on the laptop (`-m perf`); the logic that
decides whether a measurement is official and how it is recorded runs here.
"""

import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from bench.harness import (
    WARMUP_REQUIRED_S,
    BenchSession,
    Conditions,
    PowerMode,
    PowerState,
    TelemetrySampler,
    parse_nvidia_smi,
    power_mode_from_guids,
    read_gpu_telemetry,
    read_power_state,
    result_path,
    summarize,
    summarize_telemetry,
    time_device,
    time_interleaved,
    warm_up,
    write_result,
)

BALANCED_PLAN = "381b4222-f694-41f0-9685-ff5bb260df2e"
HIGH_PERFORMANCE_PLAN = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
POWER_SAVER_PLAN = "a1841308-3541-4fab-bc81-f71556f20b4a"
BALANCED_OVERLAY = "00000000-0000-0000-0000-000000000000"
BEST_PERFORMANCE_OVERLAY = "ded574b5-45a0-4f42-8737-46345c09c238"

STAMP = datetime(2026, 11, 20, 15, 45, 0)
PLUGGED_FAST = PowerState(on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, energy_saver=False)
OFFICIAL = Conditions(
    on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, warmup_s=300.0, energy_saver=False
)


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
        # The slider only applies to the Balanced plan; a stale overlay must not count.
        (POWER_SAVER_PLAN, BEST_PERFORMANCE_OVERLAY, PowerMode.UNKNOWN),
        (None, BEST_PERFORMANCE_OVERLAY, PowerMode.UNKNOWN),
        (None, None, PowerMode.UNKNOWN),
    ],
)
def test_power_mode_from_windows_guids(
    plan: str | None, overlay: str | None, expected: PowerMode
) -> None:
    assert power_mode_from_guids(plan=plan, overlay=overlay) is expected


@pytest.mark.parametrize(
    ("on_ac", "mode", "warmup_s", "energy_saver", "official"),
    [
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, False, True),
        (True, PowerMode.HIGH_PERFORMANCE_PLAN, WARMUP_REQUIRED_S + 1, False, True),
        (False, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, False, False),
        (None, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, False, False),
        (True, PowerMode.BALANCED, WARMUP_REQUIRED_S, False, False),
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S - 1, False, False),
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, True, False),
        (True, PowerMode.BEST_PERFORMANCE, WARMUP_REQUIRED_S, None, False),
    ],
)
def test_results_are_official_only_on_mains_in_performance_mode_after_warm_up(
    on_ac: bool | None, mode: PowerMode, warmup_s: float, energy_saver: bool | None, official: bool
) -> None:
    conditions = Conditions(
        on_ac=on_ac, power_mode=mode, warmup_s=warmup_s, energy_saver=energy_saver
    )

    assert conditions.official is official
    assert bool(conditions.reasons_not_official()) is not official


def test_warm_up_requirement_is_five_minutes() -> None:
    """SPEC section 3: budgets are measured after a 5-minute warm-up."""
    assert WARMUP_REQUIRED_S == 300.0


def test_a_later_power_reading_can_only_make_conditions_worse() -> None:
    unplugged = PowerState(on_ac=False, power_mode=PowerMode.BEST_PERFORMANCE, energy_saver=False)
    slower = PowerState(on_ac=True, power_mode=PowerMode.BALANCED, energy_saver=False)
    saving = PowerState(on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, energy_saver=True)

    assert OFFICIAL.after(PLUGGED_FAST).official
    assert OFFICIAL.after(unplugged).reasons_not_official() == ["not on mains power"]
    assert OFFICIAL.after(slower).power_mode is PowerMode.BALANCED
    assert not OFFICIAL.after(saving).official
    assert OFFICIAL.after(unplugged).after(PLUGGED_FAST).on_ac is False


def test_official_results_go_to_docs_and_others_to_out(tmp_path: Path) -> None:
    trial = Conditions(on_ac=True, power_mode=PowerMode.BALANCED, warmup_s=10.0)

    assert result_path("bandwidth", OFFICIAL, tmp_path, STAMP) == (
        tmp_path / "docs" / "benchmarks" / "results" / "2026-11-20-154500-bandwidth.json"
    )
    assert result_path("bandwidth", trial, tmp_path, STAMP) == (
        tmp_path / "out" / "benchmarks" / "2026-11-20-154500-bandwidth.json"
    )


def test_write_result_records_metrics_conditions_and_environment(tmp_path: Path) -> None:
    conditions = Conditions(
        on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, warmup_s=301.5, energy_saver=False
    )

    path = write_result(
        "bandwidth",
        {"copy_gb_per_s": np.float32(480.0), "repeats": np.int64(50)},
        conditions,
        tmp_path,
        stamp=STAMP,
        telemetry={"before": {"temperature_c": 70.0}},
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["name"] == "bandwidth"
    assert data["timestamp"] == "2026-11-20T15:45:00"
    assert data["metrics"] == {"copy_gb_per_s": 480.0, "repeats": 50}
    assert data["conditions"] == {
        "on_ac": True,
        "power_mode": "best_performance",
        "energy_saver": False,
        "warmup_s": 301.5,
        "official": True,
        "reasons_not_official": [],
    }
    assert data["telemetry"] == {"before": {"temperature_c": 70.0}}
    assert data["environment"]["python"].startswith("3.12")
    assert {"warp", "numpy", "git"} <= data["environment"].keys()


SMI_LINE = (
    "NVIDIA GeForce RTX 4090 Laptop GPU, 596.71, 2040, 9001, 78, 145.20, 145.00, P0, "
    "0x0000000000000004"
)


def test_parse_nvidia_smi_query_line() -> None:
    assert parse_nvidia_smi(SMI_LINE) == {
        "gpu": "NVIDIA GeForce RTX 4090 Laptop GPU",
        "driver": "596.71",
        "sm_clock_mhz": 2040.0,
        "memory_clock_mhz": 9001.0,
        "temperature_c": 78.0,
        "power_w": 145.2,
        "power_limit_w": 145.0,
        "pstate": "P0",
        "clock_event_reasons": "0x0000000000000004",
    }


@pytest.mark.parametrize("missing", ["[N/A]", "[Not Supported]", "garbage"])
def test_parse_nvidia_smi_keeps_unreadable_numbers_as_none(missing: str) -> None:
    line = f"GPU, 596.71, 210, 405, 46, {missing}, [N/A], P8, 0x1"

    parsed = parse_nvidia_smi(line)

    assert parsed["power_w"] is None
    assert parsed["power_limit_w"] is None
    assert parsed["sm_clock_mhz"] == 210.0


def test_unparseable_telemetry_is_none_rather_than_an_error() -> None:
    assert read_gpu_telemetry(query=lambda: "only, three, fields") is None
    assert read_gpu_telemetry(query=lambda: "") is None


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


def test_bench_session_rereads_power_around_each_result(tmp_path: Path) -> None:
    readings = iter([{"temperature_c": 70.0}, {"temperature_c": 72.0}])
    unplugged = PowerState(on_ac=False, power_mode=PowerMode.BEST_PERFORMANCE, energy_saver=False)
    power = iter([PLUGGED_FAST, unplugged])
    session = BenchSession(
        conditions=OFFICIAL,
        repo_root=tmp_path,
        telemetry=lambda: next(readings),
        power_state=lambda: next(power),
        sampler=lambda: TelemetrySampler(lambda: None),
        stamp=STAMP,
    )

    session.start("stencil")
    path = session.record("stencil", {"ms_per_iteration": 0.5})
    data = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path / "out" / "benchmarks"
    assert data["telemetry"] == {
        "before": {"temperature_c": 70.0},
        "during": {"samples": 0},
        "after": {"temperature_c": 72.0},
    }
    assert data["conditions"]["reasons_not_official"] == ["not on mains power"]


def test_bench_session_says_when_start_was_forgotten(tmp_path: Path) -> None:
    session = BenchSession(conditions=OFFICIAL, repo_root=tmp_path, telemetry=dict)

    with pytest.raises(RuntimeError, match=r"start\('stencil'\)"):
        session.record("stencil", {})


@pytest.mark.skipif(sys.platform != "win32", reason="Windows power API")
def test_read_power_state_on_windows() -> None:
    state = read_power_state()

    assert state.on_ac in (True, False)
    assert state.energy_saver in (True, False)
    assert isinstance(state.power_mode, PowerMode)


@wp.kernel
def _stream(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = x[i] * 2.0


@pytest.mark.gpu
def test_cuda_event_timing_scales_with_the_work() -> None:
    """Four large launches take about four times one, and match synced wall time."""
    n = 1 << 26
    x = wp.ones(n, dtype=float, device="cuda:0")
    y = wp.zeros(n, dtype=float, device="cuda:0")

    def launches(count: int) -> None:
        for _ in range(count):
            wp.launch(_stream, dim=n, inputs=[x, y], device="cuda:0")

    warm_up(1.0, device="cuda:0")  # leave idle clocks before comparing timings
    one = summarize(time_device(lambda: launches(1), repeats=20, device="cuda:0")).median_s
    four = summarize(time_device(lambda: launches(4), repeats=20, device="cuda:0")).median_s
    walls = []
    for _ in range(20):
        begin = time.perf_counter()
        launches(1)
        wp.synchronize_device("cuda:0")
        walls.append(time.perf_counter() - begin)
    wall = summarize(walls).median_s

    assert four / one == pytest.approx(4.0, rel=0.15)
    assert one == pytest.approx(wall, rel=0.3)


@pytest.mark.gpu
def test_warm_up_on_cuda_lasts_at_least_the_requested_time() -> None:
    assert warm_up(1.0, device="cuda:0") >= 1.0


def test_interleaved_timing_alternates_the_compared_passes() -> None:
    """#25: compared passes run repeat by repeat, so clock drift hits them alike."""
    order: list[str] = []
    fns = {"a": lambda: order.append("a"), "b": lambda: order.append("b")}

    samples = time_interleaved(fns, rounds=3, device="cpu", warmup_calls=1)

    assert order == ["a", "b"] + ["a", "b"] * 3  # one warm-up call each, then rounds
    assert {name: len(s) for name, s in samples.items()} == {"a": 3, "b": 3}
    assert all(t >= 0.0 for s in samples.values() for t in s)


def _smi_line(sm: float, power: float) -> str:
    return f"GPU, 581.0, {sm}, 9001, 70, {power}, 120.0, P0, 0x0"


def test_telemetry_sampled_during_a_run_is_summarised() -> None:
    lines = iter([_smi_line(2000.0, 100.0), _smi_line(1500.0, 110.0), _smi_line(1800.0, 90.0)])
    sampler = TelemetrySampler(lambda: next(lines, None))

    sampler.start()
    deadline = time.perf_counter() + 2.0
    while len(sampler.samples) < 3 and time.perf_counter() < deadline:
        time.sleep(0.01)
    samples = sampler.stop()

    summary = summarize_telemetry(samples)
    assert summary["sm_clock_mhz"] == {
        "min": 1500.0,
        "median": 1800.0,
        "max": 2000.0,
        "count": 3,
    }
    assert summary["power_w"] == {"min": 90.0, "median": 100.0, "max": 110.0, "count": 3}
    assert summary["samples"] == 3
    assert summary["pstate"] == {"P0": 3}  # text fields: how often each value was seen


def test_telemetry_summary_of_nothing_is_empty() -> None:
    assert summarize_telemetry([]) == {"samples": 0}


def test_bench_session_records_telemetry_during_the_run(tmp_path: Path) -> None:
    lines = iter([_smi_line(2100.0, 105.0)] * 50)
    session = BenchSession(
        conditions=Conditions(on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, warmup_s=300.0),
        repo_root=tmp_path,
        telemetry=lambda: None,
        power_state=lambda: PowerState(
            on_ac=True, power_mode=PowerMode.BEST_PERFORMANCE, energy_saver=False
        ),
        sampler=lambda: TelemetrySampler(lambda: next(lines, None)),
    )

    session.start("run")
    time.sleep(0.05)
    path = session.record("run", {"ms": 1.0})

    during = json.loads(path.read_text("utf-8"))["telemetry"]["during"]
    assert during["sm_clock_mhz"]["max"] == 2100.0
    assert during["samples"] >= 1


def test_an_unreadable_sample_does_not_hide_the_others() -> None:
    """#25 review: one [N/A] power reading keeps the other nine."""
    samples = [parse_nvidia_smi(_smi_line(2000.0, 100.0 + n)) for n in range(9)]
    samples.append(parse_nvidia_smi("GPU, 581.0, 2000, 9001, 70, [N/A], 120.0, P2, 0x4"))

    summary = summarize_telemetry(samples)

    assert summary["power_w"]["count"] == 9
    assert summary["power_w"]["max"] == 108.0
    assert summary["pstate"] == {"P0": 9, "P2": 1}
    assert summary["clock_event_reasons"] == {"0x0": 9, "0x4": 1}


class _FakeSampler:
    def __init__(self) -> None:
        self.stopped = 0
        self.samples: list[dict] = []

    def start(self) -> None:
        pass

    def stop(self) -> list[dict]:
        self.stopped += 1
        return []


def _session(tmp_path: Path, samplers: list[_FakeSampler]) -> BenchSession:
    def make() -> _FakeSampler:
        samplers.append(_FakeSampler())
        return samplers[-1]

    return BenchSession(
        conditions=OFFICIAL,
        repo_root=tmp_path,
        telemetry=lambda: None,
        power_state=lambda: PLUGGED_FAST,
        sampler=make,  # type: ignore[arg-type]
    )


def test_closing_a_session_stops_the_samplers_of_failed_benchmarks(tmp_path: Path) -> None:
    """#25 review: a benchmark that raises between start and record must not leave
    nvidia-smi running."""
    samplers: list[_FakeSampler] = []
    session = _session(tmp_path, samplers)

    session.start("fails")
    session.close()

    assert [s.stopped for s in samplers] == [1]


def test_starting_a_name_again_stops_its_first_sampler(tmp_path: Path) -> None:
    samplers: list[_FakeSampler] = []
    session = _session(tmp_path, samplers)

    session.start("twice")
    session.start("twice")
    session.record("twice", {"ms": 1.0})

    assert [s.stopped for s in samplers] == [1, 1]


@pytest.mark.gpu
def test_interleaved_cuda_timing_keeps_each_pass_s_own_time() -> None:
    """GPU-bound passes of 1 and 4 times the work time about 4x apart, under their own
    names, when interleaved round by round."""
    n = 1 << 22
    x = wp.zeros(n, dtype=float, device="cuda:0")
    y = wp.zeros(n, dtype=float, device="cuda:0")

    def work(times: int) -> None:
        for _ in range(times):
            wp.launch(_double, dim=n, inputs=[x, y], device="cuda:0")

    samples = time_interleaved(
        {"one": lambda: work(1), "four": lambda: work(4)}, rounds=10, device="cuda:0"
    )

    ratio = float(np.median(samples["four"]) / np.median(samples["one"]))
    assert 3.0 < ratio < 5.0


@wp.kernel
def _double(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = 2.0 * x[i]

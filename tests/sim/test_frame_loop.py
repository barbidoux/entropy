"""The frame loop (`entropy.sim.loop`): SPEC 5.9, ADR 0008 and ADR 0010.

One simulation advance per frame of the frame time times the time scale,
clamped to 1/30 s, and a 60 fps cap paced against a deadline.
"""

import math
from dataclasses import dataclass, field

import pytest

from entropy.core.profiler import FrameStats
from entropy.sim.loop import FrameLoop, LoopSettings


@dataclass
class FakeClock:
    now_s: float = 0.0
    oversleep_s: float = 0.0
    slept_s: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        """Like Windows `time.sleep`, which wakes up a little late."""
        self.slept_s.append(seconds)
        self.now_s += seconds + self.oversleep_s


@dataclass
class FakeStages:
    """Records calls; each stage takes a fixed time on the fake clock."""

    clock: FakeClock
    cost_s: dict[str, float] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    advances_s: list[float] = field(default_factory=list)
    logic_dt_s: list[float] = field(default_factory=list)

    def _run(self, name: str) -> None:
        self.calls.append(name)
        self.clock.now_s += self.cost_s.get(name, 0.0)

    def poll_input(self) -> None:
        self._run("poll_input")

    def update(self, frame_dt_s: float) -> None:
        self.logic_dt_s.append(frame_dt_s)
        self._run("update")

    def step_simulation(self, dt_s: float) -> None:
        self.advances_s.append(dt_s)
        self._run("step_simulation")

    def render(self) -> None:
        self._run("render")

    def present(self) -> None:
        self._run("present")


def _loop(
    settings: LoopSettings | None = None,
    cost_s: dict[str, float] | None = None,
    clock: FakeClock | None = None,
) -> tuple[FrameLoop, FakeStages, FakeClock]:
    clock = clock or FakeClock()
    stages = FakeStages(clock, cost_s or {})
    loop = FrameLoop(
        stages,
        clock=clock,
        sleep=clock.sleep,
        settings=settings or LoopSettings(min_frame_s=0.0),
        stats=FrameStats(),
    )
    return loop, stages, clock


def test_stages_run_in_frame_order() -> None:
    loop, stages, clock = _loop()

    clock.now_s += 1 / 60
    loop.run_frame()

    assert stages.calls == ["poll_input", "update", "step_simulation", "render", "present"]


@pytest.mark.parametrize("frame_s", [1 / 60, 1 / 45, 1 / 30])
def test_one_advance_per_frame_by_the_frame_time(frame_s: float) -> None:
    loop, stages, clock = _loop()

    clock.now_s += frame_s
    report = loop.run_frame()

    assert stages.advances_s == [pytest.approx(frame_s)]
    assert stages.logic_dt_s == [pytest.approx(frame_s)]
    assert report.advance_s == pytest.approx(frame_s)
    assert report.dropped_s == 0.0


def test_a_long_frame_advances_at_most_one_thirtieth_and_reports_the_rest() -> None:
    """SPEC 5.9: below 30 fps the world slows down instead of taking huge steps."""
    loop, stages, clock = _loop()

    clock.now_s += 0.1
    report = loop.run_frame()

    assert stages.advances_s == [pytest.approx(1 / 30)]
    assert report.dropped_s == pytest.approx(0.1 - 1 / 30)


@pytest.mark.parametrize(
    ("time_scale", "frame_s", "advance_s", "dropped_s"),
    [
        (0.5, 0.1, 1 / 30, 0.1 - 1 / 15),  # clamp after scaling: 0.05 s becomes 1/30 s
        (0.1, 0.1, 0.01, 0.0),  # slow motion below 30 fps keeps exactly 0.1x
        (2.0, 1 / 60, 1 / 30, 0.0),
        (2.0, 0.1, 1 / 30, 0.1 - 1 / 60),  # fast forward never exceeds the bound either
    ],
)
def test_the_clamp_applies_to_simulated_time(
    time_scale: float, frame_s: float, advance_s: float, dropped_s: float
) -> None:
    """ADR 0008: frame time times the time scale, clamped to 1/30 s.

    `dropped_s` is the real time the clamp did not simulate.
    """
    loop, stages, clock = _loop(LoopSettings(time_scale=time_scale, min_frame_s=0.0))

    clock.now_s += frame_s
    report = loop.run_frame()

    assert stages.advances_s == [pytest.approx(advance_s)]
    assert report.dropped_s == pytest.approx(dropped_s)


def test_time_scale_shrinks_every_advance() -> None:
    """SPEC 4.6: slow motion advances less simulated time per frame, every frame."""
    loop, stages, clock = _loop(LoopSettings(time_scale=0.1, min_frame_s=0.0))

    for _ in range(3):
        clock.now_s += 1 / 60
        loop.run_frame()

    assert stages.advances_s == [pytest.approx(1 / 600)] * 3


def test_a_zero_time_scale_pauses_the_simulation() -> None:
    loop, stages, clock = _loop(LoopSettings(time_scale=0.0, min_frame_s=0.0))

    clock.now_s += 0.1
    report = loop.run_frame()

    assert "step_simulation" not in stages.calls
    assert (report.advance_s, report.dropped_s) == (0.0, 0.0)
    assert "render" in stages.calls


def test_logic_gets_real_time_capped_against_stalls() -> None:
    """A window drag or a debugger pause must not throw the camera across the map."""
    loop, stages, clock = _loop(LoopSettings(max_logic_dt_s=0.1, min_frame_s=0.0))

    clock.now_s += 0.5
    loop.run_frame()

    assert stages.logic_dt_s == [pytest.approx(0.1)]


def test_run_starts_the_clock_when_it_starts() -> None:
    loop, stages, clock = _loop()
    clock.now_s += 5.0  # loading after construction
    answers = iter([False, True])

    loop.run(lambda: next(answers))

    assert stages.logic_dt_s == [0.0]
    assert loop.stats.summary().frame_ms_worst < 1.0


def test_frames_are_capped_at_sixty_per_second() -> None:
    loop, _, clock = _loop(LoopSettings(), cost_s={"render": 0.004, "present": 0.002})

    clock.now_s += 1 / 60
    loop.run_frame()
    report = loop.run_frame()

    assert clock.slept_s == [pytest.approx(1 / 60 - 0.006)] * 2
    assert report.frame_s == pytest.approx(1 / 60)


def test_the_cap_does_not_drift_when_sleep_wakes_up_late() -> None:
    """Windows `time.sleep` oversleeps by about 0.5 ms; frames must still average 1/60 s."""
    clock = FakeClock(oversleep_s=0.0005)
    loop, _, clock = _loop(LoopSettings(), cost_s={"render": 0.004}, clock=clock)
    loop.reset_clock()

    frames = [loop.run_frame().frame_s for _ in range(61)][1:]

    assert sum(frames) / len(frames) == pytest.approx(1 / 60, abs=1e-4)
    assert max(frames) < 1 / 60 + 0.001


def test_a_slow_frame_does_not_wait_and_does_not_rush_the_next() -> None:
    loop, stages, clock = _loop(LoopSettings(), cost_s={"render": 0.020})
    loop.reset_clock()

    loop.run_frame()
    stages.cost_s["render"] = 0.004
    loop.run_frame()

    assert clock.slept_s == [pytest.approx(1 / 60 - 0.004)]


def test_stage_times_and_world_speed_are_recorded_for_the_hud() -> None:
    cost = {"poll_input": 0.001, "update": 0.001, "step_simulation": 0.004, "render": 0.005}
    loop, _, _ = _loop(LoopSettings(time_scale=0.5), cost_s=cost)
    loop.reset_clock()

    for _ in range(3):
        loop.run_frame()
    summary = loop.stats.summary()

    assert summary.frames == 3
    assert summary.frame_ms_mean == pytest.approx(1000 / 60)
    assert summary.stages_ms_mean["logic"] == pytest.approx(2.0)
    assert summary.stages_ms_mean["rendering"] == pytest.approx(5.0)
    assert sum(summary.stages_ms_mean.values()) == pytest.approx(summary.frame_ms_mean)
    assert summary.world_speed == pytest.approx(0.5, rel=0.05)


def test_resuming_after_a_pause_is_not_a_slowdown() -> None:
    """Over a window that spans a pause and a resume, no time was asked and lost."""
    loop, _, clock = _loop(LoopSettings(time_scale=0.0, min_frame_s=0.0))
    for _ in range(60):
        clock.now_s += 1 / 60
        loop.run_frame()
    loop.settings = LoopSettings(time_scale=1.0, min_frame_s=0.0)
    for _ in range(10):
        clock.now_s += 1 / 60
        loop.run_frame()

    assert loop.stats.summary().time_kept == pytest.approx(1.0)


def test_frames_beyond_the_clamp_lose_time() -> None:
    loop, _, clock = _loop()
    for _ in range(5):
        clock.now_s += 0.1
        loop.run_frame()

    assert loop.stats.summary().time_kept == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_advance_s": 0.0},
        {"max_advance_s": math.inf},
        {"time_scale": -1.0},
        {"time_scale": math.nan},
        {"min_frame_s": -1.0},
        {"max_logic_dt_s": 0.0},
    ],
)
def test_settings_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LoopSettings(**kwargs)


def test_run_stops_when_asked() -> None:
    loop, stages, _clock = _loop()
    answers = iter([False, False, False, True])

    loop.run(lambda: next(answers))

    assert stages.calls.count("present") == 3

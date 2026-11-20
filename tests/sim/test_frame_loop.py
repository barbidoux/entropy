"""The frame loop (`entropy.sim.loop`): SPEC 5.9 and ADR 0008.

One simulation advance per frame of the frame time times the time scale,
clamped to 1/30 s, and a 60 fps cap.
"""

from dataclasses import dataclass, field

import pytest

from entropy.core.profiler import FrameStats
from entropy.sim.loop import FrameLoop, LoopSettings


@dataclass
class FakeClock:
    now_s: float = 0.0
    slept_s: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, seconds: float) -> None:
        self.slept_s.append(seconds)
        self.now_s += seconds


@dataclass
class FakeStages:
    """Records calls; each stage takes a fixed time on the fake clock."""

    clock: FakeClock
    cost_s: dict[str, float] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    advances_s: list[float] = field(default_factory=list)

    def _run(self, name: str) -> None:
        self.calls.append(name)
        self.clock.now_s += self.cost_s.get(name, 0.0)

    def poll_input(self) -> None:
        self._run("poll_input")

    def update(self, frame_dt_s: float) -> None:
        self._run("update")

    def step_simulation(self, dt_s: float) -> None:
        self.advances_s.append(dt_s)
        self._run("step_simulation")

    def render(self) -> None:
        self._run("render")

    def present(self) -> None:
        self._run("present")


def _loop(
    settings: LoopSettings | None = None, cost_s: dict[str, float] | None = None
) -> tuple[FrameLoop, FakeStages, FakeClock]:
    clock = FakeClock()
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
    assert report.advance_s == pytest.approx(frame_s)
    assert report.dropped_s == 0.0


def test_a_long_frame_advances_at_most_one_thirtieth_and_reports_the_rest() -> None:
    """SPEC 5.9: below 30 fps the world slows down instead of taking huge steps."""
    loop, stages, clock = _loop()

    clock.now_s += 0.1
    report = loop.run_frame()

    assert stages.advances_s == [pytest.approx(1 / 30)]
    assert report.dropped_s == pytest.approx(0.1 - 1 / 30)


def test_time_scale_shrinks_every_advance() -> None:
    """SPEC 4.6: slow motion advances less simulated time per frame, every frame."""
    loop, stages, clock = _loop(LoopSettings(time_scale=0.1, min_frame_s=0.0))

    for _ in range(3):
        clock.now_s += 1 / 60
        loop.run_frame()

    assert stages.advances_s == [pytest.approx(1 / 600)] * 3


def test_a_zero_time_scale_pauses_the_simulation() -> None:
    loop, stages, clock = _loop(LoopSettings(time_scale=0.0, min_frame_s=0.0))

    clock.now_s += 1 / 60
    report = loop.run_frame()

    assert "step_simulation" not in stages.calls
    assert report.advance_s == 0.0
    assert "render" in stages.calls


def test_frames_are_capped_at_sixty_per_second() -> None:
    loop, _, clock = _loop(LoopSettings(), cost_s={"render": 0.004})

    clock.now_s += 1 / 60
    loop.run_frame()
    report = loop.run_frame()

    assert clock.slept_s == [pytest.approx(1 / 60 - 0.004)] * 2
    assert report.frame_s == pytest.approx(1 / 60)


def test_a_slow_frame_does_not_wait() -> None:
    loop, _, clock = _loop(LoopSettings(), cost_s={"render": 0.020})

    clock.now_s += 1 / 60
    loop.run_frame()

    assert clock.slept_s == []


def test_stage_times_are_recorded_for_the_hud() -> None:
    cost = {"poll_input": 0.001, "update": 0.001, "step_simulation": 0.004, "render": 0.005}
    loop, _, clock = _loop(LoopSettings(), cost_s=cost)

    clock.now_s += 1 / 60
    loop.run_frame()
    clock.now_s += 1 / 60
    loop.run_frame()
    summary = loop.stats.summary()

    assert summary.frames == 2
    assert summary.stages_ms_mean == pytest.approx(
        {"logic": 2.0, "simulation": 4.0, "rendering": 5.0, "present": 0.0, "idle": 1000 / 60 - 11}
    )


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="advance"):
        LoopSettings(max_advance_s=0.0)
    with pytest.raises(ValueError, match="time scale"):
        LoopSettings(time_scale=-1.0)
    with pytest.raises(ValueError, match="frame"):
        LoopSettings(min_frame_s=-1.0)


def test_run_stops_when_asked() -> None:
    loop, stages, _clock = _loop()
    answers = iter([False, False, False, True])

    loop.run(lambda: next(answers))

    assert stages.calls.count("present") == 3

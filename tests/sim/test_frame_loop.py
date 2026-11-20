"""The game loop (`entropy.sim.loop`), SPEC 5.9: fixed steps, time scale, capped catch-up."""

from dataclasses import dataclass, field

import pytest

from entropy.core.profiler import FrameStats
from entropy.sim.loop import FrameLoop, LoopSettings


@dataclass
class FakeClock:
    now_s: float = 0.0

    def __call__(self) -> float:
        return self.now_s


@dataclass
class FakeStages:
    """Records calls; each stage takes a fixed time on the fake clock."""

    clock: FakeClock
    cost_s: dict[str, float] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    steps_s: list[float] = field(default_factory=list)

    def _run(self, name: str) -> None:
        self.calls.append(name)
        self.clock.now_s += self.cost_s.get(name, 0.0)

    def poll_input(self) -> None:
        self._run("poll_input")

    def update(self, frame_dt_s: float) -> None:
        self._run("update")

    def step_simulation(self, dt_s: float) -> None:
        self.steps_s.append(dt_s)
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
    loop = FrameLoop(stages, clock=clock, settings=settings or LoopSettings(), stats=FrameStats())
    return loop, stages, clock


def test_stages_run_in_frame_order() -> None:
    loop, stages, clock = _loop()

    clock.now_s += 1 / 60
    loop.run_frame()

    assert stages.calls == ["poll_input", "update", "step_simulation", "render", "present"]


@pytest.mark.parametrize(("frame_s", "steps"), [(1 / 60, 1), (1 / 30, 2), (1 / 120, 0)])
def test_simulated_time_advances_in_fixed_steps(frame_s: float, steps: int) -> None:
    loop, stages, clock = _loop()

    clock.now_s += frame_s
    report = loop.run_frame()

    assert report.steps == steps
    assert stages.steps_s == [pytest.approx(1 / 60)] * steps


def test_leftover_time_carries_to_the_next_frame() -> None:
    loop, _, clock = _loop()

    steps = []
    for _ in range(4):
        clock.now_s += 1 / 120
        steps.append(loop.run_frame().steps)

    assert steps == [0, 1, 0, 1]


def test_time_scale_slows_simulated_time_with_the_same_step() -> None:
    """SPEC 4.6: slow motion advances less simulated time with the same physical step."""
    loop, stages, clock = _loop(LoopSettings(time_scale=0.5))

    for _ in range(4):
        clock.now_s += 1 / 60
        loop.run_frame()

    assert stages.steps_s == [pytest.approx(1 / 60)] * 2


def test_a_long_frame_cannot_start_a_spiral_of_death() -> None:
    loop, _stages, clock = _loop(LoopSettings(max_steps_per_frame=4))

    clock.now_s += 1.0
    report = loop.run_frame()
    clock.now_s += 1 / 60
    next_report = loop.run_frame()

    assert report.steps == 4
    assert report.dropped_s == pytest.approx(1.0 - 4 / 60)
    assert next_report.steps == 1


def test_stage_times_are_recorded_for_the_hud() -> None:
    cost = {"poll_input": 0.001, "update": 0.001, "step_simulation": 0.004, "render": 0.005}
    loop, _, clock = _loop(cost_s=cost)

    clock.now_s += 1 / 60
    loop.run_frame()
    clock.now_s += 1 / 60
    loop.run_frame()
    summary = loop.stats.summary()

    assert summary.frames == 2
    assert summary.stages_ms_mean == pytest.approx(
        {"logic": 2.0, "simulation": 4.0, "rendering": 5.0, "present": 0.0}
    )


def test_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="step"):
        LoopSettings(sim_step_s=0.0)
    with pytest.raises(ValueError, match="time scale"):
        LoopSettings(time_scale=-1.0)
    with pytest.raises(ValueError, match="one step"):
        LoopSettings(max_steps_per_frame=0)


def test_run_stops_when_asked() -> None:
    loop, stages, _clock = _loop()
    answers = iter([False, False, False, True])

    loop.run(lambda: next(answers))

    assert stages.calls.count("present") == 3

"""Time control (`entropy.sim.time_control`, SPEC 4.5, 4.6, #55): pause, slow motion,
single step."""

import pytest

from entropy.core.profiler import FrameStats
from entropy.sim.loop import FrameLoop, LoopSettings
from entropy.sim.time_control import SCALES, TimeControl
from tests.sim.test_frame_loop import FakeClock, FakeStages


def _loop(control: TimeControl) -> tuple[FrameLoop, FakeStages, FakeClock]:
    clock = FakeClock()
    stages = FakeStages(clock)
    loop = FrameLoop(
        stages,
        clock=clock,
        sleep=clock.sleep,
        settings=LoopSettings(),
        stats=FrameStats(),
        time_control=control,
    )
    return loop, stages, clock


def test_the_scales_are_those_of_the_spec() -> None:
    assert SCALES == (1.0, 0.5, 0.25, 0.1)


def test_slower_and_faster_cycle_through_the_scales() -> None:
    control = TimeControl()
    assert control.scale == 1.0

    control.slower()
    control.slower()
    assert control.scale == 0.25
    control.slower()
    control.slower()  # past the slowest: back to 1x
    assert control.scale == 1.0
    control.faster()  # past 1x: to the slowest
    assert control.scale == 0.1


def test_the_loop_advances_by_the_chosen_scale() -> None:
    control = TimeControl()
    control.slower()  # 0.5x
    loop, stages, _ = _loop(control)

    loop.run_frame()  # sleeps to its 60 fps deadline: the next frame lasts 1/60 s
    loop.run_frame()

    assert stages.advances_s[-1] == pytest.approx(0.5 / 60.0, rel=1e-6)


def test_paused_the_world_does_not_advance_but_the_frame_runs() -> None:
    control = TimeControl()
    control.toggle_pause()
    loop, stages, _ = _loop(control)

    for _ in range(3):
        loop.run_frame()

    assert stages.advances_s == []
    assert stages.calls.count("update") == 3 and stages.calls.count("render") == 3


def test_a_single_step_while_paused_advances_one_frame() -> None:
    """At the chosen scale: one sixtieth of a second at 1x, a tenth of that at 0.1x."""
    control = TimeControl()
    control.toggle_pause()
    for _ in range(3):
        control.slower()  # 0.1x
    loop, stages, _ = _loop(control)

    control.step_once()
    loop.run_frame()
    loop.run_frame()

    assert stages.advances_s == [pytest.approx(0.1 / 60.0)]
    assert control.paused


def test_a_single_step_is_ignored_while_running() -> None:
    control = TimeControl()
    control.step_once()

    assert not control.paused and control.frame() == (1.0, False)
    assert not control.step_requested


def test_the_hud_line_says_what_the_world_is_doing() -> None:
    control = TimeControl()
    assert control.hud_line() == "temps : x1"
    control.slower()
    assert control.hud_line() == "temps : x0.5 (ralenti)"
    control.toggle_pause()
    assert control.hud_line() == "temps : EN PAUSE (x0.5)"

"""The frame loop (SPEC 5.9, ADR 0008): input, logic, fixed simulation steps, render.

Each frame advances simulated time by the frame time multiplied by the time
scale, in fixed steps of `sim_step_s`. At most `max_steps_per_frame` steps run
per frame; time beyond that is dropped and reported, so a slow frame makes the
world slow down instead of piling up work (no spiral of death).

The loop knows its stages only through the `FrameStages` protocol, so it runs
with a real window or, in tests, with fakes and a fake clock.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from entropy.core.profiler import FrameStats


class FrameStages(Protocol):
    def poll_input(self) -> None: ...

    def update(self, frame_dt_s: float) -> None:
        """Game logic and camera, once per frame, with the real frame time."""

    def step_simulation(self, dt_s: float) -> None:
        """One fixed simulation step of `dt_s` simulated seconds."""

    def render(self) -> None: ...

    def present(self) -> None:
        """Show the frame; waits for the display when vertical sync is on."""


@dataclass(frozen=True)
class LoopSettings:
    sim_step_s: float = 1.0 / 60.0
    max_steps_per_frame: int = 4
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.sim_step_s <= 0.0:
            raise ValueError("the simulation step must be positive")
        if self.max_steps_per_frame < 1:
            raise ValueError("at least one step per frame")
        if self.time_scale < 0.0:
            raise ValueError("the time scale cannot be negative")


@dataclass(frozen=True)
class FrameReport:
    frame_s: float
    steps: int
    dropped_s: float
    """Simulated time given up because of the step cap, in seconds."""


class FrameLoop:
    def __init__(
        self,
        stages: FrameStages,
        *,
        clock: Callable[[], float] = time.perf_counter,
        settings: LoopSettings | None = None,
        stats: FrameStats | None = None,
    ) -> None:
        self.stages = stages
        self.settings = settings or LoopSettings()
        self.stats = stats or FrameStats()
        self._clock = clock
        self._last_s = clock()
        self._accumulator_s = 0.0

    def run_frame(self) -> FrameReport:
        settings = self.settings
        start_s = self._clock()
        frame_s = start_s - self._last_s
        self._last_s = start_s

        self.stages.poll_input()
        self.stages.update(frame_s)
        logic_end_s = self._clock()

        self._accumulator_s += frame_s * settings.time_scale
        due = math.floor((self._accumulator_s + 1e-9) / settings.sim_step_s)
        steps = min(due, settings.max_steps_per_frame)
        self._accumulator_s = max(0.0, self._accumulator_s - steps * settings.sim_step_s)
        dropped_s = 0.0
        if due > steps:
            dropped_s, self._accumulator_s = self._accumulator_s, 0.0
        for _ in range(steps):
            self.stages.step_simulation(settings.sim_step_s)
        sim_end_s = self._clock()

        self.stages.render()
        render_end_s = self._clock()
        self.stages.present()
        present_end_s = self._clock()

        self.stats.record(
            frame_s,
            {
                "logic": logic_end_s - start_s,
                "simulation": sim_end_s - logic_end_s,
                "rendering": render_end_s - sim_end_s,
                "present": present_end_s - render_end_s,
            },
        )
        return FrameReport(frame_s=frame_s, steps=steps, dropped_s=dropped_s)

    def run(self, should_close: Callable[[], bool]) -> None:
        """Run frames until `should_close()` is true."""
        while not should_close():
            self.run_frame()

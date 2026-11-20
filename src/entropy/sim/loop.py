"""The frame loop (SPEC 5.9, ADR 0008): input, logic, one simulation advance, render.

Each frame advances the world once by the measured frame time multiplied by
the time scale, clamped to `max_advance_s` (1/30 s): below 30 fps the world
slows down instead of taking huge steps, and the report says by how much.
Solvers split each advance into CFL substeps themselves. Frames last at least
`min_frame_s` (1/60 s): the loop waits after presenting, whatever the display.

The loop knows its stages only through the `FrameStages` protocol, so it runs
with a real window or, in tests, with fakes and a fake clock.
"""

from __future__ import annotations

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
        """Advance the world by `dt_s` simulated seconds."""

    def render(self) -> None: ...

    def present(self) -> None:
        """Show the frame; waits for the display when vertical sync is on."""


@dataclass(frozen=True)
class LoopSettings:
    max_advance_s: float = 1.0 / 30.0
    """Largest simulated time per frame, in seconds (SPEC 5.9)."""
    min_frame_s: float = 1.0 / 60.0
    """Shortest frame, in seconds: the 60 fps cap (ADR 0008); 0 disables it."""
    time_scale: float = 1.0
    """Simulated seconds per real second; 0 pauses (SPEC 4.6)."""

    def __post_init__(self) -> None:
        if self.max_advance_s <= 0.0:
            raise ValueError("the largest advance per frame must be positive")
        if self.min_frame_s < 0.0:
            raise ValueError("the shortest frame cannot be negative")
        if self.time_scale < 0.0:
            raise ValueError("the time scale cannot be negative")


@dataclass(frozen=True)
class FrameReport:
    frame_s: float
    advance_s: float
    """Simulated time this frame, in seconds."""
    dropped_s: float
    """Real time the clamp did not simulate, in seconds (before the time scale)."""


class FrameLoop:
    def __init__(
        self,
        stages: FrameStages,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        settings: LoopSettings | None = None,
        stats: FrameStats | None = None,
    ) -> None:
        self.stages = stages
        self.settings = settings or LoopSettings()
        self.stats = stats or FrameStats()
        self._clock = clock
        self._sleep = sleep
        self._last_s = clock()

    def run_frame(self) -> FrameReport:
        settings = self.settings
        start_s = self._clock()
        frame_s = start_s - self._last_s
        self._last_s = start_s

        self.stages.poll_input()
        self.stages.update(frame_s)
        logic_end_s = self._clock()

        simulated_s = min(frame_s, settings.max_advance_s)
        advance_s = simulated_s * settings.time_scale
        if advance_s > 0.0:
            self.stages.step_simulation(advance_s)
        sim_end_s = self._clock()

        self.stages.render()
        render_end_s = self._clock()
        self.stages.present()
        present_end_s = self._clock()

        spare_s = settings.min_frame_s - (present_end_s - start_s)
        if spare_s > 0.0:
            self._sleep(spare_s)
        idle_end_s = self._clock()

        self.stats.record(
            frame_s,
            {
                "logic": logic_end_s - start_s,
                "simulation": sim_end_s - logic_end_s,
                "rendering": render_end_s - sim_end_s,
                "present": present_end_s - render_end_s,
                "idle": idle_end_s - present_end_s,
            },
        )
        return FrameReport(frame_s=frame_s, advance_s=advance_s, dropped_s=frame_s - simulated_s)

    def run(self, should_close: Callable[[], bool]) -> None:
        """Run frames until `should_close()` is true."""
        while not should_close():
            self.run_frame()

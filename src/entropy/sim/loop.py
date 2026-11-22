"""The frame loop (SPEC 5.9, ADR 0008, ADR 0010).

Each frame: poll input, update game logic with the real frame time (capped
against stalls), advance the world once by the frame time multiplied by the
time scale, clamped to `max_advance_s` (1/30 s), render, present, then wait
until the frame's deadline. Below 30 fps the clamp makes the world slow down
instead of taking huge steps. The simulation may advance less still, when a
solver's substep cap shrinks the shared advance (ADR 0013); the report and the
statistics count the time actually advanced, so both slowdowns show. Solvers
split each advance into CFL substeps themselves.

Frames are paced against a deadline that advances by `min_frame_s` (1/60 s)
each frame, so a sleep that wakes up late is made up on the next frame instead
of accumulating (Windows `time.sleep` oversleeps by about 0.5 ms).

The loop sees its stages only through the `FrameStages` protocol, so `sim`
imports nothing from `render`, `ui` or `app` (ADR 0010); stage times are CPU
wall times (see `entropy.core.profiler`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from entropy.core.profiler import FrameStats
from entropy.sim.time_control import MIN_TIME_SCALE, TimeControl


class FrameStages(Protocol):
    def poll_input(self) -> None: ...

    def update(self, frame_dt_s: float) -> None:
        """Game logic and camera, once per frame, with real time (not simulated)."""

    def step_simulation(self, dt_s: float) -> float:
        """Advance the world by up to `dt_s` simulated seconds; return the time advanced.

        It may be less than `dt_s` when a solver's substep cap shrinks the advance
        (ADR 0013, `entropy.sim.simulation`), never more."""
        ...

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
    """Simulated seconds per real second; 0 pauses (SPEC 4.6), otherwise at least
    `MIN_TIME_SCALE`."""
    max_logic_dt_s: float = 0.1
    """Largest real time handed to game logic, in seconds, after a stall."""

    def __post_init__(self) -> None:
        if not 0.0 < self.max_advance_s < math.inf:
            raise ValueError("the largest advance per frame must be positive and finite")
        if not 0.0 <= self.min_frame_s < math.inf:
            raise ValueError("the shortest frame must be zero or positive")
        if not (self.time_scale == 0.0 or MIN_TIME_SCALE <= self.time_scale < math.inf):
            raise ValueError(
                f"the time scale must be 0 (paused) or at least {MIN_TIME_SCALE} (float32 "
                "positions, #36)"
            )
        if not 0.0 < self.max_logic_dt_s < math.inf:
            raise ValueError("the largest logic step must be positive and finite")


@dataclass(frozen=True)
class FrameReport:
    frame_s: float
    """Real time since the previous frame started, in seconds."""
    advance_s: float
    """Simulated time this frame, in seconds."""
    dropped_s: float
    """Real time the clamp did not simulate, in seconds (0 while paused)."""


class FrameLoop:
    def __init__(
        self,
        stages: FrameStages,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        settings: LoopSettings | None = None,
        stats: FrameStats | None = None,
        time_control: TimeControl | None = None,
    ) -> None:
        self.stages = stages
        self.settings = settings or LoopSettings()
        self.time_control = time_control
        """The player's pause, slow motion and single step; when set, it gives each
        frame's time scale instead of `settings.time_scale`."""
        self.stats = stats or FrameStats()
        self._clock = clock
        self._sleep = sleep
        self._last_start_s = clock()
        self._next_start_s = self._last_start_s

    def reset_clock(self) -> None:
        """Start timing from now, so set-up time is not taken for a long frame."""
        self._last_start_s = self._next_start_s = self._clock()

    def run_frame(self) -> FrameReport:
        settings = self.settings
        start_s = self._clock()
        frame_s = start_s - self._last_start_s
        self._last_start_s = start_s

        self.stages.poll_input()
        self.stages.update(min(frame_s, settings.max_logic_dt_s))
        logic_end_s = self._clock()

        scale, single_step = settings.time_scale, False
        if self.time_control is not None:
            scale, single_step = self.time_control.frame()
        asked_s = min(frame_s * scale, settings.max_advance_s)
        if single_step:  # one 60 fps frame's worth at the chosen scale, whatever the clock
            asked_s = (settings.min_frame_s or 1.0 / 60.0) * scale
        advance_s = 0.0
        if asked_s > 0.0:
            advance_s = self.stages.step_simulation(asked_s)
            if not 0.0 <= advance_s <= asked_s * (1.0 + 1e-9):  # also refuses NaN
                raise RuntimeError(
                    f"the simulation advanced {advance_s} s when asked for {asked_s} s"
                )
        dropped_s = max(frame_s - advance_s / scale, 0.0) if scale > 0.0 else 0.0
        sim_end_s = self._clock()

        self.stages.render()
        render_end_s = self._clock()
        self.stages.present()
        present_end_s = self._clock()
        self._wait_for_deadline(start_s, present_end_s)
        end_s = self._clock()

        self.stats.record(
            end_s - start_s,
            {
                "logic": logic_end_s - start_s,
                "simulation": sim_end_s - logic_end_s,
                "rendering": render_end_s - sim_end_s,
                "present": present_end_s - render_end_s,
                "idle": end_s - present_end_s,
            },
            advance_s=advance_s,
            real_s=frame_s,
            requested_s=frame_s * scale,
        )
        return FrameReport(frame_s=frame_s, advance_s=advance_s, dropped_s=dropped_s)

    def _wait_for_deadline(self, start_s: float, now_s: float) -> None:
        period_s = self.settings.min_frame_s
        if period_s <= 0.0:
            return
        # Normally this frame was due at `_next_start_s`, perhaps a little earlier than
        # it started (a late sleep): keep that rhythm. After a stall, start a new one.
        base_s = self._next_start_s
        if start_s - base_s > period_s / 2:
            base_s = start_s
        deadline_s = base_s + period_s
        if now_s >= deadline_s:
            self._next_start_s = now_s
            return
        self._sleep(deadline_s - now_s)
        self._next_start_s = deadline_s

    def run(self, should_close: Callable[[], bool]) -> None:
        """Run frames until `should_close()` is true."""
        self.reset_clock()
        while not should_close():
            self.run_frame()

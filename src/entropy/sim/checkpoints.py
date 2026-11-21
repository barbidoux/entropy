"""Checkpoints (SPEC 4.6, #55): the world's state saved in host memory, to go back to.

A ring keeps the last `keep` checkpoints: one every `every_s` seconds of simulated time
(10 s, ring of 6, by default) and one whenever the player asks (F5). Restoring (F9)
returns every solver to the latest one, through the solver interface's `checkpoint()`
and `restore()`, and the simulated clock with it. A re-run from a checkpoint is not
bit-identical on the GPU (SPEC 9.6).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any, Protocol


class Checkpointed(Protocol):
    def checkpoint(self) -> Any: ...

    def restore(self, state: Any) -> None: ...


class CheckpointRing:
    def __init__(
        self, solvers: Sequence[Checkpointed], *, every_s: float = 10.0, keep: int = 6
    ) -> None:
        if not (every_s > 0 and keep > 0):
            raise ValueError("every_s and keep must be positive")
        self.solvers = tuple(solvers)
        self.every_s = every_s
        self.entries: deque[tuple[float, list[Any]]] = deque(maxlen=keep)
        self.clock_s = 0.0
        """Simulated time since the world started (s)."""
        self._next_s = every_s

    def advance(self, dt_s: float) -> None:
        """The world advanced by `dt_s` of simulated time; save when a period is up."""
        self.clock_s += dt_s
        if self.clock_s >= self._next_s - 1e-9:
            self.save_now()
            while self._next_s <= self.clock_s + 1e-9:
                self._next_s += self.every_s

    def save_now(self) -> None:
        """Take a checkpoint of every solver now; asynchronous where a solver offers it
        (`checkpoint_async`), so taking one does not wait for the GPU."""
        states = [
            getattr(solver, "checkpoint_async", solver.checkpoint)() for solver in self.solvers
        ]
        self.entries.append((self.clock_s, states))

    def drop_unchecked(self) -> None:
        """Drop the checkpoints saved since the last step, when the world froze: the NaN
        guard reads a step's state only at the next frame, so they may hold the
        non-finite state (SPEC 5.9). Those saved earlier were checked by a later step."""
        while self.entries and self.entries[-1][0] >= self.clock_s - 1e-9:
            self.entries.pop()

    def saved_times_s(self) -> list[float]:
        return [time_s for time_s, _ in self.entries]

    def restore_latest(self) -> float | None:
        """Return every solver to the latest checkpoint; its simulated time, or None."""
        if not self.entries:
            return None
        time_s, states = self.entries[-1]
        for solver, state in zip(self.solvers, states, strict=True):
            solver.restore(state)
        self.clock_s = time_s
        self._next_s = time_s + self.every_s
        return time_s

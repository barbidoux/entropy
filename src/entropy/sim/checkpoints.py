"""Checkpoints (SPEC 4.6, #55, ADR 0024): the world's state saved in host memory, to go
back to.

A ring keeps the last `keep` checkpoints: one every `every_s` seconds of simulated time
(10 s, ring of 6, by default) and one whenever the player asks (F5). Restoring (F9)
returns every solver to the latest one, through the solver interface's `checkpoint()`
and `restore()`, and the simulated clock with it; F9 again, before the world runs on,
goes one checkpoint further back. Once the world runs on from a restored checkpoint,
the later ones are forgotten. A re-run from a checkpoint is not bit-identical on the
GPU (SPEC 9.6).

`Rewind` keeps the ring and the NaN guard (SPEC 5.9) in step: the guard reads a step's
state only at the next frame, so a checkpoint saved after the last step is unchecked
until then.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from entropy.sim.simulation import Simulation
from entropy.solvers.interface import AsyncCheckpointed


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
        self._restored: int | None = None
        """Index of the entry last restored, while the world has not run on from it."""

    def advance(self, dt_s: float) -> None:
        """The world advanced by `dt_s` of simulated time; save when a period is up."""
        if dt_s > 0.0:
            self._forget_later()
        self.clock_s += dt_s
        if self.clock_s >= self._next_s - 1e-9:
            self.save_now()
            while self._next_s <= self.clock_s + 1e-9:
                self._next_s += self.every_s

    def save_now(self) -> None:
        """Take a checkpoint of every solver now; asynchronous where a solver offers it
        (`checkpoint_async`), so taking one does not wait for the GPU. When the ring is
        full, the checkpoint that falls off it is handed over for its buffers."""
        self._forget_later()
        evicted = self.entries[0][1] if len(self.entries) == self.entries.maxlen else None
        states = []
        for index, solver in enumerate(self.solvers):
            if isinstance(solver, AsyncCheckpointed):
                states.append(solver.checkpoint_async(reuse=evicted[index] if evicted else None))
            else:
                states.append(solver.checkpoint())
        self.entries.append((self.clock_s, states))

    def drop_unchecked(self) -> None:
        """Drop the checkpoints saved since the last step, when the world froze: the NaN
        guard reads a step's state only at the next frame, so they may hold the
        non-finite state (SPEC 5.9). Those saved earlier were checked by a later step."""
        while self.entries and self.entries[-1][0] >= self.clock_s - 1e-9:
            self.entries.pop()
        if self._restored is not None and self._restored >= len(self.entries):
            self._restored = None

    def saved_times_s(self) -> list[float]:
        return [time_s for time_s, _ in self.entries]

    def restore_latest(self) -> float | None:
        """Return every solver to the latest checkpoint, or, right after a restore, to the
        one before it (down to the oldest); its simulated time, or None if there is none."""
        if not self.entries:
            return None
        index = len(self.entries) - 1
        if self._restored is not None:
            index = max(self._restored - 1, 0)
        time_s, states = self.entries[index]
        for solver, state in zip(self.solvers, states, strict=True):
            solver.restore(state)
        self.clock_s = time_s
        self._next_s = time_s + self.every_s
        self._restored = index
        return time_s

    def _forget_later(self) -> None:
        """The world runs on from a restored checkpoint: the ones after it are gone."""
        if self._restored is not None:
            while len(self.entries) > self._restored + 1:
                self.entries.pop()
            self._restored = None


class Rewind:
    """The world's checkpoints and its NaN guard together (SPEC 4.6, 5.9; ADR 0024).

    - A checkpoint is taken at the start, so there is always one to go back to.
    - While the world is frozen, saving is refused: it would save the non-finite state.
    - When a step finds the world non-finite, the checkpoints saved since the last good
      step are dropped (`CheckpointRing.drop_unchecked`).
    - Restoring reads the last step's verdict first (`Simulation.check`), so an unchecked
      checkpoint is never restored; then it unfreezes the world.
    """

    def __init__(self, simulation: Simulation, ring: CheckpointRing) -> None:
        self.simulation = simulation
        self.ring = ring
        ring.save_now()

    def step(self, requested_s: float, *, player: Callable[[float], None] | None = None) -> float:
        """`Simulation.step`, keeping the checkpoints in step; returns the advance (s)."""
        running = self.simulation.frozen_by is None
        advanced_s = self.simulation.step(requested_s, player=player)
        if running and self.simulation.frozen_by is not None:
            self.ring.drop_unchecked()
        self.ring.advance(advanced_s)
        return advanced_s

    def save(self) -> bool:
        """Save a checkpoint now (F5); False, saving nothing, while the world is frozen."""
        if self.simulation.frozen_by is not None:
            return False
        self.ring.save_now()
        return True

    def restore(self) -> float | None:
        """Go back to the latest checked checkpoint (F9; again: further back) and run
        again; its simulated time, or None if there is none."""
        if not self.simulation.check():
            self.ring.drop_unchecked()
        time_s = self.ring.restore_latest()
        if time_s is not None:
            self.simulation.unfreeze()
        return time_s

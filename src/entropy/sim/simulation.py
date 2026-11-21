"""One shared advance per frame for every solver (ADR 0013, SPEC 5.9).

The frame asks for an advance: frame time x time scale, clamped to 1/30 s by
the frame loop. Before any solver steps, the CPU shrinks it to the tightest
solver limit. Each solver states its limit from the speeds it read back
from the previous frame, so no synchronisation point is added. Every solver
then steps by that same advance, in the given order (the frame order of SPEC
5.9), and the advance actually taken is returned. A slowdown forced by a
solver's substep cap therefore shows in the HUD like one from the clamp.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from entropy.solvers.interface import NonFiniteStateError

_log = logging.getLogger(__name__)


@runtime_checkable
class PacedSolver(Protocol):
    """The part of `entropy.solvers.interface.Solver` the simulation uses (SPEC 8.2):
    every solver is one, and tests pace small fakes with it."""

    def max_advance_s(self) -> float:
        """Largest advance (s) this solver can take within its substep cap; math.inf if
        unlimited. From the previous frame's speeds: never waits for the GPU. Always
        positive for a finite state: 0 would hold the world for good, since a held
        world never updates the speeds the limit comes from."""
        ...

    def step(self, dt_s: float, ctx: Any) -> None: ...


class Simulation:
    """The solvers of a world, stepped together by one advance per frame."""

    def __init__(self, solvers: Sequence[PacedSolver], context: Any = None) -> None:
        self.solvers = tuple(solvers)
        self.context = context
        """What each solver's `step` receives for coupling; nothing until M2 couples solvers."""
        self.frozen_because: str | None = None
        """Why the world stopped (the NaN guard's alert), or None while it runs."""

    def advance_for(self, requested_s: float) -> float:
        """The advance every solver can take: `requested_s`, shrunk to the tightest limit."""
        if not (math.isfinite(requested_s) and requested_s >= 0.0):
            raise ValueError(f"requested advance must be zero or positive, got {requested_s}")
        advance_s = requested_s
        for solver in self.solvers:
            limit_s = solver.max_advance_s()
            if not limit_s > 0.0:  # also refuses NaN
                name = type(solver).__name__
                raise ValueError(f"{name}.max_advance_s() gave {limit_s}: a bug, or NaN upstream")
            advance_s = min(advance_s, limit_s)
        return advance_s

    def step(self, requested_s: float) -> float:
        """Advance every solver by one shared advance; return it (s).

        NaN guard (SPEC 5.9): when a solver finds its state non-finite, it has dumped it
        and raises; the world then freezes, nothing advances until `unfreeze`, and
        `frozen_because` says where the dump is."""
        if self.frozen_because is not None:
            return 0.0
        try:
            advance_s = self.advance_for(requested_s)
            if advance_s > 0.0:
                for solver in self.solvers:
                    solver.step(advance_s, self.context)
        except NonFiniteStateError as error:
            self.frozen_because = f"état non fini, sauvegardé dans {error.dump_path}"
            _log.error("the world froze: %s", error)
            return 0.0
        return advance_s

    def unfreeze(self) -> None:
        """Run again, after the solvers were restored to a finite state."""
        self.frozen_because = None

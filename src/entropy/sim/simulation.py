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

import math
from collections.abc import Sequence
from typing import Any, Protocol


class PacedSolver(Protocol):
    def max_advance_s(self) -> float:
        """Largest advance (s) this solver can take within its substep cap; math.inf if
        unlimited. From the previous frame's speeds: never waits for the GPU."""
        ...

    def step(self, dt_s: float, ctx: Any) -> None: ...


class Simulation:
    """The solvers of a world, stepped together by one advance per frame."""

    def __init__(self, solvers: Sequence[PacedSolver], context: Any = None) -> None:
        self.solvers = tuple(solvers)
        self.context = context
        """What each solver's `step` receives for coupling; nothing until M2 couples solvers."""

    def advance_for(self, requested_s: float) -> float:
        """The advance every solver can take: `requested_s`, shrunk to the tightest limit."""
        if not (math.isfinite(requested_s) and requested_s >= 0.0):
            raise ValueError(f"requested advance must be zero or positive, got {requested_s}")
        advance_s = requested_s
        for solver in self.solvers:
            limit_s = solver.max_advance_s()
            if math.isnan(limit_s) or limit_s < 0.0:
                name = type(solver).__name__
                raise ValueError(f"{name}.max_advance_s() gave {limit_s}: a bug, or NaN upstream")
            advance_s = min(advance_s, limit_s)
        return advance_s

    def step(self, requested_s: float) -> float:
        """Advance every solver by one shared advance; return it (s)."""
        advance_s = self.advance_for(requested_s)
        if advance_s > 0.0:
            for solver in self.solvers:
                solver.step(advance_s, self.context)
        return advance_s

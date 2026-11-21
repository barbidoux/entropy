"""The solver interface every solver implements (SPEC 8.2)."""

import math
from typing import Any

from entropy.solvers.interface import Solver


class _Complete:
    def allocate(self, world: Any, device: str) -> None: ...

    def step(self, dt_s: float, ctx: Any) -> None: ...

    def fields(self) -> dict[str, Any]:
        return {}

    def checkpoint(self) -> Any: ...

    def restore(self, state: Any) -> None: ...

    def diagnostics(self) -> dict[str, float]:
        return {}

    def max_advance_s(self) -> float:
        return math.inf


class _Unpaced(_Complete):
    """Every method of M0's interface, but no `max_advance_s` (ADR 0013)."""

    max_advance_s = None  # pyright: ignore[reportAssignmentType]


class _NoCheckpoint:
    def allocate(self, world: Any, device: str) -> None: ...

    def step(self, dt_s: float, ctx: Any) -> None: ...

    def fields(self) -> dict[str, Any]:
        return {}

    def diagnostics(self) -> dict[str, float]:
        return {}


def test_a_class_with_every_method_is_a_solver() -> None:
    assert isinstance(_Complete(), Solver)


def test_a_class_missing_a_method_is_not() -> None:
    assert not isinstance(_NoCheckpoint(), Solver)


def test_a_solver_states_how_far_it_can_advance() -> None:
    assert not isinstance(_Unpaced(), Solver)

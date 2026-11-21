"""The interface every solver implements (SPEC 8.2).

Solvers own their device arrays and talk to each other only through the
coupling layer. `allocate` receives the world as the simulation presents it to
that solver: `solvers` may not import `world` (SPEC 8.1). Each solver states in
its docstring what it reads from `world`.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Solver(Protocol):
    def allocate(self, world: Any, device: str) -> None:
        """Create the solver's arrays on `device` from the world's initial state."""
        ...

    def step(self, dt_s: float, ctx: Any) -> None:
        """Advance by `dt_s` seconds (the solver splits it into its own substeps)."""
        ...

    def fields(self) -> dict[str, Any]:
        """Named device arrays offered to coupling and rendering (read-only for them)."""
        ...

    def checkpoint(self) -> Any:
        """A copy of the full state, for checkpoints (SPEC 4.6)."""
        ...

    def restore(self, state: Any) -> None:
        """Return to a state from `checkpoint`."""
        ...

    def diagnostics(self) -> dict[str, float]:
        """Counts and timings for the HUD: particles, cells, substeps, iterations."""
        ...

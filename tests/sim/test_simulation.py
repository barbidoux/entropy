"""One shared advance per frame for every solver (`entropy.sim.simulation`), ADR 0013."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from entropy.sim.simulation import Simulation
from entropy.solvers.interface import NonFiniteStateError


@dataclass
class FakeSolver:
    name: str
    limit_s: float = math.inf
    log: list[tuple[str, float]] = field(default_factory=list)

    def max_advance_s(self) -> float:
        return self.limit_s

    def step(self, dt_s: float, ctx: Any) -> None:
        self.log.append((self.name, dt_s))


def _solvers(*limits: float) -> tuple[list[FakeSolver], list[tuple[str, float]]]:
    log: list[tuple[str, float]] = []
    return [FakeSolver(f"s{i}", limit, log) for i, limit in enumerate(limits)], log


def test_every_solver_steps_by_the_same_advance_in_order() -> None:
    solvers, log = _solvers(math.inf, math.inf)

    advanced = Simulation(solvers).step(1 / 60)

    assert advanced == pytest.approx(1 / 60)
    assert log == [("s0", pytest.approx(1 / 60)), ("s1", pytest.approx(1 / 60))]


def test_the_advance_shrinks_to_the_tightest_solver_before_any_steps() -> None:
    """The CPU fixes the advance first, from the solvers' limits (previous frame's speeds)."""
    solvers, log = _solvers(0.02, 0.01, math.inf)

    advanced = Simulation(solvers).step(1 / 30)

    assert advanced == pytest.approx(0.01)
    assert [dt for _, dt in log] == [pytest.approx(0.01)] * 3


def test_no_solver_steps_when_there_is_nothing_to_advance() -> None:
    solvers, log = _solvers(math.inf)

    assert Simulation(solvers).step(0.0) == 0.0
    assert log == []


def test_limits_are_all_read_before_any_solver_steps() -> None:
    """A solver stepping first must not change the advance the others get."""
    solvers, log = _solvers(math.inf, 0.01)
    original_step = solvers[0].step

    def step_and_tighten(dt_s: float, ctx: object) -> None:
        original_step(dt_s, ctx)
        solvers[1].limit_s = 0.001

    solvers[0].step = step_and_tighten  # type: ignore[method-assign]

    Simulation(solvers).step(1 / 30)

    assert [dt for _, dt in log] == [pytest.approx(0.01)] * 2


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan])
def test_an_impossible_limit_names_the_solver(limit: float) -> None:
    solvers, _ = _solvers(math.inf, limit)

    with pytest.raises(ValueError, match=r"FakeSolver\.max_advance_s"):
        Simulation(solvers).step(1 / 60)


def test_a_world_without_solvers_still_advances() -> None:
    """M0's scene has no solver: simulated time still passes."""
    assert Simulation([]).step(1 / 60) == pytest.approx(1 / 60)


def test_a_negative_request_is_refused() -> None:
    with pytest.raises(ValueError, match="requested"):
        Simulation([]).step(-0.01)


class _BrokenSolver:
    """Finds its state non-finite on its next call, as the liquid's NaN guard does."""

    def __init__(self) -> None:
        self.calls = 0

    def max_advance_s(self) -> float:
        return math.inf

    def step(self, dt_s: float, ctx: Any) -> None:
        self.calls += 1
        from pathlib import Path

        from entropy.solvers.interface import NonFiniteStateError

        raise NonFiniteStateError(Path("out/nan_dumps/test.npz"))


def test_a_non_finite_solver_freezes_the_world() -> None:
    """SPEC 5.9, 8.1: the simulation stops stepping and says why; nothing advances."""
    broken = _BrokenSolver()
    simulation = Simulation([broken])

    assert simulation.step(1.0 / 60.0) == 0.0
    assert simulation.step(1.0 / 60.0) == 0.0  # stays frozen: no second dump
    assert broken.calls == 1
    assert simulation.frozen_by is not None
    assert simulation.frozen_by.dump_path == Path("out/nan_dumps/test.npz")


def test_unfreezing_lets_the_world_run_again() -> None:
    """After a checkpoint is restored (#55), the game unfreezes the world."""
    solver = FakeSolver(name="ok")
    simulation = Simulation([solver])
    simulation.frozen_by = NonFiniteStateError(Path("out/nan_dumps/earlier.npz"))

    simulation.unfreeze()

    assert simulation.step(1.0 / 60.0) > 0.0


def test_the_player_acts_first_with_the_shared_advance() -> None:
    """SPEC 5.9, step 1 of the frame: the tools act with the advance the solvers will take,
    before any of them steps."""
    solvers, log = _solvers(0.01)
    simulation = Simulation(solvers)

    advance_s = simulation.step(1.0 / 30.0, player=lambda dt_s: log.append(("player", dt_s)))

    assert advance_s == 0.01
    assert log == [("player", 0.01), ("s0", 0.01)]


def test_the_player_does_not_act_on_a_frozen_world() -> None:
    """While frozen, the tools add or remove nothing either."""
    solvers, log = _solvers(math.inf)
    simulation = Simulation(solvers)
    simulation.frozen_by = NonFiniteStateError(Path("out/nan_dumps/earlier.npz"))

    assert simulation.step(1.0 / 60.0, player=lambda dt_s: log.append(("player", dt_s))) == 0.0
    assert log == []


class _BrokenLimit:
    """Finds its state non-finite when reading back its speeds, as the liquid can."""

    def max_advance_s(self) -> float:
        from pathlib import Path

        from entropy.solvers.interface import NonFiniteStateError

        raise NonFiniteStateError(Path("out/nan_dumps/limit.npz"))

    def step(self, dt_s: float, ctx: Any) -> None:
        raise AssertionError("a frozen solver must not step")


def test_a_state_found_non_finite_while_pacing_freezes_the_world() -> None:
    """The speeds read back for the limit may be the first to show NaN: the world freezes
    there too, and the player does not act."""
    acted: list[float] = []
    simulation = Simulation([_BrokenLimit()])

    assert simulation.step(1.0 / 60.0, player=acted.append) == 0.0
    assert acted == []
    assert simulation.frozen_by is not None
    assert simulation.frozen_by.dump_path.name == "limit.npz"

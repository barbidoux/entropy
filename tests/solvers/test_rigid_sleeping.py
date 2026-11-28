"""Sleeping islands of the rigid bodies (`entropy.solvers.rigid`, SPEC 5.4, #89, ADR
0033): bodies at rest sleep in islands, frozen, and wake on contact or a tool."""

import numpy as np

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, Solids

GRAVITY = Environment().gravity_m_per_s2
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))


def _solver(
    device: str, solids: Solids | None = FLOOR, gravity=GRAVITY, max_bodies: int = 16, **kw
) -> RigidSolver:
    setup = RigidSetup(
        gravity_m_per_s2=gravity,
        max_bodies=max_bodies,
        solids=solids.to_device(device) if solids is not None else None,
        **kw,
    )
    return RigidSolver.create(setup, device)


def _box(position_m, half=(0.1, 0.1, 0.1), **kw) -> Body:
    return Body(Shape.BOX, half, 600.0, position_m=position_m, **kw)


def _run(solver: RigidSolver, seconds: float) -> None:
    for _ in range(round(seconds * 60)):
        solver.step(1.0 / 60.0)


def test_a_resting_box_becomes_quiet(device: str) -> None:
    """A box resting on the floor: its speed stays under 0.05 m/s, so after 1 s its quiet
    timer has run at least the 0.5 s it needs (ADR 0033)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1), static_friction=0.5, dynamic_friction=0.5)])

    _run(solver, 1.0)

    assert solver.checkpoint()["sleep_timer_s"][0] >= 0.5


def test_a_sliding_box_stays_unquiet(device: str) -> None:
    """A box sliding at 1 m/s on a frictionless floor is never quiet: its timer stays 0."""
    solver = _solver(device)
    solver.add_bodies([_box((0.0, 0.0, 0.1), velocity_m_per_s=(1.0, 0.0, 0.0))])

    _run(solver, 1.0)

    assert solver.checkpoint()["sleep_timer_s"][0] == 0.0
    assert np.isfinite(solver.body_positions()).all()

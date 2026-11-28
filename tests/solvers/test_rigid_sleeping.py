"""Sleeping islands of the rigid bodies (`entropy.solvers.rigid`, SPEC 5.4, #89, ADR
0033): bodies at rest sleep in islands, frozen, and wake on contact or a tool."""

import numpy as np

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, Solids

GRAVITY = Environment().gravity_m_per_s2
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))


def _solver(
    device: str,
    solids: Solids | None = FLOOR,
    gravity=GRAVITY,
    max_bodies: int = 16,
    mu: float = 0.0,
    sleeping: bool = True,
) -> RigidSolver:
    """A solver whose static solids have friction `mu`, static and dynamic."""
    setup = RigidSetup(
        gravity_m_per_s2=gravity,
        max_bodies=max_bodies,
        solids=solids.to_device(device) if solids is not None else None,
        static_friction=mu,
        dynamic_friction=mu,
        sleeping=sleeping,
    )
    return RigidSolver.create(setup, device)


def _box(position_m, half=(0.1, 0.1, 0.1), mu: float = 0.0, **kw) -> Body:
    return Body(Shape.BOX, half, 600.0, position_m=position_m, static_friction=mu,
                dynamic_friction=mu, **kw)  # fmt: skip


def _run(solver: RigidSolver, seconds: float) -> None:
    for _ in range(round(seconds * 60)):
        solver.step(1.0 / 60.0)


def test_a_resting_box_becomes_quiet(device: str) -> None:
    """A box resting on the floor: its speed stays under 0.05 m/s, so after 1 s its quiet
    timer has run at least the 0.5 s it needs (ADR 0033)."""
    solver = _solver(device, mu=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1), mu=0.5)])

    _run(solver, 1.0)

    assert solver.checkpoint()["sleep_timer_s"][0] >= 0.5


def test_a_sliding_box_stays_unquiet(device: str) -> None:
    """A box sliding at 1 m/s on a frictionless floor is never quiet: its timer stays 0."""
    solver = _solver(device)
    solver.add_bodies([_box((0.0, 0.0, 0.1), velocity_m_per_s=(1.0, 0.0, 0.0))])

    _run(solver, 1.0)

    assert solver.checkpoint()["sleep_timer_s"][0] == 0.0
    assert np.isfinite(solver.body_positions()).all()


def _stack(device: str, count: int = 3, sleeping: bool = True) -> RigidSolver:
    """`count` 20 cm boxes stacked on the floor, μ = 0.5."""
    solver = _solver(device, mu=0.5, sleeping=sleeping)
    solver.add_bodies([_box((0.0, 0.0, 0.1 + 0.2 * k), mu=0.5) for k in range(count)])
    return solver


def test_a_resting_stack_falls_asleep_whole(device: str) -> None:
    """Three boxes stacked on the floor, μ = 0.5: within 1.5 s of a 1 s settle all three
    sleep, in the same frame: the count is never 1 or 2 (ADR 0033)."""
    solver = _stack(device)
    _run(solver, 1.0)
    counts = []
    for _ in range(90):
        solver.step(1.0 / 60.0)
        counts.append(int(solver.diagnostics()["sleeping"]))

    assert counts[-1] == 3
    assert set(counts) <= {0, 3}


def test_a_sleeping_stack_does_not_move_and_costs_no_contacts(device: str) -> None:
    """Once asleep, a stack's positions and orientations do not change at all over 10 s,
    and the solver reports no contacts for it: it solves nothing (ADR 0033)."""
    solver = _stack(device)
    _run(solver, 3.0)
    assert solver.diagnostics()["sleeping"] == 3
    x, q = solver.body_positions(), solver.body_orientations()

    _run(solver, 10.0)

    np.testing.assert_array_equal(solver.body_positions(), x)
    np.testing.assert_array_equal(solver.body_orientations(), q)
    assert solver.diagnostics()["contacts"] == 0


def test_with_no_gravity_nothing_sleeps(device: str) -> None:
    """A box at rest in zero gravity, touching nothing, is quiet but rests on no static
    solid: it never sleeps, as a body at the top of a throw does not (ADR 0033)."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box((0.0, 0.0, 1.0))])

    _run(solver, 3.0)

    assert solver.diagnostics()["sleeping"] == 0


def test_a_box_sliding_faster_than_the_threshold_stays_awake(device: str) -> None:
    """A box sliding at 0.2 m/s on the floor, μ = 0.01 (it slows by 0.1 m/s²): over 1 s it
    stays faster than 0.05 m/s and does not sleep."""
    solver = _solver(device, mu=0.01)
    solver.add_bodies([_box((0.0, 0.0, 0.1), mu=0.01, velocity_m_per_s=(0.2, 0.0, 0.0))])

    _run(solver, 1.0)

    assert solver.diagnostics()["sleeping"] == 0
    assert solver.body_velocities()[0, 0] > 0.05


def test_a_rolling_ball_stays_awake(device: str) -> None:
    """A ball rolling at 0.5 m/s on the floor, μ = 0.5: it does not sleep in 2 s."""
    solver = _solver(device, mu=0.5)
    solver.add_bodies([Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 0.1),
                            velocity_m_per_s=(0.5, 0.0, 0.0),
                            angular_velocity_rad_per_s=(0.0, 5.0, 0.0),
                            static_friction=0.5, dynamic_friction=0.5)])  # fmt: skip

    _run(solver, 2.0)

    assert solver.diagnostics()["sleeping"] == 0


def test_with_sleeping_off_nothing_sleeps(device: str) -> None:
    """`RigidSetup.sleeping = False`: the resting stack stays awake for 5 s."""
    solver = _stack(device, sleeping=False)

    _run(solver, 5.0)

    assert solver.diagnostics()["sleeping"] == 0
    assert not solver.checkpoint()["sleep_asleep"].any()


def test_a_pile_falls_asleep(device: str) -> None:
    """Forty boxes of random sizes dropped into a pile on the floor, μ = 0.5: every one
    sleeps within 15 s, the pile's slow creep frozen (Matt's decision, ADR 0033)."""
    rng = np.random.default_rng(0)
    solver = _solver(device, max_bodies=64, mu=0.5)
    solver.add_bodies(
        [
            _box((rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.2 + 0.25 * k / 4),
                 half=tuple(rng.uniform(0.04, 0.1, 3)), mu=0.5)
            for k in range(40)
        ]
    )  # fmt: skip

    _run(solver, 15.0)

    assert solver.diagnostics()["sleeping"] == 40

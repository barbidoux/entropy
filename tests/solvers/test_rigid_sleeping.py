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


def _asleep(solver: RigidSolver) -> np.ndarray:
    return solver.checkpoint()["sleep_asleep"].astype(bool)


def test_a_box_dropped_on_a_sleeping_stack_wakes_it(device: str) -> None:
    """A box dropped 0.5 m onto a sleeping stack wakes every body of it in the frame of
    the contact, and the top box takes the blow (ADR 0033)."""
    solver = _stack(device)
    _run(solver, 3.0)
    assert _asleep(solver).all()
    top = solver.body_positions()[2].copy()
    solver.add_bodies([_box((0.0, 0.0, 1.1), mu=0.5)])
    woke = False
    for _ in range(30):
        solver.step(1.0 / 60.0)
        asleep = _asleep(solver)
        if not asleep[:3].any():
            woke = True
            break
        assert asleep[:3].all()  # whole or not at all

    assert woke
    _run(solver, 0.2)
    assert np.linalg.norm(solver.body_positions()[2] - top) > 1e-4  # it took the blow


def _blow(device: str, sleeping: bool) -> tuple[np.ndarray, float]:
    """A box dropped 0.5 m on a 3-stack: where every body ends 2 s later, and the fastest
    speed any of them reached."""
    solver = _stack(device, sleeping=sleeping)
    _run(solver, 3.0)
    solver.add_bodies([_box((0.0, 0.0, 1.1), mu=0.5)])
    fastest = 0.0
    for _ in range(120):
        solver.step(1.0 / 60.0)
        fastest = max(fastest, float(np.linalg.norm(solver.body_velocities()[:3], axis=1).max()))
    return solver.body_positions(), fastest


def test_a_sleeping_stack_takes_a_blow_as_an_awake_one(device: str) -> None:
    """The same drop on a sleeping stack and on an awake one: every body ends within 1 mm
    of where it ends awake, and the stack moves no faster than 1.1 times the awake case:
    waking costs no pop (ADR 0033)."""
    asleep_end, asleep_fastest = _blow(device, sleeping=True)
    awake_end, awake_fastest = _blow(device, sleeping=False)

    assert np.linalg.norm(asleep_end - awake_end, axis=1).max() < 1e-3
    assert asleep_fastest <= 1.1 * awake_fastest + 1e-3


def test_setting_a_velocity_wakes_the_whole_island(device: str) -> None:
    """A tool setting the top box's velocity wakes the whole sleeping stack at once, and
    holds it awake that frame."""
    solver = _stack(device)
    _run(solver, 3.0)
    assert _asleep(solver).all()

    solver.set_body_velocity(2, (0.0, 0.0, 0.0))

    assert not _asleep(solver).any()
    solver.step(1.0 / 60.0)
    assert not _asleep(solver).any()


def test_wake_in_box_wakes_the_islands_it_meets(device: str) -> None:
    """Two sleeping stacks 2 m apart: a box around the first wakes it whole and leaves the
    other asleep."""
    solver = _solver(device, mu=0.5)
    solver.add_bodies([_box((x, 0.0, 0.1 + 0.2 * k), mu=0.5) for x in (0.0, 2.0) for k in range(3)])
    _run(solver, 3.0)
    assert _asleep(solver).all()

    solver.wake_in_box((-0.5, -0.5, 0.45), (0.5, 0.5, 0.6))  # meets the first's top box

    asleep = _asleep(solver)
    assert not asleep[:3].any()
    assert asleep[3:].all()


def _crush(device: str, sleeping: bool, drop_m: float) -> tuple[float, float]:
    """A box dropped `drop_m` on a 3-stack: the smallest gap between the top and middle
    boxes' centres (0.2 m at rest), and the bottom box's lowest centre (0.1 m at rest)."""
    solver = _stack(device, sleeping=sleeping)
    _run(solver, 3.0)
    solver.add_bodies([_box((0.0, 0.0, 0.6 + drop_m), mu=0.5)])
    gap, low = 1.0, 1.0
    for _ in range(120):
        solver.step(1.0 / 60.0)
        x = solver.body_positions()
        gap, low = min(gap, float(x[2, 2] - x[1, 2])), min(low, float(x[0, 2]))
    return gap, low


def test_a_struck_sleeping_stack_is_held_as_an_awake_one(device: str) -> None:
    """A box dropped 0.5 m and 2 m on a sleeping stack: the stack is woken in the substep
    of the contact and held from then on, so it sinks no deeper than an awake one, within
    0.5 mm, and the bottom box stays on the floor. Woken only at the frame's end, the
    top box sank 3.2 mm deeper into the middle one and the bottom one 0.24 mm into the
    floor (ADR 0033)."""
    for drop_m in (0.5, 2.0):
        asleep_gap, asleep_low = _crush(device, True, drop_m)
        awake_gap, awake_low = _crush(device, False, drop_m)

        assert asleep_gap >= awake_gap - 5e-4
        assert asleep_low >= awake_low - 5e-5


def test_a_checkpoint_keeps_the_sleep_state() -> None:
    """The sleep state is state (ADR 0033): restored from a checkpoint taken while a stack
    falls asleep, the solver moves on exactly as it did, bit for bit on the CPU."""
    solver = _stack("cpu")
    _run(solver, 1.2)
    saved = solver.checkpoint()
    _run(solver, 2.0)
    expected = solver.body_positions(), _asleep(solver)
    assert expected[1].all()

    solver.restore(saved)
    _run(solver, 2.0)

    np.testing.assert_array_equal(solver.body_positions(), expected[0])
    np.testing.assert_array_equal(_asleep(solver), expected[1])


def test_a_checkpoint_without_sleep_restores_everything_awake() -> None:
    """A checkpoint that carries no sleep state (an older save) restores every body awake,
    its quiet timer at 0."""
    solver = _stack("cpu")
    _run(solver, 3.0)
    state = {k: v for k, v in solver.checkpoint().items() if not k.startswith("sleep_")}

    solver.restore(state)

    assert not _asleep(solver).any()
    assert not solver.checkpoint()["sleep_timer_s"].any()


def test_a_dump_replays_a_frame_with_sleeping_bodies(tmp_path) -> None:
    """A NaN dump holds the sleep state before its frame: replayed, a sleeping stack stays
    exactly where it slept, bit for bit on the CPU (rule 8)."""
    import math

    from entropy.solvers.interface import NonFiniteStateError
    from entropy.solvers.rigid import load_dump

    floor = FLOOR.to_device("cpu")
    setup = RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=8, dump_dir=tmp_path, solids=floor,
                       static_friction=0.5, dynamic_friction=0.5)  # fmt: skip
    solver = RigidSolver.create(setup, "cpu")
    solver.add_bodies([_box((0.0, 0.0, 0.1 + 0.2 * k), mu=0.5) for k in range(3)])
    solver.add_bodies([_box((3.0, 0.0, 5.0))])
    _run(solver, 3.0)
    assert _asleep(solver)[:3].all()
    solver.set_body_velocity(3, (math.nan, 0.0, 0.0))
    solver.step(1.0 / 60.0)
    try:
        solver.step(1.0 / 60.0)
    except NonFiniteStateError as raised:
        dump = load_dump(raised.dump_path)
    else:
        raise AssertionError("the NaN was not caught")

    replay = RigidSolver.from_dump(dump, "cpu")
    replay.step(float(dump["dt_s"]))

    np.testing.assert_array_equal(replay.body_positions()[:3], dump["positions_m"][:3])
    assert replay.checkpoint()["sleep_asleep"][:3].all()

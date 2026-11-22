"""Rigid contacts (`entropy.solvers.rigid`, SPEC 5.4, #87, ADR 0031): bodies against the
static solids and against each other, solved as XPBD position constraints. V-RB-04 is in
`test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import Box, HalfSpace, OpenBox, Solids

GRAVITY = Environment().gravity_m_per_s2
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))
DENSITY_KG_PER_M3 = 600.0  # any density will do
LYING = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # a capsule's axis along y
ALONG_X = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # a capsule's axis along x
TANK = Solids((OpenBox((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 0.02),))  # the Test Tank's 2 cm glass


def _solver(device: str, solids: Solids | None = FLOOR, gravity=GRAVITY, **kw) -> RigidSolver:
    setup = RigidSetup(
        gravity_m_per_s2=gravity,
        max_bodies=64,
        solids=solids.to_device(device) if solids is not None else None,
        **kw,
    )
    return RigidSolver.create(setup, device)


def _run(solver: RigidSolver, seconds: float) -> None:
    for _ in range(round(seconds * 60)):
        solver.step(1.0 / 60.0)


def _lowest_z(body: Body, position, orientation) -> float:
    """The lowest point of a body's surface (m)."""
    x, y, z, w = orientation
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    if body.shape == Shape.SPHERE:
        return float(position[2] - body.size_m[0])
    if body.shape == Shape.CAPSULE:
        ends = [position + rotation @ (0.0, 0.0, s * body.size_m[1]) for s in (-1, 1)]
        return float(min(e[2] for e in ends) - body.size_m[0])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return float(min((position + rotation @ (c * np.asarray(body.size_m)))[2] for c in corners))


SHAPES = {
    "box": Body(Shape.BOX, (0.3, 0.2, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.6)),
    "sphere": Body(Shape.SPHERE, (0.15, 0.0, 0.0), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.6)),
    "capsule": Body(
        Shape.CAPSULE,
        (0.1, 0.25, 0.0),
        DENSITY_KG_PER_M3,
        position_m=(0.0, 0.0, 0.6),
        orientation=LYING,
    ),
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_a_body_dropped_on_the_floor_comes_to_rest_on_it(device: str, name: str) -> None:
    """Dropped from 0.5 m: after 2 s it rests on the floor, sunk by under 0.3 mm, still to
    within 1 mm/s."""
    body = SHAPES[name]
    solver = _solver(device)
    solver.add_bodies([body])

    _run(solver, 2.0)

    lowest = _lowest_z(body, solver.body_positions()[0], solver.body_orientations()[0])
    assert -3e-4 <= lowest <= 3e-4
    assert np.abs(solver.body_velocities()[0]).max() < 1e-3
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-2


def test_a_tilted_box_lands_flat_and_stops_turning(device: str) -> None:
    """A box dropped at a tilt on a frictionless floor tumbles onto its large face and
    stops turning: the contacts' rotations are right (#101's review)."""
    q = np.array([0.2, 0.3, 0.1, 1.0])
    q /= np.linalg.norm(q)
    solver = _solver(device)
    solver.add_bodies([Body(Shape.BOX, (0.3, 0.2, 0.1), DENSITY_KG_PER_M3,
                            position_m=(0.0, 0.0, 0.8), orientation=tuple(q))])  # fmt: skip

    _run(solver, 4.0)

    x, y = solver.body_orientations()[0, :2]
    assert abs(1 - 2 * (x * x + y * y)) > 1 - 1e-4  # its short axis vertical
    assert solver.body_positions()[0, 2] == pytest.approx(0.1, abs=1e-3)
    assert np.abs(solver.body_angular_velocities()[0]).max() < 0.05


@pytest.mark.parametrize("name", list(SHAPES))
def test_a_body_comes_to_rest_on_a_box_on_the_floor(device: str, name: str) -> None:
    """Bodies against bodies: a body dropped on a wide box resting on the floor; neither
    sinks into the other by more than 1 mm."""
    base = Body(Shape.BOX, (0.6, 0.6, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1))
    top = SHAPES[name]
    solver = _solver(device)
    raised = Body(top.shape, top.size_m, top.density_kg_per_m3, position_m=(0.0, 0.0, 0.7),
                  orientation=top.orientation)  # fmt: skip
    solver.add_bodies([base, raised])

    _run(solver, 2.0)

    positions, orientations = solver.body_positions(), solver.body_orientations()
    assert _lowest_z(base, positions[0], orientations[0]) >= -1e-3
    lowest_top = _lowest_z(top, positions[1], orientations[1])
    assert 0.2 - 1e-3 <= lowest_top <= 0.2 + 2e-3
    assert np.abs(solver.body_velocities()).max() < 1e-3


PAIRS = [("sphere", "sphere"), ("sphere", "box"), ("capsule", "box"), ("box", "box"),
         ("capsule", "sphere"), ("capsule", "capsule")]  # fmt: skip


@pytest.mark.parametrize(("a", "b"), PAIRS)
def test_bodies_thrown_at_each_other_do_not_pass_through(device: str, a: str, b: str) -> None:
    """No gravity, no solids: two bodies meet head-on at 4 m/s each. They stay stuck
    together (restitution 0 by default): each on its own side, overlapping by under 1 mm,
    moving apart at under 5 cm/s."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    first, second = SHAPES[a], SHAPES[b]
    solver.add_bodies(
        [
            Body(first.shape, first.size_m, 600.0, position_m=(-1.0, 0.0, 0.0),
                 orientation=first.orientation, velocity_m_per_s=(4.0, 0.0, 0.0)),
            Body(second.shape, second.size_m, 600.0, position_m=(1.0, 0.0, 0.0),
                 orientation=second.orientation, velocity_m_per_s=(-4.0, 0.0, 0.0)),
        ]
    )  # fmt: skip

    _run(solver, 1.0)

    x = solver.body_positions()[:, 0]
    reach = {"sphere": 0.15, "box": 0.3, "capsule": 0.1}  # half extent along x
    assert x[0] < x[1]
    assert x[1] - x[0] >= reach[a] + reach[b] - 1e-3
    velocities = solver.body_velocities()
    assert np.linalg.norm(velocities[1] - velocities[0]) < 0.05


def test_bodies_spawned_overlapping_do_not_fly_apart(device: str) -> None:
    """Two boxes spawned 5 cm into each other, and one 5 cm into the floor, are pushed out
    without the push turning into speed: under 1 cm/s (it was 11 m/s before the velocity
    pass), as the spawner may do (#93)."""
    solver = _solver(device)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(-0.075, 0.0, 0.1)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.075, 0.0, 0.1)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(3.0, 0.0, 0.05)),
        ]
    )

    for _ in range(30):
        solver.step(1.0 / 60.0)
        assert np.abs(solver.body_velocities()).max() < 0.01

    positions = solver.body_positions()
    assert positions[1, 0] - positions[0, 0] >= 0.2 - 1e-3
    assert positions[2, 2] == pytest.approx(0.1, abs=1e-3)


def test_each_pair_is_met_once(device: str) -> None:
    """A sphere resting on a sphere on the floor: one contact with the floor and one
    between the two (a sphere's sample against the other's exact distance), 2 in all;
    meeting the pair twice would make 3."""
    solver = _solver(device)
    ball = (0.1, 0.0, 0.0)
    solver.add_bodies(
        [
            Body(Shape.SPHERE, ball, DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1)),
            Body(Shape.SPHERE, ball, DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.3)),
        ]
    )

    _run(solver, 0.5)
    solver.max_advance_s()  # reads the last step's status

    assert solver.diagnostics()["contacts"] == 2


def test_a_small_body_first_in_the_list_finds_a_big_one(device: str) -> None:
    """The broad phase with unequal sizes: a small sphere (body 0) falls on the corner of a
    wide box (body 1) and rests on it."""
    solver = _solver(device)
    density = DENSITY_KG_PER_M3
    solver.add_bodies(
        [
            Body(Shape.SPHERE, (0.05, 0.0, 0.0), density, position_m=(0.9, 0.9, 0.4)),
            Body(Shape.BOX, (1.0, 1.0, 0.1), density, position_m=(0.0, 0.0, 0.1)),
        ]
    )

    _run(solver, 1.0)

    assert solver.body_positions()[0, 2] == pytest.approx(0.25, abs=2e-3)


def test_a_long_capsule_does_not_fall_through_a_thin_beam(device: str) -> None:
    """A capsule of 5 cm radius and 1 m long dropped across two 8 cm beams, each between
    two of its samples' old places (25 cm apart): it rests on them (its samples are now
    at most a radius apart)."""
    beams = Solids(
        (Box((-0.125, 0.0, 0.5), (0.04, 1.0, 0.02)), Box((0.125, 0.0, 0.5), (0.04, 1.0, 0.02)))
    )
    solver = _solver(device, solids=beams)
    solver.add_bodies([Body(Shape.CAPSULE, (0.05, 0.5, 0.0), DENSITY_KG_PER_M3,
                            position_m=(0.0, 0.0, 0.6), orientation=ALONG_X)])  # fmt: skip

    _run(solver, 0.5)

    assert solver.body_positions()[0, 2] > 0.5


def test_a_long_capsule_does_not_fall_through_another(device: str) -> None:
    """Such a capsule dropped across two others lying on the floor, each crossing midway
    between two samples of both: it rests on them, lower by at most the sag of the
    samples' envelope there, (1 - cos 30°) r = 6.7 mm (named in ADR 0031)."""
    solver = _solver(device)
    capsule = (0.05, 0.5, 0.0)
    solver.add_bodies(
        [
            Body(Shape.CAPSULE, capsule, DENSITY_KG_PER_M3, position_m=(0.0, -0.125, 0.05),
                 orientation=ALONG_X),
            Body(Shape.CAPSULE, capsule, DENSITY_KG_PER_M3, position_m=(0.0, 0.125, 0.05),
                 orientation=ALONG_X),
            Body(Shape.CAPSULE, capsule, DENSITY_KG_PER_M3, position_m=(0.125, 0.0, 0.3),
                 orientation=LYING),
        ]
    )  # fmt: skip

    _run(solver, 1.0)

    sag = (1.0 - math.cos(math.pi / 6)) * 0.05
    assert 0.15 - sag - 1e-3 <= solver.body_positions()[2, 2] <= 0.15 + 1e-3


def test_the_tank_s_walls_keep_a_thrown_sphere_in(device: str) -> None:
    """The Test Tank's open box (the world's solids): a sphere thrown at 6 m/s along the
    floor stays within the cavity."""
    tank = Solids((OpenBox((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 0.02),))
    solver = _solver(device, solids=tank)
    solver.add_bodies([Body(Shape.SPHERE, (0.2, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 0.5),
                            velocity_m_per_s=(6.0, 1.0, 0.0))])  # fmt: skip

    for _ in range(120):
        solver.step(1.0 / 60.0)
        x, y, z = solver.body_positions()[0]
        assert -1.8 - 1e-3 <= x <= 1.8 + 1e-3 and -1.8 - 1e-3 <= y <= 1.8 + 1e-3
        assert z >= 0.2 - 1e-3


@pytest.mark.parametrize("speed_m_per_s", [5.0, 8.0, 12.0])
@pytest.mark.parametrize("tumbling", [False, True])
def test_the_tank_s_glass_keeps_a_thrown_box_in(
    device: str, speed_m_per_s: float, tumbling: bool
) -> None:
    """A 20 cm box thrown at the Test Tank's 2 cm glass, face on or tumbling, at up to
    12 m/s (a fall of 7 m): it stays in the cavity and comes back no faster than it came
    (#101's review)."""
    solver = _solver(device, solids=TANK)
    spin = (4.0, 6.0, 9.0) if tumbling else (0.0, 0.0, 0.0)
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=(1.0, 0.0, 0.3 if tumbling else 0.1),
                            velocity_m_per_s=(speed_m_per_s, 0.0, 0.0),
                            angular_velocity_rad_per_s=spin)])  # fmt: skip

    for _ in range(120):
        solver.step(1.0 / 60.0)
        x, y, z = solver.body_positions()[0]
        assert abs(x) < 2.0 and abs(y) < 2.0 and z > 0.0  # its centre never in the glass

    assert abs(solver.body_positions()[0, 0]) <= 1.9 + 2e-3
    assert np.linalg.norm(solver.body_velocities()[0]) <= speed_m_per_s


def test_contacts_past_the_capacity_are_counted_not_lost_silently(device: str) -> None:
    """With room for 4 contacts between bodies, a small box resting on a wide one (its 9
    bottom samples in) is still held, and the overflow of the last step is counted for the
    HUD: one step's, not a sum that grows (and wraps), and cleared by a restore. The
    contacts with the static solids need no room."""
    solver = _solver(device, max_contacts=4)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.3, 0.3, 0.1), 600.0, position_m=(0.0, 0.0, 0.1)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), 600.0, position_m=(0.0, 0.0, 0.3)),
        ]
    )

    _run(solver, 0.5)
    solver.max_advance_s()

    assert solver.diagnostics()["contact_overflows"] > 0
    # One step's at most: 8 substeps of the pair's 52 samples, past the 4 places.
    assert solver.diagnostics()["contact_overflows"] <= 8 * (52 - 4)
    positions = solver.body_positions()
    assert positions[0, 2] > 0.099 and positions[1, 2] - positions[0, 2] > 0.15
    solver.restore({**solver.checkpoint(), "count": 0})
    assert solver.diagnostics()["contacts"] == 0
    assert solver.diagnostics()["contact_overflows"] == 0

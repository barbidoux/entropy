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
    together (no restitution yet, #88): each on its own side, overlapping by under 1 mm,
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


@pytest.mark.parametrize(
    "throw",
    [(6.0, 1.0, 0.0), (-6.0, -1.0, 0.0), (1.0, 6.0, 0.0), (-1.0, -6.0, 0.0)],
    ids=["+x", "-x", "+y", "-y"],
)
def test_the_tank_s_walls_keep_a_thrown_sphere_in(device: str, throw) -> None:
    """The Test Tank's open box (the world's solids): a sphere thrown at 6 m/s along the
    floor at each of its four walls stays within the cavity (each wall is its own plane,
    #101's fourth review)."""
    tank = Solids((OpenBox((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 0.02),))
    solver = _solver(device, solids=tank)
    solver.add_bodies([Body(Shape.SPHERE, (0.2, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 0.5),
                            velocity_m_per_s=throw)])  # fmt: skip

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


def test_pairs_past_the_capacity_are_counted_not_lost_silently(device: str) -> None:
    """With room for 1 pair of bodies, two small boxes resting on a wide one make 2: one is
    dropped each substep and counted for the HUD, as one step's count, not a sum that
    grows (and wraps), and cleared by a restore. The contacts with the static solids need
    no room."""
    solver = _solver(device, max_pairs=1)
    small = (0.1, 0.1, 0.1)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.5, 0.5, 0.1), 600.0, position_m=(0.0, 0.0, 0.1)),
            Body(Shape.BOX, small, 600.0, position_m=(-0.25, 0.0, 0.3)),
            Body(Shape.BOX, small, 600.0, position_m=(0.25, 0.0, 0.3)),
        ]
    )

    _run(solver, 0.5)
    solver.max_advance_s()
    first = solver.diagnostics()["pair_overflows"]
    solver.step(1.0 / 60.0)
    solver.max_advance_s()

    assert first == 8  # one pair dropped in each of the step's 8 substeps
    assert solver.diagnostics()["pair_overflows"] == first
    solver.restore({**solver.checkpoint(), "count": 0})
    assert solver.diagnostics()["contacts"] == 0
    assert solver.diagnostics()["pair_overflows"] == 0


@pytest.mark.parametrize("overhang_m", [0.01, 0.05, 0.09])
@pytest.mark.parametrize("drop_m", [0.0, 0.05])
def test_a_box_over_a_table_s_edge_stays_on_it(
    device: str, overhang_m: float, drop_m: float
) -> None:
    """A 20 cm box over the edge of a static table by 1, 5 or 9 cm (under half its width),
    placed or dropped from 5 cm, stays on it for 10 s: the table's edge under its face
    holds it (#101's review: it toppled off from 1 cm)."""
    table = Solids(
        (Box((0.0, 0.0, 0.4), (0.5, 0.5, 0.4)), HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    )
    solver = _solver(device, solids=table)
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=(0.4 + overhang_m, 0.0, 0.9 + drop_m))])  # fmt: skip

    _run(solver, 10.0)

    position = solver.body_positions()[0]
    assert position[2] == pytest.approx(0.9, abs=1e-3)
    assert position[0] == pytest.approx(0.4 + overhang_m, abs=1e-3)


def test_a_box_dropped_tilted_on_an_equal_one_ends_level(device: str) -> None:
    """A box dropped 1° tilted on an equal box comes to lie level on it (under 0.1°): the
    pair shares its axis of least overlap as the normal (7.6° without it)."""
    a = math.radians(1.0) / 2.0
    solver = _solver(device)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.31),
                 orientation=(0.6 * math.sin(a), 0.8 * math.sin(a), 0.0, math.cos(a))),
        ]
    )  # fmt: skip

    _run(solver, 5.0)

    x, y = solver.body_orientations()[1, :2]
    assert math.degrees(2.0 * max(abs(x), abs(y))) < 0.1


@pytest.mark.parametrize("name", ["sphere", "capsule"])
def test_a_curved_body_stays_put_on_a_box(device: str, name: str) -> None:
    """A sphere, or a capsule lying, resting on a box on the floor for 10 s, moves under
    1 mm: against a sphere only its own sample is tested, and a box tests only its
    corners and edges against a capsule (its face points would push sideways)."""
    base = Body(Shape.BOX, (0.6, 0.6, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1))
    top = SHAPES[name]
    solver = _solver(device)
    solver.add_bodies([base, Body(top.shape, top.size_m, top.density_kg_per_m3,
                                  position_m=(0.0, 0.0, 0.2 + (0.15 if name == "sphere" else 0.1)),
                                  orientation=top.orientation)])  # fmt: skip
    _run(solver, 1.0)
    settled = solver.body_positions()[1].copy()

    _run(solver, 9.0)

    assert np.linalg.norm(solver.body_positions()[1] - settled) < 1e-3
    assert np.abs(solver.body_velocities()).max() < 1e-3


def test_nine_boxes_rest_on_one_slab(device: str) -> None:
    """Nine boxes resting on one slab, the slab in nine pairs at once: it neither jitters
    nor sinks, and each box rests on it. (Solved contact by contact in parallel, it
    jittered and sank, ADR 0031. The pairs' 1 / n scale is not what holds it here:
    unscaled, it holds too, #101's fourth review.)"""
    solver = _solver(device)
    small, density = (0.1, 0.1, 0.1), DENSITY_KG_PER_M3
    boxes = [Body(Shape.BOX, small, density, position_m=(0.25 * ix, 0.25 * iy, 0.3))
             for ix in (-1, 0, 1) for iy in (-1, 0, 1)]  # fmt: skip
    slab = Body(Shape.BOX, (0.5, 0.5, 0.1), density, position_m=(0.0, 0.0, 0.1))
    solver.add_bodies([slab, *boxes])

    _run(solver, 3.0)

    positions = solver.body_positions()
    assert positions[0, 2] == pytest.approx(0.1, abs=3e-4)
    np.testing.assert_allclose(positions[1:, 2], 0.3, atol=1e-3)
    assert np.abs(solver.body_velocities()).max() < 1e-3


def test_bodies_already_parting_keep_their_speed(device: str) -> None:
    """No gravity: two boxes spawned 1 cm into each other and already moving apart at
    1 m/s each are pushed out and keep parting at 2 m/s: the velocity pass never slows a
    separation under way."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(-0.095, 0.0, 0.0),
                 velocity_m_per_s=(-1.0, 0.0, 0.0)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.095, 0.0, 0.0),
                 velocity_m_per_s=(1.0, 0.0, 0.0)),
        ]
    )  # fmt: skip

    _run(solver, 0.5)

    velocities = solver.body_velocities()
    assert velocities[1, 0] - velocities[0, 0] == pytest.approx(2.0, rel=1e-3)


SLABS = Solids(
    (Box((-5.0, 0.0, -0.15), (5.0, 5.0, 0.15)), Box((5.0, 0.0, -0.15), (5.0, 5.0, 0.15)))
)
"""Two floor slabs side by side, flush, meeting at x = 0 (a map built of boxes)."""


@pytest.mark.parametrize("speed_m_per_s", [0.5, 1.0, 4.0])
def test_a_box_slides_across_the_seam_of_two_slabs(device: str, speed_m_per_s: float) -> None:
    """A box sliding (frictionless) across the seam of two flush slabs keeps its speed:
    the slabs' inner faces are not walls (#101's third review: it bounced back)."""
    solver = _solver(device, solids=SLABS)
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=(-1.0031, 0.0, 0.1),
                            velocity_m_per_s=(speed_m_per_s, 0.0, 0.0))])  # fmt: skip

    _run(solver, 2.5 / speed_m_per_s)

    v = solver.body_velocities()[0]
    assert solver.body_positions()[0, 0] > 1.0
    np.testing.assert_allclose(v, (speed_m_per_s, 0.0, 0.0), atol=0.01 * speed_m_per_s)


def test_a_box_slides_along_a_wall_of_two_boxes(device: str) -> None:
    """A box sliding along a wall built of two flush boxes keeps its speed and is not
    pushed off the wall at their seam."""
    wall = Solids(
        (
            FLOOR.shapes[0],
            Box((-1.0, 0.0, 0.5), (1.0, 0.1, 0.5)),
            Box((1.0, 0.0, 0.5), (1.0, 0.1, 0.5)),
        )
    )
    solver = _solver(device, solids=wall)
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=(-1.0013, 0.2005, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0))])  # fmt: skip

    _run(solver, 1.0)

    np.testing.assert_allclose(solver.body_velocities()[0], (2.0, 0.0, 0.0), atol=0.02)


def test_a_box_falling_beside_a_table_does_not_touch_it(device: str) -> None:
    """A 10 cm box falling from 2 m, 1 mm beside a table's side face, falls freely past
    its edge: no contact before the floor (#101's third review: the edge's diagonal
    tangent plane kicked it sideways)."""
    table = Solids((FLOOR.shapes[0], Box((0.0, 0.0, 0.4), (0.5, 0.5, 0.4))))
    solver = _solver(device, solids=table)
    solver.add_bodies([Body(Shape.BOX, (0.05, 0.05, 0.05), DENSITY_KG_PER_M3,
                            position_m=(0.551, 0.0, 2.0013))])  # fmt: skip

    for _ in range(140):  # 0.58 s: past the table's top edge, still above the floor
        solver.step(1.0 / 240.0)

    v = solver.body_velocities()[0]
    assert abs(v[0]) < 1e-6 and abs(v[1]) < 1e-6
    assert v[2] == pytest.approx(GRAVITY[2] * 140 / 240.0, rel=1e-3)
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-6


def test_a_box_sliding_past_a_pillar_does_not_touch_it(device: str) -> None:
    """A box sliding at 3 m/s past a pillar's corner, 0.5 mm off its face, keeps its
    velocity and does not turn."""
    pillar = Solids((FLOOR.shapes[0], Box((0.0, 0.0, 1.0), (0.2, 0.2, 1.0))))
    solver = _solver(device, solids=pillar)
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=(-1.0013, 0.3005, 0.1),
                            velocity_m_per_s=(3.0, 0.0, 0.0))])  # fmt: skip

    _run(solver, 1.0)

    np.testing.assert_allclose(solver.body_velocities()[0], (3.0, 0.0, 0.0), atol=1e-3)
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-3


def test_a_heavy_box_on_a_stack_does_not_jitter(device: str) -> None:
    """A box 100 times heavier on a stack of three: after it settles, it moves under
    20 mm from 5 to 10 s (the pairs' 1 / (larger pair count) scaling; unscaled it
    jitters 82 mm)."""
    solver = _solver(device)
    cube = (0.1, 0.1, 0.1)
    stack = [Body(Shape.BOX, cube, 600.0, position_m=(0.0, 0.0, 0.1 + 0.2 * k)) for k in range(3)]
    solver.add_bodies([*stack, Body(Shape.BOX, cube, 60000.0, position_m=(0.0, 0.0, 0.7))])
    _run(solver, 5.0)
    heights = []
    for _ in range(300):
        solver.step(1.0 / 60.0)
        heights.append(solver.body_positions()[3, 2])

    assert max(heights) - min(heights) < 0.02


def test_a_long_plank_on_its_edge_over_a_table_stays(device: str) -> None:
    """A 1 m plank 2 cm thick standing on its narrow side, 35 % of it over a table's
    edge, stays for 15 s: the table's edge is tested along the plank's own extent, at
    points under its narrow face (#101's third review: it fell at 11 s)."""
    table = Solids((FLOOR.shapes[0], Box((0.0, 0.0, 0.4), (0.5, 0.5, 0.4))))
    solver = _solver(device, solids=table)
    solver.add_bodies([Body(Shape.BOX, (0.5, 0.01, 0.05), DENSITY_KG_PER_M3,
                            position_m=(0.35, 0.023, 0.85))])  # fmt: skip

    _run(solver, 15.0)

    assert solver.body_positions()[0, 2] == pytest.approx(0.85, abs=2e-3)


def test_a_body_above_the_tank_s_rim_flies_over_the_glass(device: str) -> None:
    """A ball inside the tank's column but above its rim, thrown sideways, flies over
    the glass: the walls' planes stop at the rim."""
    solver = _solver(device, solids=TANK, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([Body(Shape.SPHERE, (0.1, 0.0, 0.0), DENSITY_KG_PER_M3,
                            position_m=(1.5, 0.0, 2.2),
                            velocity_m_per_s=(3.0, 0.0, 0.0))])  # fmt: skip

    _run(solver, 1.0)

    assert solver.body_positions()[0, 0] > 4.0


def test_a_box_at_rest_does_not_start_turning(device: str) -> None:
    """A box resting on the floor for 20 s keeps still about the vertical (|w| under
    1e-4 rad/s): within a static sweep the body turns to first order, once at the end."""
    solver = _solver(device)
    solver.add_bodies([Body(Shape.BOX, (0.3, 0.2, 0.1), DENSITY_KG_PER_M3,
                            position_m=(0.0, 0.0, 0.1),
                            orientation=(0.0, 0.0, 0.2, 0.98))])  # fmt: skip

    _run(solver, 20.0)

    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-4


def test_room_for_pairs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_pairs"):
        RigidSetup(gravity_m_per_s2=GRAVITY, max_pairs=0)


def _energy(solver: RigidSolver, body: Body) -> float:
    """Body 0's kinetic and potential energy (J), in float64."""
    from entropy.solvers.rigid import mass_properties

    mass, inertia = mass_properties(body.shape, body.size_m, body.density_kg_per_m3)
    x, y, z, w = solver.body_orientations()[0].astype(np.float64)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    v = solver.body_velocities()[0].astype(np.float64)
    omega = solver.body_angular_velocities()[0].astype(np.float64)
    height = float(solver.body_positions()[0, 2])
    spin = 0.5 * omega @ rotation @ np.diag(inertia) @ rotation.T @ omega
    return 0.5 * mass * float(v @ v) + float(spin) - mass * GRAVITY[2] * height


@pytest.mark.parametrize(
    "half_m", [(0.04, 0.01, 0.0025), (0.08, 0.02, 0.005)], ids=["plank-8cm", "plank-16cm"]
)
def test_thin_planks_thrown_hard_at_the_floor_never_gain_energy(device: str, half_m) -> None:
    """Planks thrown at the floor at 30 m/s from 20 random orientations, no restitution:
    none goes non-finite, and none ends a frame with more energy (kinetic and potential)
    than it was thrown with, 0.5 % allowed for float32. The hit spins them up to some
    thousand rad/s: the gyroscopic step froze the solver with a NaN in 11 of 20 throws,
    and one plank rebounded 13 m with 1.95 times its energy (#101's fourth review)."""
    floor = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)).to_device(device)
    rng = np.random.default_rng(7)
    for _ in range(20):
        aim = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -1.0])
        turn = rng.normal(size=4)
        plank = Body(Shape.BOX, half_m, 600.0, position_m=(0.0, 0.0, 0.5),
                     velocity_m_per_s=tuple(30.0 * aim / np.linalg.norm(aim)),
                     orientation=tuple(turn / np.linalg.norm(turn)))  # fmt: skip
        solver = RigidSolver.create(
            RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=2, solids=floor), device
        )
        solver.add_bodies([plank])
        thrown = _energy(solver, plank)
        for _ in range(90):
            solver.step(1.0 / 60.0)
            assert _energy(solver, plank) <= 1.005 * thrown
        solver.max_advance_s()  # raises if the last step left a non-finite state


@pytest.mark.parametrize(
    ("place", "solid"),
    [
        ((0.0, 0.0, 0.4), "table"),  # 50 cm into the table's top
        ((0.3, 0.0, 0.5), "table"),  # 30 cm into its side
        ((0.1, 0.0, 0.5), "table"),  # 50 cm into its side
        ((-0.1, 0.0, 1.0), "wall"),  # 50 cm into a wall's face
    ],
    ids=["table-top-50cm", "table-side-30cm", "table-side-50cm", "wall-50cm"],
)
def test_a_box_spawned_deep_inside_a_static_box_is_not_flung(device: str, place, solid) -> None:
    """A 20 cm box whose centre starts 30 to 50 cm inside a static box comes out without
    being flung: under 5 m/s and 60 rad/s for 3 s (it may then fall off). The depth
    corrections pushed it out in one substep and the spin they gave it stayed: 57 to
    211 m/s and up to 1,058 rad/s (#101's fourth review)."""
    if solid == "table":
        shapes = (
            HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            Box((0.0, 0.0, 0.4), (0.5, 0.5, 0.4)),
        )
    else:
        shapes = (
            HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            Box((0.0, 0.0, 1.5), (0.3, 3.0, 1.5)),
        )
    solids = Solids(shapes).to_device(device)
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=2, solids=solids),
                                device)  # fmt: skip
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), 600.0, position_m=place)])

    fastest = spun = 0.0
    for _ in range(180):
        solver.step(1.0 / 60.0)
        fastest = max(fastest, float(np.linalg.norm(solver.body_velocities()[0])))
        spun = max(spun, float(np.linalg.norm(solver.body_angular_velocities()[0])))

    assert fastest < 5.0
    assert spun < 60.0


@pytest.mark.parametrize(
    ("half_m", "gap_m"),
    [(0.05, 0.001), (0.05, 0.003), (0.1, 0.001)],
    ids=["10cm-1mm", "10cm-3mm", "20cm-1mm"],
)
def test_a_box_falling_past_the_tank_s_outer_glass_is_untouched(device: str, half_m, gap_m) -> None:
    """A box falling from 3 m beside the Test Tank's outer glass, 1 or 3 mm from it, past
    its rim at 2 m: it falls freely, neither pushed nor turned. The rim's outer edge gave
    it a diagonal plane: knocked sideways at up to 2.1 m/s and spun at 36 rad/s (#101's
    fourth review; the same bug as beside a table, third review)."""
    tank = Solids(
        (
            OpenBox((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 0.02),
            HalfSpace((0.0, 0.0, -0.02), (0.0, 0.0, 1.0)),
        )
    ).to_device(device)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=2, solids=tank), device
    )
    solver.add_bodies(
        [Body(Shape.BOX, (half_m,) * 3, 600.0, position_m=(2.02 + half_m + gap_m, 0.0, 3.0))]
    )

    for _ in range(140):
        solver.step(1.0 / 240.0)

    v = solver.body_velocities()[0]
    assert np.linalg.norm(v[:2]) < 1e-3
    assert v[2] == pytest.approx(GRAVITY[2] * 140 / 240, rel=1e-3)
    assert np.linalg.norm(solver.body_angular_velocities()[0]) < 1e-2


def test_a_sphere_resting_off_centre_on_a_box_stays(device: str) -> None:
    """A sphere resting 1 mm off the centre of a box's top face (a box on the floor) stays
    for 5 s, under 1 mm: the box's face centre, just off the sphere's lowest point, is
    not tested against it, or it would push the sphere sideways and roll it off (#101's
    fourth review: the centred case could not show it)."""
    solver = _solver(device)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.3, 0.3, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1)),
            Body(Shape.SPHERE, (0.1, 0.0, 0.0), DENSITY_KG_PER_M3, position_m=(0.001, 0.0, 0.3)),
        ]
    )
    _run(solver, 0.5)
    settled = solver.body_positions()[1].copy()

    _run(solver, 5.0)

    assert np.linalg.norm(solver.body_positions()[1] - settled) < 1e-3


def test_a_box_thrown_into_a_corner_stops_there(device: str) -> None:
    """A box thrown down into the corner of the floor and a wall, at 4.8 m/s into the wall
    and 2.4 m/s down, meeting both in the same substep: it stops there, without climbing
    the wall. (The floor and the wall are two patches, each with its own normal; the
    position solve already stops this box, so the split is pinned by #88's friction,
    which acts per patch, #101's fifth review.)"""
    room = Solids(
        (HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), HalfSpace((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    )
    solver = _solver(device, solids=room)
    box = Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.89, 0.0, 0.105),
               velocity_m_per_s=(4.8, 0.0, -2.4))  # fmt: skip
    solver.add_bodies([box])

    for _ in range(30):
        solver.step(1.0 / 60.0)
        assert solver.body_positions()[0, 2] < 0.105 + 1e-3  # it never climbs the wall

    assert solver.body_positions()[0, 0] == pytest.approx(0.9, abs=1e-3)
    assert solver.body_positions()[0, 2] == pytest.approx(0.1, abs=1e-3)
    assert np.abs(solver.body_velocities()[0]).max() < 1e-2


@pytest.mark.parametrize(
    ("spawned", "neighbour"),
    [((0.0, 0.0, 0.4), (0.0, 0.0, 0.9)), ((0.3, 0.0, 0.5), (0.3, 0.0, 0.69))],
    ids=["50cm-into-the-top-under-a-box", "30cm-into-the-side-touching-a-box"],
)
def test_a_box_spawned_deep_in_a_table_beside_another_box_is_not_flung(
    device: str, spawned, neighbour
) -> None:
    """As above, with a second box touching the first, so that it is in a pair too:
    neither is flung (under 5 m/s and 60 rad/s for 3 s). Bodies in a pair had no energy
    budget against the static solids, and the one touching the table's side flew at
    135 m/s (#101's fifth review). Two boxes spawned into each other as well pop apart
    (named in ADR 0031)."""
    table = Solids(
        (HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), Box((0.0, 0.0, 0.4), (0.5, 0.5, 0.4)))
    ).to_device(device)
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=4, solids=table),
                                device)  # fmt: skip
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), 600.0, position_m=spawned),
                       Body(Shape.BOX, (0.1, 0.1, 0.1), 600.0, position_m=neighbour)])  # fmt: skip

    fastest = spun = 0.0
    for _ in range(180):
        solver.step(1.0 / 60.0)
        fastest = max(fastest, float(np.linalg.norm(solver.body_velocities(), axis=1).max()))
        spun = max(spun, float(np.linalg.norm(solver.body_angular_velocities(), axis=1).max()))

    assert fastest < 5.0
    assert spun < 60.0


def test_a_box_on_a_frictionless_slope_slides_at_g_sin_theta(device: str) -> None:
    """A box resting flat on a frictionless 30° slope slides at g sin θ t: 24.5 m/s after
    5 s, within 0.05 %. The static solids' energy budget includes gravity's work over the
    substep (without it the box would not move, #101's fifth review), and their
    corrections enter the motion exactly, as increments: taken as a float32 difference
    of positions, the box lost 0.7 % of its speed in 5 s (sixth review)."""
    theta = math.radians(30.0)
    normal = (math.sin(theta), 0.0, math.cos(theta))
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), normal),))
    solver = _solver(device, solids=slope)
    tilt = (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
    place = tuple(0.1 * np.asarray(normal))
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=place,
                            orientation=tilt)])  # fmt: skip

    _run(solver, 5.0)

    speed = float(np.linalg.norm(solver.body_velocities()[0]))
    assert speed == pytest.approx(-GRAVITY[2] * math.sin(theta) * 5.0, rel=5e-4)


def test_a_box_sliding_into_a_resting_one_shares_its_speed(device: str) -> None:
    """A box sliding at 2 m/s along the frictionless floor into an equal one at rest: they
    leave together at 1 m/s each (momentum kept, no restitution), within 5 %. They meet
    in the middle of a substep, so the pair's depth correction gives the struck box its
    speed: capped at its energy without the contact, it left at 0.84 m/s (#101's fifth
    and sixth reviews)."""
    solver = _solver(device)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.1),
                 velocity_m_per_s=(2.0, 0.0, 0.0)),
            Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3, position_m=(0.2611, 0.0, 0.1)),
        ]
    )  # fmt: skip

    _run(solver, 0.5)

    np.testing.assert_allclose(solver.body_velocities()[:, 0], [1.0, 1.0], rtol=0.05)


@pytest.mark.parametrize("name", ["box", "sphere"])
def test_a_body_sliding_far_from_the_origin_keeps_its_speed(device: str, name: str) -> None:
    """A box or a sphere sliding at 5 m/s along a frictionless 30° slope's level line,
    100 m down the slope, gravity pressing it onto the slope: after 10 s it still slides
    at 5 m/s, within 0.1 %. The static solids' corrections enter its motion as
    increments; as a float32 difference of positions, 100 m down, they gave the sphere
    2.2 to 2.8 % (#101's sixth and seventh reviews)."""
    theta = math.radians(30.0)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    out = 100.0 * np.array([math.cos(theta), 0.0, -math.sin(theta)])  # down the slope
    slope = Solids((HalfSpace(tuple(out), tuple(normal)),))
    gravity = tuple(-9.81 * normal)
    solver = _solver(device, solids=slope, gravity=gravity)
    if name == "box":
        tilt = (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
        body = Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                    position_m=tuple(out + 0.1 * normal), orientation=tilt,
                    velocity_m_per_s=(0.0, 5.0, 0.0))  # fmt: skip
    else:
        body = Body(Shape.SPHERE, (0.1, 0.0, 0.0), DENSITY_KG_PER_M3,
                    position_m=tuple(out + 0.1 * normal),
                    velocity_m_per_s=(0.0, 5.0, 0.0))  # fmt: skip
    solver.add_bodies([body])

    _run(solver, 10.0)

    assert float(np.linalg.norm(solver.body_velocities()[0])) == pytest.approx(5.0, rel=1e-3)


@pytest.mark.parametrize("name", ["box", "lying-capsule"])
def test_a_body_spinning_on_the_floor_keeps_its_spin(device: str, name: str) -> None:
    """A box lying flat, or a capsule lying, spinning about the vertical at 10 rad/s on
    the frictionless floor: after 3 s it still spins at 10 rad/s, within 1e-3. The static
    solids' energy budget counts the spin, in the body's frame: without the spin, every
    spin stopped at its first contact; with the spin in the world frame, the lying
    capsule (whose vertical is not its body's) lost 99 % of it (#101's sixth and seventh
    reviews)."""
    solver = _solver(device)
    if name == "box":
        body = Body(Shape.BOX, (0.2, 0.1, 0.05), DENSITY_KG_PER_M3, position_m=(0.0, 0.0, 0.05),
                    angular_velocity_rad_per_s=(0.0, 0.0, 10.0))  # fmt: skip
    else:
        body = Body(Shape.CAPSULE, (0.05, 0.1, 0.0), DENSITY_KG_PER_M3,
                    position_m=(0.0, 0.0, 0.05), orientation=LYING,
                    angular_velocity_rad_per_s=(0.0, 0.0, 10.0))  # fmt: skip
    solver.add_bodies([body])

    _run(solver, 3.0)

    np.testing.assert_allclose(solver.body_angular_velocities()[0], [0.0, 0.0, 10.0], atol=1e-2)


def test_a_stack_resting_on_a_frictionless_slope_stays_stacked(device: str) -> None:
    """Two 20 cm boxes stacked on a frictionless 30° slope at the origin, gravity pressing
    them onto it (along its normal), so nothing pushes them along it: after 10 s the
    upper box is within 1 mm of where it sat on the lower one. The pairs' corrections
    enter the motion as increments; as a float32 difference of positions they pushed it
    5.9 mm on CUDA and 49 mm on the CPU, and from 10 m down the slope off the lower box
    (#101's seventh review)."""
    theta = math.radians(30.0)
    normal = np.array([math.sin(theta), 0.0, math.cos(theta)])
    down = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), tuple(normal)),))
    solver = _solver(device, solids=slope, gravity=tuple(-9.81 * normal))
    tilt = (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
    solver.add_bodies([Body(Shape.BOX, (0.1, 0.1, 0.1), DENSITY_KG_PER_M3,
                            position_m=tuple(height * normal), orientation=tilt)
                       for height in (0.1, 0.3)])  # fmt: skip

    _run(solver, 10.0)

    x = solver.body_positions()
    assert abs(float((x[1] - x[0]) @ down)) < 1e-3

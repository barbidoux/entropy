"""Rigid contacts (`entropy.solvers.rigid`, SPEC 5.4, #87, ADR 0031): bodies against the
static solids and against each other, solved as XPBD position constraints. V-RB-04 is in
`test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, OpenBox, Solids

GRAVITY = Environment().gravity_m_per_s2
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))
DENSITY_KG_PER_M3 = 600.0  # any density will do
LYING = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # a capsule's axis along y


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
    """Dropped from 0.5 m: after 2 s it rests on the floor, sunk by under 1 mm, still to
    within 1 mm/s."""
    body = SHAPES[name]
    solver = _solver(device)
    solver.add_bodies([body])

    _run(solver, 2.0)

    lowest = _lowest_z(body, solver.body_positions()[0], solver.body_orientations()[0])
    assert -1e-3 <= lowest <= 1e-3
    assert np.abs(solver.body_velocities()[0]).max() < 1e-3
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-2


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
    """No gravity, no solids: two bodies meet head-on at 4 m/s each. They stop, stuck
    together (no restitution yet, #88), each on its own side, overlapping by under 1 mm."""
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


def test_contacts_past_the_capacity_are_counted_not_lost_silently(device: str) -> None:
    """With room for 4 contacts, a box on the floor (up to 20 sample points touching) is
    still held, and the overflow is counted for the HUD."""
    solver = _solver(device, max_contacts=4)
    solver.add_bodies([Body(Shape.BOX, (0.3, 0.3, 0.1), 600.0, position_m=(0.0, 0.0, 0.1))])

    _run(solver, 0.5)

    assert solver.diagnostics()["contact_overflows"] > 0
    assert solver.body_positions()[0, 2] > 0.09

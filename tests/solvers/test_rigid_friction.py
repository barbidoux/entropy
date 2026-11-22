"""Friction and restitution of the rigid contacts (`entropy.solvers.rigid`, SPEC 5.4, #88,
ADR 0032). V-RB-02 and V-RB-03 are in `test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment
from entropy.world.sdf import HalfSpace, Solids

GRAVITY = Environment().gravity_m_per_s2
G = -GRAVITY[2]
FLOOR = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),))


def _solver(device: str, solids: Solids | None = FLOOR, gravity=GRAVITY, **surface) -> RigidSolver:
    setup = RigidSetup(
        gravity_m_per_s2=gravity,
        max_bodies=16,
        solids=solids.to_device(device) if solids is not None else None,
        **surface,
    )
    return RigidSolver.create(setup, device)


def test_a_sliding_box_stops_where_kinetic_friction_says(device: str) -> None:
    """A box sliding at 2 m/s on the floor, μd = 0.3 for both: it stops after v²/(2μg) =
    0.680 m, within 5 %, and stays stopped."""
    solver = _solver(device, static_friction=0.3, dynamic_friction=0.3)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0), static_friction=0.3,
                            dynamic_friction=0.3)])  # fmt: skip

    for _ in range(90):
        solver.step(1.0 / 60.0)

    assert solver.body_positions()[0, 0] == pytest.approx(2.0**2 / (2 * 0.3 * G), rel=0.05)
    assert np.abs(solver.body_velocities()[0]).max() < 1e-3


def test_without_friction_a_box_keeps_sliding(device: str) -> None:
    """The defaults: no friction and no bounce unless a body or the surface has some."""
    solver = _solver(device)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                            velocity_m_per_s=(2.0, 0.0, 0.0))])  # fmt: skip

    for _ in range(60):
        solver.step(1.0 / 60.0)

    assert solver.body_velocities()[0, 0] == pytest.approx(2.0, rel=1e-3)


def test_the_pair_takes_the_geometric_mean_of_friction_and_the_larger_restitution(
    device: str,
) -> None:
    """ADR 0032: μ = √(μa μb) for both frictions, e = max(ea, eb). A box at μd = 0.6 on a
    floor at 0.15 slides as at μd = 0.3; a sphere at e = 0.2 dropped on a floor at 0.6
    bounces as at e = 0.6."""
    solver = _solver(device, static_friction=0.15, dynamic_friction=0.15, restitution=0.6)
    solver.add_bodies(
        [
            Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 0.1),
                 velocity_m_per_s=(2.0, 0.0, 0.0), static_friction=0.6, dynamic_friction=0.6),
            Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, position_m=(5.0, 0.0, 1.1),
                 restitution=0.2),
        ]
    )  # fmt: skip
    peak = 0.0
    landed = False
    for _ in range(150):
        solver.step(1.0 / 60.0)
        z, vz = solver.body_positions()[1, 2], solver.body_velocities()[1, 2]
        landed = landed or vz > 0.0
        if landed:
            peak = max(peak, z - 0.1)

    assert solver.body_positions()[0, 0] == pytest.approx(2.0**2 / (2 * 0.3 * G), rel=0.05)
    assert peak == pytest.approx(0.6**2 * 1.0, rel=0.05)


def test_a_box_dropped_flat_bounces_as_a_sphere_does(device: str) -> None:
    """Restitution acts once per contact patch, at its centre of pressure: a box dropped
    flat from 1 m with e = 0.5 rises to e² of its fall, within 5 %, without tipping."""
    solver = _solver(device, restitution=0.5)
    solver.add_bodies([Body(Shape.BOX, (0.2, 0.2, 0.1), 600.0, position_m=(0.0, 0.0, 1.1),
                            restitution=0.5)])  # fmt: skip
    peak, rising = 0.0, False
    for _ in range(90):
        solver.step(1.0 / 60.0)
        z, vz = solver.body_positions()[0, 2], solver.body_velocities()[0, 2]
        rising = rising or vz > 0.0
        if rising:
            peak = max(peak, z - 0.1)
            if vz < 0.0:
                break

    assert peak == pytest.approx(0.25, rel=0.05)
    assert np.abs(solver.body_angular_velocities()[0]).max() < 1e-3


def test_friction_between_bodies_keeps_momentum(device: str) -> None:
    """No gravity: a sphere spinning at 20 rad/s grazes a box, both with friction. The
    friction pushes the box sideways (0.24 m/s; exactly 0 without friction) and slows the
    spin, and its impulses are given to both alike: momentum within 1e-4."""
    solver = _solver(device, solids=None, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies(
        [
            Body(Shape.SPHERE, (0.2, 0.0, 0.0), 800.0, position_m=(-1.0, 0.1, 0.0),
                 velocity_m_per_s=(3.0, 0.0, 0.0), angular_velocity_rad_per_s=(0.0, 0.0, 20.0),
                 static_friction=0.6, dynamic_friction=0.5, restitution=0.3),
            Body(Shape.BOX, (0.3, 0.3, 0.3), 500.0, position_m=(0.5, 0.0, 0.0),
                 static_friction=0.6, dynamic_friction=0.5, restitution=0.3),
        ]
    )  # fmt: skip
    masses = solver.checkpoint()["masses_kg"][:, None].astype(np.float64)
    before = (masses * solver.body_velocities()).sum(axis=0)

    for _ in range(60):
        solver.step(1.0 / 60.0)

    after = (masses * solver.body_velocities()).sum(axis=0)
    np.testing.assert_allclose(after, before, atol=1e-4 * np.abs(before).sum())
    assert solver.body_velocities()[1, 1] > 0.1  # friction pushed the box sideways
    assert solver.body_angular_velocities()[0, 2] < 15.0  # and slowed the sphere's spin


@pytest.mark.parametrize("value", [-0.1, 1.5, math.nan])
def test_a_restitution_outside_0_to_1_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="restitution"):
        Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, restitution=value)
    with pytest.raises(ValueError, match="restitution"):
        RigidSetup(gravity_m_per_s2=GRAVITY, restitution=value)


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_a_negative_or_infinite_friction_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match="friction"):
        Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, dynamic_friction=value)
    with pytest.raises(ValueError, match="friction"):
        RigidSetup(gravity_m_per_s2=GRAVITY, static_friction=value)

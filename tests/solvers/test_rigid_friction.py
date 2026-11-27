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


MU = {"static_friction": 0.5, "dynamic_friction": 0.5}


def _box(position_m, half=(0.1, 0.1, 0.1), density=600.0, **kw) -> Body:
    return Body(Shape.BOX, half, density, position_m=position_m, **{**MU, **kw})


def _drift(solver: RigidSolver, seconds: float) -> float:
    """How far the bodies moved at most over `seconds` (m), after 1 s to settle."""
    for _ in range(60):
        solver.step(1.0 / 60.0)
    settled = solver.body_positions().copy()
    for _ in range(round(seconds * 60)):
        solver.step(1.0 / 60.0)
    return float(np.linalg.norm(solver.body_positions() - settled, axis=1).max())


@pytest.mark.parametrize("offset_m", [0.01, 0.04, 0.08])
def test_with_friction_a_box_off_centre_on_another_stays(device: str, offset_m: float) -> None:
    """A box resting on an equal one, offset by 1, 4 or 8 cm, μ = 0.5: it stays for 10 s
    (moves under 1 mm), where without friction it slides off (ADR 0031)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1)), _box((offset_m, 0.0, 0.3))])

    assert _drift(solver, 10.0) < 1e-3


def test_with_friction_a_pyramid_stands(device: str) -> None:
    """A 4-3-2-1 pyramid of boxes, each row resting on the joints of the one below, μ =
    0.5: it stands for 10 s, every box moving under 1 mm (without friction it collapses,
    ADR 0031)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    boxes = []
    for row, count in enumerate((4, 3, 2, 1)):
        for k in range(count):
            boxes.append(_box((0.2 * (k - (count - 1) / 2.0), 0.0, 0.1 + 0.2 * row)))
    solver.add_bodies(boxes)

    assert _drift(solver, 10.0) < 1e-3


def test_with_friction_a_crate_under_a_heavy_slab_stays(device: str) -> None:
    """A 20 cm crate under a concrete-heavy slab 64 times its mass, μ = 0.5: it is not
    squeezed out in 10 s (without friction it is, within 2 s, ADR 0031)."""
    solver = _solver(device, static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies(
        [_box((0.0, 0.0, 0.1)), _box((0.0, 0.0, 0.3), half=(0.4, 0.4, 0.1), density=2400.0)]
    )

    assert _drift(solver, 10.0) < 1e-3


def test_with_friction_a_box_on_a_box_holds_on_a_slope(device: str) -> None:
    """Two boxes stacked on an incline 3° below the friction angle (μ = 0.5): neither moves
    by more than 1 mm in 5 s. The upper one rests on a wide patch: friction moves it
    without turning it, and the lower one takes the torque."""
    theta = math.atan(0.5) - math.radians(3.0)
    normal = np.array([0.0, -math.sin(theta), math.cos(theta)])
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), tuple(normal)),))
    solver = _solver(device, solids=slope, static_friction=0.5, dynamic_friction=0.5)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))
    solver.add_bodies([_box(tuple(0.1 * normal), orientation=tilt),
                       _box(tuple(0.3 * normal), orientation=tilt)])  # fmt: skip

    assert _drift(solver, 5.0) < 1e-3


def test_a_checkpoint_keeps_the_friction_anchors() -> None:
    """The friction anchors are state (ADR 0032): restored from a checkpoint, a stack
    resting with friction on the floor moves on exactly as it did, bit for bit on the
    CPU."""
    solver = _solver("cpu", static_friction=0.5, dynamic_friction=0.5)
    solver.add_bodies([_box((0.0, 0.0, 0.1)), _box((0.04, 0.0, 0.3))])
    for _ in range(30):
        solver.step(1.0 / 60.0)
    saved = solver.checkpoint()
    for _ in range(30):
        solver.step(1.0 / 60.0)
    expected = solver.body_positions(), solver.body_orientations()

    solver.restore(saved)
    for _ in range(30):
        solver.step(1.0 / 60.0)

    np.testing.assert_array_equal(solver.body_positions(), expected[0])
    np.testing.assert_array_equal(solver.body_orientations(), expected[1])


def test_a_ball_rolls_down_a_slope_at_five_sevenths_g_sin_theta(device: str) -> None:
    """A sphere on a 20° slope with μ = 0.6 rolls without slipping: after 1 s its speed is
    (5/7) g sin θ t within 2 % and its spin v / r. Static friction holds the point that
    touches the slope now, not a point of the ball's surface, which rolls away from it."""
    theta = math.radians(20.0)
    normal = (0.0, -math.sin(theta), math.cos(theta))
    slope = Solids((HalfSpace((0.0, 0.0, 0.0), normal),))
    solver = _solver(device, solids=slope, static_friction=0.6, dynamic_friction=0.6)
    radius = 0.1
    solver.add_bodies([Body(Shape.SPHERE, (radius, 0.0, 0.0), 600.0,
                            position_m=tuple(radius * np.asarray(normal)),
                            static_friction=0.6, dynamic_friction=0.6)])  # fmt: skip

    for _ in range(60):
        solver.step(1.0 / 60.0)

    speed = float(np.linalg.norm(solver.body_velocities()[0]))
    assert speed == pytest.approx(5.0 / 7.0 * G * math.sin(theta) * 1.0, rel=0.02)
    spin = float(np.linalg.norm(solver.body_angular_velocities()[0]))
    assert spin == pytest.approx(speed / radius, rel=0.02)

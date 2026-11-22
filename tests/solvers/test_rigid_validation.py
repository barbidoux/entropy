"""Validation of the rigid bodies: V-RB-01 (SPEC 9.4)."""

import math

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment

GRAVITY = Environment().gravity_m_per_s2


@pytest.mark.parametrize("frame_s", [1.0 / 60.0, 1.0 / 30.0, 1.0 / 50.0, 1.0 / 45.0])
def test_v_rb_01_a_body_falls_as_z0_minus_half_g_t_squared(device: str, frame_s: float) -> None:
    """V-RB-01: a 1 s free fall, z = z₀ - ½gt², within 0.5 % of the distance fallen, at
    60, 50, 45 and 30 fps. A body moving sideways keeps its horizontal speed.

    Symplectic Euler over substeps of h = 1/480 s falls ½gt² + ½ght: h/t = 0.21 % more
    after 1 s, a named error of the integrator (SPEC 5.4). That exact fall is asserted
    too, which pins the integrator and the substep h (1/480 s at 60 and 30 fps, 1/500 s
    at 50, 1/495 s at 45)."""
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=4), device)
    solver.add_bodies(
        [
            Body(Shape.SPHERE, (0.1, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 20.0)),
            Body(
                Shape.BOX,
                (0.2, 0.2, 0.2),
                600.0,
                position_m=(0.0, 1.0, 20.0),
                velocity_m_per_s=(3.0, 0.0, 0.0),
            ),
        ]
    )
    frames = round(1.0 / frame_s)

    for _ in range(frames):
        solver.step(frame_s)

    g = -GRAVITY[2]
    fallen = 20.0 - solver.body_positions()[:, 2]
    np.testing.assert_allclose(fallen, 0.5 * g, rtol=0.005)  # V-RB-01
    h = frame_s / math.ceil(frame_s * 480.0 - 1e-9)
    np.testing.assert_allclose(fallen, 0.5 * g * (1.0 + h), rtol=1e-4)  # t = 1 s
    assert solver.body_positions()[1, 0] == pytest.approx(3.0, rel=1e-5)
    np.testing.assert_allclose(solver.body_velocities()[:, 2], -g, rtol=1e-5)


def _angular_momentum(solver: RigidSolver) -> tuple[np.ndarray, float]:
    """The bodies' total angular momentum about the origin (kg·m²/s), and the sum of its
    terms' sizes."""
    state = solver.checkpoint()
    total, size = np.zeros(3), 0.0
    for i in range(state["count"]):
        x, y, z, w = state["orientations"][i].astype(np.float64)
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        inertia = rotation @ np.diag(state["inertia_kg_m2"][i].astype(np.float64)) @ rotation.T
        mass = float(state["masses_kg"][i])
        orbit = np.cross(state["positions_m"][i], mass * state["velocities_m_per_s"][i])
        spin = inertia @ state["angular_velocities_rad_per_s"][i]
        total += orbit + spin
        size += float(np.linalg.norm(orbit) + np.linalg.norm(spin))
    return total, size


def test_v_rb_04_collisions_without_gravity_conserve_momentum(device: str) -> None:
    """V-RB-04: twelve bodies of every shape thrown at each other with no gravity and no
    solids; after 2 s of collisions the total momentum is unchanged within 1e-4 of the sum
    of |momentum|. The angular momentum is kept within 2e-3 of the sum of its terms'
    sizes (8.7e-4 in this scene, up to 1.35e-3 over seeds 0 to 9), a provisional bound
    until the M2 physics audit: the depth corrections turn the bodies about where they
    are in the substep, not where they started it (ADR 0031)."""
    rng = np.random.default_rng(4)
    shapes = [Shape.BOX, Shape.SPHERE, Shape.CAPSULE] * 4
    sizes = {
        Shape.BOX: (0.2, 0.15, 0.1),
        Shape.SPHERE: (0.15, 0.0, 0.0),
        Shape.CAPSULE: (0.1, 0.2, 0.0),
    }
    bodies = []
    for k, shape in enumerate(shapes):
        position = (1.6 * (k % 4) - 2.4, 1.6 * (k // 4) - 1.6, 0.0)
        velocity = tuple(-0.8 * np.asarray(position) + rng.normal(0.0, 0.3, 3))
        spin = tuple(rng.normal(0.0, 1.0, 3))
        bodies.append(Body(shape, sizes[shape], 400.0 + 100.0 * k, position_m=position,
                           velocity_m_per_s=velocity, angular_velocity_rad_per_s=spin))  # fmt: skip
    solver = RigidSolver.create(RigidSetup(gravity_m_per_s2=(0.0, 0.0, 0.0), max_bodies=16), device)
    solver.add_bodies(bodies)
    masses = solver.checkpoint()["masses_kg"][:, None].astype(np.float64)
    before = solver.body_velocities().astype(np.float64)
    turning_before, turning_size = _angular_momentum(solver)

    contacts = 0
    for _ in range(120):
        solver.step(1.0 / 60.0)
        contacts += solver.diagnostics()["contacts"]

    after = solver.body_velocities().astype(np.float64)
    scale = (masses * np.linalg.norm(before, axis=1, keepdims=True)).sum()
    np.testing.assert_allclose(
        (masses * after).sum(axis=0), (masses * before).sum(axis=0), atol=1e-4 * scale
    )
    turning_after, _ = _angular_momentum(solver)
    assert np.linalg.norm(turning_after - turning_before) <= 2e-3 * turning_size
    assert contacts > 0
    assert np.abs(after - before).max() > 0.5  # they did collide


@pytest.mark.parametrize("restitution", [0.5, 0.8])
def test_v_rb_02_a_bounce_rises_e_squared_of_the_drop(device: str, restitution: float) -> None:
    """V-RB-02: a sphere dropped from 1 m onto the floor, both of restitution e, rises to
    e² of the height it fell, within 3 %."""
    from entropy.world.sdf import HalfSpace, Solids

    floor = Solids((HalfSpace((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),)).to_device(device)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=2, solids=floor, restitution=restitution),
        device,
    )
    radius = 0.1
    drop = Body(Shape.SPHERE, (radius, 0.0, 0.0), 600.0, position_m=(0.0, 0.0, 1.0 + radius),
                restitution=restitution)  # fmt: skip
    solver.add_bodies([drop])

    peak, rising = 0.0, False
    for _ in range(round(1.6 * 60)):
        solver.step(1.0 / 60.0)
        z, vz = solver.body_positions()[0, 2], solver.body_velocities()[0, 2]
        rising = rising or vz > 0.0
        if rising:
            peak = max(peak, z - radius)
            if vz < 0.0:
                break

    assert peak == pytest.approx(restitution**2 * 1.0, rel=0.03)


@pytest.mark.parametrize("margin_deg", [-1.0, 1.0])
def test_v_rb_03_a_block_holds_below_the_friction_angle_and_slides_above(
    device: str, margin_deg: float
) -> None:
    """V-RB-03: a block on an incline with μ = 0.5 (atan μ = 26.57°): 1° below it holds
    (moves under 1 mm in 1 s), 1° above it slides (over 5 cm)."""
    from entropy.world.sdf import HalfSpace, Solids

    mu = 0.5
    theta = math.atan(mu) + math.radians(margin_deg)
    normal = (0.0, -math.sin(theta), math.cos(theta))
    incline = Solids((HalfSpace((0.0, 0.0, 0.0), normal),)).to_device(device)
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=GRAVITY, max_bodies=2, solids=incline,
                   static_friction=mu, dynamic_friction=mu),
        device,
    )  # fmt: skip
    half = (0.2, 0.2, 0.05)
    tilt = (math.sin(theta / 2.0), 0.0, 0.0, math.cos(theta / 2.0))  # lies on the slope
    start = tuple(0.05 * np.asarray(normal))
    solver.add_bodies([Body(Shape.BOX, half, 600.0, position_m=start, orientation=tilt,
                            static_friction=mu, dynamic_friction=mu)])  # fmt: skip
    for _ in range(6):
        solver.step(1.0 / 60.0)  # settles onto the slope
    settled = solver.body_positions()[0].copy()

    for _ in range(60):
        solver.step(1.0 / 60.0)

    moved = float(np.linalg.norm(solver.body_positions()[0] - settled))
    if margin_deg < 0:
        assert moved < 1e-3
    else:
        assert moved > 0.05

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


def test_v_rb_04_collisions_without_gravity_conserve_momentum(device: str) -> None:
    """V-RB-04: twelve bodies of every shape thrown at each other with no gravity and no
    solids; after 2 s of collisions the total momentum is unchanged within 1e-4 of the sum
    of |momentum|."""
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

    contacts = 0
    for _ in range(120):
        solver.step(1.0 / 60.0)
        contacts += solver.diagnostics()["contacts"]

    after = solver.body_velocities().astype(np.float64)
    scale = (masses * np.linalg.norm(before, axis=1, keepdims=True)).sum()
    np.testing.assert_allclose(
        (masses * after).sum(axis=0), (masses * before).sum(axis=0), atol=1e-4 * scale
    )
    assert contacts > 0
    assert np.abs(after - before).max() > 0.5  # they did collide

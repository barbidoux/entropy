"""Validation of the rigid bodies: V-RB-01 (SPEC 9.4)."""

import numpy as np
import pytest

from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import Environment

GRAVITY = Environment().gravity_m_per_s2


@pytest.mark.parametrize("frame_s", [1.0 / 60.0, 1.0 / 30.0])
def test_v_rb_01_a_body_falls_as_z0_minus_half_g_t_squared(device: str, frame_s: float) -> None:
    """V-RB-01: a 1 s free fall, z = z₀ - ½gt², within 0.5 % of the distance fallen, at
    60 and 30 fps. A body moving sideways keeps its horizontal speed.

    Symplectic Euler over substeps of h = 1/480 s falls ½gt² + ½ght: h/t = 0.21 % more
    after 1 s, a named error of the integrator (SPEC 5.4)."""
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
    np.testing.assert_allclose(fallen, 0.5 * g, rtol=0.005)
    assert solver.body_positions()[1, 0] == pytest.approx(3.0, rel=1e-5)
    np.testing.assert_allclose(solver.body_velocities()[:, 2], -g, rtol=1e-5)

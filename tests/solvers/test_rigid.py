"""Rigid bodies (`entropy.solvers.rigid`, SPEC 5.4, #86): bodies, XPBD substeps, gravity,
the NaN guard and checkpoints. V-RB-01 is in `test_rigid_validation.py`."""

import math

import numpy as np
import pytest

from entropy.sim.simulation import PacedSolver, Simulation
from entropy.solvers.interface import NonFiniteStateError, Solver
from entropy.solvers.rigid import (
    MAX_SUBSTEP_S,
    Body,
    RigidSetup,
    RigidSolver,
    Shape,
    load_dump,
    mass_properties,
)
from entropy.world.maps import Environment

GRAVITY = Environment().gravity_m_per_s2
WOOD_KG_PER_M3 = 600.0  # a test density, not a material constant


def _solver(device: str, tmp_path=None, gravity=GRAVITY) -> RigidSolver:
    return RigidSolver.create(
        RigidSetup(gravity_m_per_s2=gravity, max_bodies=64, dump_dir=tmp_path), device
    )


def _box(**kw) -> Body:
    base = {
        "shape": Shape.BOX,
        "size_m": (0.3, 0.2, 0.1),
        "density_kg_per_m3": WOOD_KG_PER_M3,
        "position_m": (0.0, 0.0, 5.0),
    }
    base.update(kw)
    return Body(**base)


def test_the_rigid_solver_is_a_solver_and_paced() -> None:
    solver = _solver("cpu")
    assert isinstance(solver, Solver)
    assert isinstance(solver, PacedSolver)


def test_rigid_substeps_never_exceed_1_480_s(device: str) -> None:
    """SPEC 5.4: ⌈advance / (1/480 s)⌉ equal substeps."""
    solver = _solver(device)
    solver.add_bodies([_box()])
    for dt_s, expected in ((1.0 / 30.0, 16), (1.0 / 60.0, 8), (0.001, 1), (1.0 / 480.0, 1)):
        solver.step(dt_s)
        assert solver.diagnostics()["substeps"] == expected
        assert dt_s / expected <= MAX_SUBSTEP_S * (1.0 + 1e-9)


def test_the_rigid_solver_never_limits_the_world_s_advance() -> None:
    """Its substeps follow the advance (ADR 0013): no cap to hold the world to."""
    solver = _solver("cpu")
    solver.add_bodies([_box(velocity_m_per_s=(50.0, 0.0, 0.0))])
    solver.step(1.0 / 60.0)
    assert solver.max_advance_s() == math.inf


@pytest.mark.parametrize(
    ("shape", "size_m"),
    [
        (Shape.BOX, (0.3, 0.2, 0.1)),
        (Shape.SPHERE, (0.25, 0.0, 0.0)),
        (Shape.CAPSULE, (0.1, 0.3, 0.0)),
    ],
)
def test_mass_and_inertia_match_a_numerical_integral(shape: Shape, size_m) -> None:
    """Mass and principal moments (body frame) against a Monte Carlo integral of the
    shape: box half sizes (x, y, z); sphere radius; capsule radius and the half length of
    its cylinder along z."""
    mass_kg, inertia = mass_properties(shape, size_m, 1000.0)
    rng = np.random.default_rng(3)
    if shape == Shape.BOX:
        half = np.asarray(size_m)
    elif shape == Shape.SPHERE:
        half = np.full(3, size_m[0])
    else:
        half = np.array([size_m[0], size_m[0], size_m[0] + size_m[1]])
    points = rng.uniform(-half, half, (2_000_000, 3))
    if shape == Shape.SPHERE:
        inside = (points**2).sum(axis=1) <= size_m[0] ** 2
    elif shape == Shape.CAPSULE:
        z = np.clip(points[:, 2], -size_m[1], size_m[1])
        inside = points[:, 0] ** 2 + points[:, 1] ** 2 + (points[:, 2] - z) ** 2 <= size_m[0] ** 2
    else:
        inside = np.ones(len(points), dtype=bool)
    volume = np.prod(2.0 * half) * inside.mean()
    p = points[inside]
    expected_inertia = (
        1000.0
        * volume
        * np.array(
            [
                (p[:, 1] ** 2 + p[:, 2] ** 2).mean(),
                (p[:, 0] ** 2 + p[:, 2] ** 2).mean(),
                (p[:, 0] ** 2 + p[:, 1] ** 2).mean(),
            ]
        )
    )
    np.testing.assert_allclose(mass_kg, 1000.0 * volume, rtol=3e-3)
    np.testing.assert_allclose(inertia, expected_inertia, rtol=5e-3)


def test_bodies_are_refused_when_their_shape_or_density_cannot_be(device: str) -> None:
    solver = _solver(device)
    with pytest.raises(ValueError, match="size"):
        solver.add_bodies([_box(size_m=(0.3, -0.2, 0.1))])
    with pytest.raises(ValueError, match="density"):
        solver.add_bodies([_box(density_kg_per_m3=0.0)])
    with pytest.raises(ValueError, match="orientation"):
        solver.add_bodies([_box(orientation=(0.0, 0.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="capacity"):
        solver.add_bodies([_box()] * 65)
    assert solver.body_count() == 0


def test_an_orientation_is_stored_normalised(device: str) -> None:
    solver = _solver(device)
    solver.add_bodies([_box(orientation=(0.0, 0.0, 2.0, 2.0))])
    np.testing.assert_allclose(
        solver.body_orientations()[0], [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)], atol=1e-6
    )


def test_a_body_spinning_about_a_principal_axis_turns_steadily(device: str) -> None:
    """No gravity: 2 rad/s about z for 1 s turns the box by 2 rad about z, and its spin
    stays."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box(angular_velocity_rad_per_s=(0.0, 0.0, 2.0))])
    for _ in range(60):
        solver.step(1.0 / 60.0)
    x, y, z, w = solver.body_orientations()[0]
    angle = 2.0 * math.atan2(z, w)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert angle == pytest.approx(2.0, abs=2e-3)
    np.testing.assert_allclose(solver.body_angular_velocities()[0], [0.0, 0.0, 2.0], atol=1e-4)


def test_free_rotation_keeps_the_angular_momentum(device: str) -> None:
    """No gravity, no torque: a box spinning about no principal axis tumbles, and its
    angular momentum in world space, R I Rᵀ ω, stays within 1 % over 2 s (the gyroscopic
    term is integrated explicitly per substep, SPEC 5.4)."""
    solver = _solver(device, gravity=(0.0, 0.0, 0.0))
    solver.add_bodies([_box(angular_velocity_rad_per_s=(1.0, 3.0, 0.5))])
    _, inertia = mass_properties(Shape.BOX, (0.3, 0.2, 0.1), WOOD_KG_PER_M3)

    def momentum() -> np.ndarray:
        x, y, z, w = solver.body_orientations()[0]
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        omega = solver.body_angular_velocities()[0]
        return rotation @ np.diag(inertia) @ rotation.T @ omega

    before = momentum()
    for _ in range(120):
        solver.step(1.0 / 60.0)
    after = momentum()
    assert np.linalg.norm(after - before) <= 0.01 * np.linalg.norm(before)
    assert np.abs(solver.body_angular_velocities()[0] - (1.0, 3.0, 0.5)).max() > 0.1  # it tumbles


def test_a_checkpoint_restores_every_body_exactly(device: str) -> None:
    solver = _solver(device)
    solver.add_bodies(
        [_box(), _box(position_m=(1.0, 0.0, 3.0), angular_velocity_rad_per_s=(0, 1, 0))]
    )
    solver.step(1.0 / 60.0)
    saved = solver.checkpoint()
    state = (solver.body_positions().copy(), solver.body_orientations().copy())
    for _ in range(10):
        solver.step(1.0 / 60.0)

    solver.restore(saved)

    np.testing.assert_array_equal(solver.body_positions(), state[0])
    np.testing.assert_array_equal(solver.body_orientations(), state[1])
    assert solver.body_count() == 2


def test_a_non_finite_body_freezes_the_solver_and_dumps_its_frame(device: str, tmp_path) -> None:
    """SPEC 5.9: the guard reads the step's status at the next call, raises with a dump of
    the state before and after the frame and its time step, and keeps raising until a
    restore."""
    solver = _solver(device, tmp_path)
    solver.add_bodies([_box(), _box(position_m=(1.0, 0.0, 5.0))])
    saved = solver.checkpoint()
    solver.set_body_velocity(1, (math.nan, 0.0, 0.0))
    solver.step(1.0 / 60.0)

    with pytest.raises(NonFiniteStateError) as raised:
        solver.step(1.0 / 60.0)
    with pytest.raises(NonFiniteStateError):
        solver.max_advance_s()

    dump = load_dump(raised.value.dump_path)
    assert float(dump["dt_s"]) == pytest.approx(1.0 / 60.0)
    assert np.isnan(dump["before_velocities_m_per_s"][1]).any()
    assert not np.isfinite(dump["positions_m"][1]).all()
    assert np.isfinite(dump["positions_m"][0]).all()
    solver.restore(saved)
    solver.step(1.0 / 60.0)  # runs again


def test_the_simulation_paces_the_rigid_solver_with_the_liquid_s_advance() -> None:
    """One shared advance per frame (ADR 0013): the rigid solver steps by it."""
    solver = _solver("cpu")
    solver.add_bodies([_box()])
    simulation = Simulation([solver])

    advance = simulation.step(1.0 / 30.0)

    assert advance == pytest.approx(1.0 / 30.0)
    assert solver.diagnostics()["substeps"] == 16

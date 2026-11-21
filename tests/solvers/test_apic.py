"""APIC transfers between liquid particles and the MAC grid (SPEC 5.2; Jiang et al. 2015)."""

import numpy as np
import pytest
import warp as wp
from hypothesis import given, settings
from hypothesis import strategies as st

from entropy.fields.mac import GridSpec, MacVelocity
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.transfer import grid_to_particles, particles_to_grid

SPEC = GridSpec(shape=(10, 9, 8), dx_m=0.25, origin_m=(0.5, -1.0, 2.0))
MASS_KG = 0.3


def _face_positions_m(spec: GridSpec, axis: int) -> np.ndarray:
    shape = spec.face_shape(axis)
    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    offset = np.full(3, 0.5)
    offset[axis] = 0.0
    return np.asarray(spec.origin_m) + (index + offset) * spec.dx_m


def _interior_points_m(count: int, seed: int) -> np.ndarray:
    """Points whose full quadratic stencil lies inside the grid: one cell in from every side."""
    rng = np.random.default_rng(seed)
    low = np.asarray(SPEC.origin_m) + 1.0 * SPEC.dx_m
    high = np.asarray(SPEC.origin_m) + (np.asarray(SPEC.shape) - 1.0) * SPEC.dx_m
    return rng.uniform(low, high, size=(count, 3)).astype(np.float32)


def _affine_grid(device: str, a: np.ndarray, b: np.ndarray) -> MacVelocity:
    components = [(a[c] + _face_positions_m(SPEC, c) @ b[c]).astype(np.float32) for c in range(3)]
    return MacVelocity(SPEC, *(wp.array(c, dtype=float, device=device) for c in components))


def _p2g(particles: LiquidParticles, device: str) -> tuple[MacVelocity, MacVelocity]:
    velocity, mass = MacVelocity.zeros(SPEC, device), MacVelocity.zeros(SPEC, device)
    particles_to_grid(particles, MASS_KG, velocity, mass)
    return velocity, mass


def test_particles_hold_position_velocity_and_affine_matrix(device: str) -> None:
    points = _interior_points_m(5, 1)

    particles = LiquidParticles.from_numpy(points, device=device)

    assert particles.count == 5
    np.testing.assert_array_equal(particles.position_m.numpy(), points)
    assert np.all(particles.velocity_m_per_s.numpy() == 0.0)
    assert particles.affine_per_s.numpy().shape == (5, 3, 3)
    assert str(particles.position_m.device) == device


def test_grid_to_particles_reads_a_uniform_field_exactly(device: str) -> None:
    particles = LiquidParticles.from_numpy(_interior_points_m(50, 2), device=device)

    grid_to_particles(_affine_grid(device, np.array([1.0, -2.0, 0.5]), np.zeros((3, 3))), particles)

    np.testing.assert_allclose(
        particles.velocity_m_per_s.numpy(), [[1.0, -2.0, 0.5]] * 50, atol=1e-5
    )
    np.testing.assert_allclose(particles.affine_per_s.numpy(), 0.0, atol=1e-4)


def test_grid_to_particles_reads_an_affine_field_and_its_gradient(device: str) -> None:
    """Quadratic B-splines reproduce linear fields, and APIC's C recovers their gradient."""
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(3), rng.standard_normal((3, 3))
    points = _interior_points_m(100, 4)
    particles = LiquidParticles.from_numpy(points, device=device)

    grid_to_particles(_affine_grid(device, a, b), particles)

    np.testing.assert_allclose(particles.velocity_m_per_s.numpy(), a + points @ b.T, atol=1e-4)
    np.testing.assert_allclose(
        particles.affine_per_s.numpy(), np.broadcast_to(b, (100, 3, 3)), atol=1e-3
    )


def test_particles_to_grid_keeps_an_affine_field_exactly(device: str) -> None:
    """Particles carrying v = a + B x and C = B give the grid that field wherever it has mass."""
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(3), rng.standard_normal((3, 3))
    points = _interior_points_m(4000, 6)
    particles = LiquidParticles.from_numpy(
        points, velocities=a + points @ b.T, affine=np.broadcast_to(b, (4000, 3, 3)), device=device
    )

    velocity, mass = _p2g(particles, device)

    for c, (component, weight) in enumerate(
        zip((velocity.u, velocity.v, velocity.w), (mass.u, mass.v, mass.w), strict=True)
    ):
        covered = weight.numpy() > 1e-3 * MASS_KG
        expected = a[c] + _face_positions_m(SPEC, c) @ b[c]
        assert covered.sum() > 100
        np.testing.assert_allclose(
            component.numpy()[covered], expected[covered], rtol=1e-3, atol=1e-3
        )


def test_particles_to_grid_conserves_mass_and_momentum(device: str) -> None:
    rng = np.random.default_rng(7)
    points = _interior_points_m(500, 8)
    velocities = rng.standard_normal((500, 3)).astype(np.float32)
    affine = rng.standard_normal((500, 3, 3)).astype(np.float32)
    particles = LiquidParticles.from_numpy(
        points, velocities=velocities, affine=affine, device=device
    )

    velocity, mass = _p2g(particles, device)

    for c, (component, weight) in enumerate(
        zip((velocity.u, velocity.v, velocity.w), (mass.u, mass.v, mass.w), strict=True)
    ):
        assert weight.numpy().sum() == pytest.approx(500 * MASS_KG, rel=1e-4)
        momentum = (weight.numpy() * component.numpy()).sum()
        assert momentum == pytest.approx(MASS_KG * velocities[:, c].sum(), rel=1e-3, abs=1e-3)


def _grid_angular_momentum(velocity: MacVelocity, mass: MacVelocity) -> np.ndarray:
    total = np.zeros(3)
    for c, (component, weight) in enumerate(
        zip((velocity.u, velocity.v, velocity.w), (mass.u, mass.v, mass.w), strict=True)
    ):
        e = np.zeros(3)
        e[c] = 1.0
        momentum = (weight.numpy() * component.numpy())[..., None] * e
        total += np.cross(_face_positions_m(SPEC, c), momentum).reshape(-1, 3).sum(axis=0)
    return total


def _particle_angular_momentum(points: np.ndarray, velocities: np.ndarray, affine: np.ndarray):
    """APIC angular momentum: x x m v plus the affine part m (Δx²/4) Σ_c C[c] x e_c."""
    spin = sum(np.cross(affine[:, c, :], np.eye(3)[c]) for c in range(3))
    return MASS_KG * (np.cross(points, velocities) + SPEC.dx_m**2 / 4.0 * spin).sum(axis=0)


@settings(max_examples=20, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16))
def test_particles_to_grid_conserves_apic_angular_momentum(seed: int) -> None:
    """Jiang et al. (2015): APIC's transfer conserves angular momentum, affine part included."""
    rng = np.random.default_rng(seed)
    points = _interior_points_m(200, seed + 1).astype(np.float64)
    velocities = rng.standard_normal((200, 3))
    affine = rng.standard_normal((200, 3, 3))
    particles = LiquidParticles.from_numpy(
        points, velocities=velocities, affine=affine, device="cpu"
    )

    velocity, mass = _p2g(particles, "cpu")

    expected = _particle_angular_momentum(points, velocities, affine)
    np.testing.assert_allclose(
        _grid_angular_momentum(velocity, mass), expected, rtol=1e-3, atol=1e-3
    )


def test_faces_without_mass_get_zero_velocity(device: str) -> None:
    centre = np.asarray(SPEC.origin_m) + np.asarray(SPEC.shape) * SPEC.dx_m / 2
    particles = LiquidParticles.from_numpy(
        centre[None].astype(np.float32), velocities=np.ones((1, 3)), device=device
    )

    velocity, mass = _p2g(particles, device)

    for component, weight in zip(
        (velocity.u, velocity.v, velocity.w), (mass.u, mass.v, mass.w), strict=True
    ):
        empty = weight.numpy() == 0.0
        assert empty.any() and np.all(component.numpy()[empty] == 0.0)


def test_mismatched_inputs_are_refused() -> None:
    particles = LiquidParticles.from_numpy(_interior_points_m(3, 9), device="cpu")
    other = MacVelocity.zeros(GridSpec(shape=(4, 4, 4), dx_m=0.25), "cpu")
    with pytest.raises(ValueError, match="grid"):
        particles_to_grid(particles, MASS_KG, MacVelocity.zeros(SPEC, "cpu"), other)
    with pytest.raises(ValueError, match="mass"):
        particles_to_grid(
            particles, 0.0, MacVelocity.zeros(SPEC, "cpu"), MacVelocity.zeros(SPEC, "cpu")
        )
    with pytest.raises(ValueError, match="velocities"):
        LiquidParticles.from_numpy(
            _interior_points_m(3, 9), velocities=np.zeros((2, 3)), device="cpu"
        )


@pytest.mark.gpu
def test_cpu_and_cuda_transfer_alike() -> None:
    """SPEC 9.1 rule 4 (atomic order differs on the GPU: tolerance, not bits)."""
    rng = np.random.default_rng(10)
    points = _interior_points_m(300, 11)
    velocities = rng.standard_normal((300, 3))
    results = []
    for device in ("cpu", "cuda:0"):
        particles = LiquidParticles.from_numpy(points, velocities=velocities, device=device)
        velocity, _ = _p2g(particles, device)
        grid_to_particles(velocity, particles)
        results.append(particles.velocity_m_per_s.numpy())

    np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-5)

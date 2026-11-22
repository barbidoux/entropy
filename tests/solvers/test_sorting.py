"""Liquid particles sorted by half-cell (`entropy.solvers.liquid.sorting`, #65, ADR 0025)."""

import numpy as np
import pytest
import warp as wp

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.sorting import CellSort

SPEC = GridSpec(shape=(6, 5, 4), dx_m=0.25, origin_m=(-1.0, 0.5, 0.0))


def _keys(positions_m: np.ndarray) -> np.ndarray:
    """x-major index of the half-cell (cells split in two along each axis), clamped to
    the grid; 0 for a position that is not finite."""
    shape = 2 * np.asarray(SPEC.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        g = (positions_m - np.asarray(SPEC.origin_m)) / (SPEC.dx_m / 2)
        g = np.clip(np.nan_to_num(g, nan=0.0, posinf=1e9, neginf=-1e9), -1.0, shape)
    g = np.clip(np.floor(g).astype(np.int64), 0, shape - 1)
    keys = (g[:, 0] * shape[1] + g[:, 1]) * shape[2] + g[:, 2]
    return np.where(np.isfinite(positions_m).all(axis=1), keys, 0)


def _particles(positions_m: np.ndarray, device: str) -> LiquidParticles:
    """Velocity 2 x and affine diag(x): each particle's other arrays tell where it was."""
    affine = np.zeros((len(positions_m), 3, 3))
    for c in range(3):
        affine[:, c, c] = positions_m[:, c]
    return LiquidParticles.from_numpy(
        positions_m, velocities=2.0 * positions_m, affine=affine, device=device
    )


def _empty(n: int, device: str) -> LiquidParticles:
    return LiquidParticles.from_numpy(np.zeros((n, 3)), device=device)


def test_particles_come_out_in_half_cell_order_with_their_own_velocity_and_affine(
    device: str,
) -> None:
    rng = np.random.default_rng(1)
    low, high = np.asarray(SPEC.origin_m), np.asarray(SPEC.origin_m) + 0.25 * np.asarray(SPEC.shape)
    positions = rng.uniform(low, high, (500, 3))
    sort = CellSort(SPEC, capacity=600, device=device)
    out = _empty(500, device)

    sort.sort(_particles(positions, device), out)

    x = out.position_m.numpy()
    assert np.all(np.diff(_keys(x)) >= 0)
    np.testing.assert_allclose(np.sort(x, axis=0), np.sort(positions, axis=0).astype(np.float32))
    np.testing.assert_allclose(out.velocity_m_per_s.numpy(), 2.0 * x, rtol=1e-6)
    affine = out.affine_per_s.numpy()
    np.testing.assert_allclose(np.diagonal(affine, axis1=1, axis2=2), x, rtol=1e-6)


def test_particles_that_share_a_half_cell_keep_their_order(device: str) -> None:
    """A stable sort: still water's order does not churn from frame to frame."""
    centre = np.asarray(SPEC.origin_m) + 0.25 * np.array([2.25, 1.25, 0.25])  # a half-cell
    same = centre + np.linspace(-0.03, 0.03, 9)[:, None] * np.array([1.0, 0.0, 0.0])
    other = centre - np.array([0.5, 0.0, 0.0])
    positions = np.vstack([same[:4], other, same[4:]])
    sort = CellSort(SPEC, capacity=10, device=device)
    out = _empty(10, device)

    sort.sort(_particles(positions, device), out)

    expected = np.vstack([other, same]).astype(np.float32)
    np.testing.assert_array_equal(out.position_m.numpy(), expected)


def test_particles_outside_the_grid_sort_with_the_nearest_half_cell(device: str) -> None:
    """Out of range on one axis, on mixed axes (x low, y high), or by 1e30 m: each axis is
    clamped to the grid on its own, the same on CPU and CUDA."""
    origin = np.asarray(SPEC.origin_m)
    positions = origin + np.array(
        [
            [1.4, 1.1, 0.9],
            [-3.0, -1.0, -0.5],
            [0.06, 0.06, 0.06],
            [-0.5, 9.0, 0.5],
            [9.0, -0.5, 0.5],
            [1e30, 0.3, 0.3],
            [-1e30, 0.3, -1e30],
            [0.3, 1e30, 0.3],
        ]
    )
    sort = CellSort(SPEC, capacity=len(positions), device=device)
    out = _empty(len(positions), device)

    sort.sort(_particles(positions, device), out)

    expected = positions[np.argsort(_keys(positions), kind="stable")].astype(np.float32)
    np.testing.assert_array_equal(out.position_m.numpy(), expected)


def test_a_non_finite_particle_is_kept_for_the_nan_guard(device: str) -> None:
    positions = np.asarray(SPEC.origin_m) + 0.25 * np.array([[3.5, 2.5, 1.5], [0.5, 0.5, 0.5]])
    particles = _particles(positions, device)
    velocities = particles.velocity_m_per_s.numpy()
    velocities[1] = np.nan
    particles.velocity_m_per_s.assign(velocities)
    broken = particles.position_m.numpy()
    broken[0] = np.inf
    particles.position_m.assign(broken)
    sort = CellSort(SPEC, capacity=2, device=device)
    out = _empty(2, device)

    sort.sort(particles, out)

    assert not np.isfinite(out.position_m.numpy()).all()
    assert not np.isfinite(out.velocity_m_per_s.numpy()).all()


def test_the_sort_refuses_more_particles_than_it_was_made_for() -> None:
    sort = CellSort(SPEC, capacity=2, device="cpu")
    with pytest.raises(ValueError, match="capacity"):
        sort.sort(_particles(np.zeros((3, 3)), "cpu"), _empty(3, "cpu"))


def test_the_sort_refuses_to_gather_into_its_own_input() -> None:
    """A gather in place would read what it already overwrote (#75's review)."""
    sort = CellSort(SPEC, capacity=4, device="cpu")
    particles = _particles(np.zeros((3, 3)), "cpu")
    with pytest.raises(ValueError, match="other arrays"):
        sort.sort(particles, particles)


def test_no_particles_is_nothing_to_sort(device: str) -> None:
    sort = CellSort(SPEC, capacity=2, device=device)
    empty = LiquidParticles(
        wp.zeros(0, dtype=wp.vec3, device=device),
        wp.zeros(0, dtype=wp.vec3, device=device),
        wp.zeros(0, dtype=wp.mat33, device=device),
    )
    sort.sort(empty, empty)  # does not raise

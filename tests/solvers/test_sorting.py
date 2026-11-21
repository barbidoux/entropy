"""Liquid particles sorted by cell (`entropy.solvers.liquid.sorting`, #65)."""

import numpy as np
import pytest
import warp as wp

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.sorting import CellSort

SPEC = GridSpec(shape=(6, 5, 4), dx_m=0.25, origin_m=(-1.0, 0.5, 0.0))


def _keys(positions_m: np.ndarray) -> np.ndarray:
    """x-major cell index, cells clamped to the grid."""
    g = np.floor((positions_m - np.asarray(SPEC.origin_m)) / SPEC.dx_m).astype(np.int64)
    g = np.clip(g, 0, np.asarray(SPEC.shape) - 1)
    return (g[:, 0] * SPEC.shape[1] + g[:, 1]) * SPEC.shape[2] + g[:, 2]


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


def test_particles_come_out_in_cell_order_with_their_own_velocity_and_affine(
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


def test_particles_that_share_a_cell_keep_their_order(device: str) -> None:
    """A stable sort: still water's order does not churn from frame to frame."""
    centre = np.asarray(SPEC.origin_m) + 0.25 * np.array([2.5, 1.5, 0.5])
    same_cell = centre + np.linspace(-0.1, 0.1, 9)[:, None] * np.array([1.0, 0.0, 0.0])
    other = centre - np.array([0.5, 0.0, 0.0])
    positions = np.vstack([same_cell[:4], other, same_cell[4:]])
    sort = CellSort(SPEC, capacity=10, device=device)
    out = _empty(10, device)

    sort.sort(_particles(positions, device), out)

    expected = np.vstack([other, same_cell]).astype(np.float32)
    np.testing.assert_array_equal(out.position_m.numpy(), expected)


def test_particles_outside_the_grid_sort_with_the_nearest_cell(device: str) -> None:
    inside = np.asarray(SPEC.origin_m) + 0.25 * np.array([[0.5, 0.5, 0.5], [5.5, 4.5, 3.5]])
    outside = np.asarray(SPEC.origin_m) + np.array([[-3.0, -1.0, -0.5], [9.0, 9.0, 9.0]])
    positions = np.vstack([inside[1], outside[0], inside[0], outside[1]])
    sort = CellSort(SPEC, capacity=4, device=device)
    out = _empty(4, device)

    sort.sort(_particles(positions, device), out)

    expected = np.vstack([outside[0], inside[0], inside[1], outside[1]]).astype(np.float32)
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


def test_no_particles_is_nothing_to_sort(device: str) -> None:
    sort = CellSort(SPEC, capacity=2, device=device)
    empty = LiquidParticles(
        wp.zeros(0, dtype=wp.vec3, device=device),
        wp.zeros(0, dtype=wp.vec3, device=device),
        wp.zeros(0, dtype=wp.mat33, device=device),
    )
    sort.sort(empty, empty)  # does not raise

"""Weighted Jacobi for the 7-point Poisson stencil (`entropy.linalg.jacobi`)."""

import numpy as np
import warp as wp

from entropy.linalg.jacobi import jacobi_solve, jacobi_step


def _numpy_jacobi_step(p: np.ndarray, rhs: np.ndarray, h_m: float, omega: float) -> np.ndarray:
    padded = np.pad(p, 1)  # zero outside the grid (homogeneous Dirichlet)
    neighbours = (
        padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
    )
    return (1.0 - omega) * p + omega * (neighbours - h_m * h_m * rhs) / 6.0


def test_jacobi_step_matches_numpy_reference(device: str) -> None:
    rng = np.random.default_rng(seed=1)
    p = rng.standard_normal((6, 5, 4)).astype(np.float32)
    rhs = rng.standard_normal((6, 5, 4)).astype(np.float32)
    out = wp.zeros(p.shape, dtype=float, device=device)

    jacobi_step(wp.array(p, device=device), wp.array(rhs, device=device), out, h_m=0.1, omega=0.8)

    expected = _numpy_jacobi_step(p, rhs, h_m=0.1, omega=0.8)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_jacobi_solve_converges_to_a_manufactured_solution(device: str) -> None:
    """p = sin(πx) sin(πy) sin(πz) on the unit cube, p = 0 on its boundary.

    Unknowns sit at x = (i + 1) h with h = 1/(n + 1), so the zero values outside
    the grid fall exactly on the boundary. The discrete solution differs from
    the exact one by O(h²), about 0.3 % here.
    """
    n = 16
    h_m = 1.0 / (n + 1)
    x = (np.arange(n) + 1) * h_m
    exact = np.einsum("i,j,k->ijk", *(np.sin(np.pi * x),) * 3).astype(np.float32)
    rhs = (-3.0 * np.pi**2 * exact).astype(np.float32)

    p = jacobi_solve(
        wp.zeros((n, n, n), dtype=float, device=device),
        wp.array(rhs, device=device),
        h_m=h_m,
        iterations=1500,
    )

    error = np.abs(p.numpy() - exact).max() / np.abs(exact).max()
    assert error < 0.01

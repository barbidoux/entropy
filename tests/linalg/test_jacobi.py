"""Weighted Jacobi for the 7-point Poisson stencil (`entropy.linalg.jacobi`)."""

import numpy as np
import pytest
import warp as wp

from entropy.linalg.jacobi import jacobi_solve, jacobi_step

SHAPE = (19, 5, 4)  # 19 is not a multiple of the kernel's marching chunk


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


def _random(shape: tuple[int, ...], seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_jacobi_step_matches_numpy_reference(device: str) -> None:
    p, rhs = _random(SHAPE, 1), _random(SHAPE, 2)
    out = wp.zeros(SHAPE, dtype=float, device=device)

    jacobi_step(wp.array(p, device=device), wp.array(rhs, device=device), out, h_m=0.1, omega=0.8)

    expected = _numpy_jacobi_step(p, rhs, h_m=0.1, omega=0.8)
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("iterations", [1, 2, 3])
def test_jacobi_solve_equals_repeated_reference_steps(device: str, iterations: int) -> None:
    p, rhs = _random(SHAPE, 3), _random(SHAPE, 4)

    result = jacobi_solve(
        wp.array(p, device=device), wp.array(rhs, device=device), h_m=0.1, iterations=iterations
    )

    expected = p
    for _ in range(iterations):
        expected = _numpy_jacobi_step(expected, rhs, h_m=0.1, omega=1.0)
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_jacobi_solve_uses_a_given_scratch_array(device: str) -> None:
    p, rhs = _random(SHAPE, 5), _random(SHAPE, 6)
    p_wp = wp.array(p, device=device)
    scratch = wp.zeros(SHAPE, dtype=float, device=device)

    result = jacobi_solve(
        p_wp, wp.array(rhs, device=device), h_m=0.1, iterations=1, scratch=scratch
    )

    assert result.ptr == scratch.ptr


def test_jacobi_solve_converges_to_a_manufactured_solution(device: str) -> None:
    """p = sin(πx) sin(πy) sin(πz) on the unit cube, p = 0 on its boundary.

    Unknowns sit at x = (i + 1) h with h = 1/(n + 1), so the zero values outside
    the grid fall exactly on the boundary. sin(πx) is an eigenvector of the
    discrete Laplacian, so the exact discrete solution is the exact solution
    times ((πh/2) / sin(πh/2))², about 0.3 % larger at n = 16.
    """
    n = 16
    h_m = 1.0 / (n + 1)
    x = (np.arange(n) + 1) * h_m
    exact = np.einsum("i,j,k->ijk", *(np.sin(np.pi * x),) * 3).astype(np.float32)
    discrete = exact * ((np.pi * h_m / 2) / np.sin(np.pi * h_m / 2)) ** 2
    rhs = (-3.0 * np.pi**2 * exact).astype(np.float32)

    p = jacobi_solve(
        wp.zeros((n, n, n), dtype=float, device=device),
        wp.array(rhs, device=device),
        h_m=h_m,
        iterations=1500,
    )

    assert np.abs(p.numpy() - discrete).max() < 1e-4
    assert np.abs(p.numpy() - exact).max() / np.abs(exact).max() < 0.01


def test_jacobi_step_rejects_mismatched_shapes(device: str) -> None:
    p = wp.zeros(SHAPE, dtype=float, device=device)
    small = wp.zeros((4, 4, 4), dtype=float, device=device)

    with pytest.raises(ValueError, match="shape"):
        jacobi_step(p, small, wp.zeros_like(p), h_m=0.1)
    with pytest.raises(ValueError, match="shape"):
        jacobi_step(p, wp.zeros_like(p), small, h_m=0.1)


def test_jacobi_step_rejects_writing_over_its_inputs(device: str) -> None:
    """In place, a sweep would silently become Gauss-Seidel on CPU and a race on CUDA."""
    p = wp.zeros(SHAPE, dtype=float, device=device)
    rhs = wp.zeros_like(p)

    with pytest.raises(ValueError, match="alias"):
        jacobi_step(p, rhs, p, h_m=0.1)
    with pytest.raises(ValueError, match="alias"):
        jacobi_step(p, rhs, rhs, h_m=0.1)


def test_jacobi_solve_rejects_negative_iterations() -> None:
    p = wp.zeros(SHAPE, dtype=float, device="cpu")

    with pytest.raises(ValueError, match="iterations"):
        jacobi_solve(p, wp.zeros_like(p), h_m=0.1, iterations=-1)


@pytest.mark.gpu
def test_jacobi_step_rejects_arrays_on_different_devices() -> None:
    """A CPU array in a CUDA launch is an illegal access that kills the CUDA context."""
    p = wp.zeros(SHAPE, dtype=float, device="cuda:0")

    with pytest.raises(ValueError, match="device"):
        jacobi_step(p, wp.zeros(SHAPE, dtype=float, device="cpu"), wp.zeros_like(p), h_m=0.1)

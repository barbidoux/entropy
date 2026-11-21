"""Multigrid-preconditioned conjugate gradient (`entropy.linalg.mgpcg`), SPEC 5.2, V-LIQ-05."""

import dataclasses

import numpy as np
import pytest
import warp as wp

from entropy.linalg.mgpcg import (
    CELL_AIR,
    CELL_FLUID,
    CELL_SOLID,
    Mgpcg,
    apply,
    system_from_labels,
)


def _labels(array: np.ndarray, device: str) -> wp.array:
    return wp.array(array.astype(np.int32), dtype=wp.int32, device=device)


def _field(array: np.ndarray, device: str) -> wp.array:
    return wp.array(array.astype(np.float32), dtype=float, device=device)


def _random_labels(shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """Mostly fluid, some solid, an air layer on top so the problem is not singular."""
    rng = np.random.default_rng(seed)
    labels = np.where(rng.random(shape) < 0.15, CELL_SOLID, CELL_FLUID)
    labels[:, :, -1] = CELL_AIR
    return labels


def _dirichlet_cube(n: int, device: str):
    """V-LIQ-05: -∇²p = f on the unit cube, p = 0 on its faces, n³ fluid cells.

    One layer of air cells surrounds the fluid; the cube's faces lie halfway between
    the last fluid centre and the first air centre, so θ = 1/2 (second order).
    """
    h = 1.0 / n
    labels = np.full((n + 2,) * 3, CELL_AIR)
    labels[1:-1, 1:-1, 1:-1] = CELL_FLUID
    centres = (np.arange(n + 2) - 0.5) * h
    x, y, z = np.meshgrid(centres, centres, centres, indexing="ij")
    exact = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
    rhs = 3.0 * np.pi**2 * exact
    system = system_from_labels(_labels(labels, device), dx_m=h, air_theta=0.5)
    return system, labels, exact, rhs


def _neumann_box_with_open_top(n: int, device: str):
    """V-LIQ-05 with walls: closed sides and bottom (Neumann), p = 0 on the top face.

    p = cos(πx)·cos(πy)·cos(πz/2) has zero normal derivative on the sides and the
    bottom, and is zero at z = 1. The grid's own boundary is the wall.
    """
    h = 1.0 / n
    labels = np.full((n, n, n + 1), CELL_FLUID)
    labels[:, :, -1] = CELL_AIR
    cx = (np.arange(n) + 0.5) * h
    cz = (np.arange(n + 1) + 0.5) * h
    x, y, z = np.meshgrid(cx, cx, cz, indexing="ij")
    exact = np.cos(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z / 2.0)
    rhs = (2.25 * np.pi**2) * exact
    system = system_from_labels(_labels(labels, device), dx_m=h, air_theta=0.5)
    return system, labels, exact, rhs


def _solve_error(builder, n: int, device: str) -> float:
    system, labels, exact, rhs = builder(n, device)
    x = wp.zeros(labels.shape, dtype=float, device=device)
    result = Mgpcg(system).solve(_field(np.where(labels == CELL_FLUID, rhs, 0.0), device), x)
    assert result.converged
    fluid = labels == CELL_FLUID
    return float(np.abs(x.numpy()[fluid] - exact[fluid]).max())


@pytest.mark.parametrize("builder", [_dirichlet_cube, _neumann_box_with_open_top])
def test_v_liq_05_the_error_falls_four_times_per_refinement(device: str, builder) -> None:
    """V-LIQ-05: second order, the error falls about 4x each time the cells halve."""
    errors = [_solve_error(builder, n, device) for n in (8, 16, 32)]

    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    assert all(3.4 < ratio < 4.6 for ratio in ratios), (errors, ratios)


def test_the_operator_is_the_7_point_laplacian_among_fluid_cells(device: str) -> None:
    shape = (6, 5, 4)
    labels = np.full(shape, CELL_FLUID)
    x = np.random.default_rng(1).standard_normal(shape)
    out = wp.zeros(shape, dtype=float, device=device)
    h = 0.1

    apply(system_from_labels(_labels(labels, device), dx_m=h), _field(x, device), out)

    padded = np.pad(x, 1)
    count = np.pad(np.ones(shape), 1)  # outside the grid is a wall: no neighbour, no flux
    neighbours = sum(
        np.roll(padded, shift, axis)[1:-1, 1:-1, 1:-1] for axis in range(3) for shift in (1, -1)
    )
    degree = sum(
        np.roll(count, shift, axis)[1:-1, 1:-1, 1:-1] for axis in range(3) for shift in (1, -1)
    )
    np.testing.assert_allclose(out.numpy(), (degree * x - neighbours) / h**2, rtol=1e-4, atol=1e-3)


def test_stored_face_weights_give_the_same_operator_as_unit_weights(device: str) -> None:
    """The general path (weights read from arrays) matches the label-derived fast path."""
    shape = (7, 6, 5)
    unit = system_from_labels(_labels(_random_labels(shape, 12), device), dx_m=0.1, air_theta=0.6)
    stored = dataclasses.replace(unit, unit_weights=False)
    x = _field(np.random.default_rng(13).standard_normal(shape), device)
    out_unit = wp.zeros(shape, dtype=float, device=device)
    out_stored = wp.zeros(shape, dtype=float, device=device)

    apply(unit, x, out_unit)
    apply(stored, x, out_stored)

    assert unit.unit_weights
    np.testing.assert_allclose(out_stored.numpy(), out_unit.numpy(), rtol=1e-6, atol=1e-6)


def test_the_operator_is_symmetric(device: str) -> None:
    """CG needs a symmetric operator: x·Ay = y·Ax, with walls, air and θ < 1."""
    shape = (7, 6, 5)
    system = system_from_labels(_labels(_random_labels(shape, 2), device), dx_m=0.1, air_theta=0.4)
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal(shape), rng.standard_normal(shape)
    ax = wp.zeros(shape, dtype=float, device=device)
    ay = wp.zeros(shape, dtype=float, device=device)

    apply(system, _field(x, device), ax)
    apply(system, _field(y, device), ay)

    fluid = system.labels.numpy() == CELL_FLUID
    assert np.dot(x[fluid], ay.numpy()[fluid]) == pytest.approx(
        np.dot(y[fluid], ax.numpy()[fluid]), rel=1e-4
    )


def test_the_preconditioner_is_symmetric(device: str) -> None:
    """A V-cycle with the same smoothing before and after, and R = Pᵀ/8, is symmetric."""
    shape = (12, 10, 9)
    solver = Mgpcg(system_from_labels(_labels(_random_labels(shape, 4), device), dx_m=0.1))
    rng = np.random.default_rng(5)
    labels = solver.system.labels.numpy()
    x, y = (np.where(labels == CELL_FLUID, rng.standard_normal(shape), 0.0) for _ in range(2))
    mx = wp.zeros(shape, dtype=float, device=device)
    my = wp.zeros(shape, dtype=float, device=device)

    solver.precondition(_field(x, device), mx)
    solver.precondition(_field(y, device), my)

    assert np.dot(y.ravel(), mx.numpy().ravel()) == pytest.approx(
        np.dot(x.ravel(), my.numpy().ravel()), rel=1e-4
    )


def test_the_solve_reaches_its_tolerance_and_leaves_other_cells_at_zero(device: str) -> None:
    shape = (16, 14, 12)
    labels = _random_labels(shape, 6)
    solver = Mgpcg(system_from_labels(_labels(labels, device), dx_m=0.0625, air_theta=0.7))
    b = np.where(labels == CELL_FLUID, np.random.default_rng(7).standard_normal(shape), 0.0)
    x = wp.zeros(shape, dtype=float, device=device)

    result = solver.solve(_field(b, device), x, tolerance=1e-5)

    residual = wp.zeros(shape, dtype=float, device=device)
    apply(solver.system, x, residual)
    fluid = labels == CELL_FLUID
    measured = np.linalg.norm((b - residual.numpy())[fluid]) / np.linalg.norm(b[fluid])
    assert result.converged and result.relative_residual <= 1e-5
    assert measured <= 2e-5  # float32 recomputation of the same residual
    assert np.all(x.numpy()[~fluid] == 0.0)


def test_multigrid_needs_far_fewer_iterations_than_a_jacobi_preconditioner(device: str) -> None:
    system, labels, _, rhs = _dirichlet_cube(32, device)
    b = _field(np.where(labels == CELL_FLUID, rhs, 0.0), device)

    multigrid = Mgpcg(system).solve(b, wp.zeros(labels.shape, dtype=float, device=device))
    jacobi = Mgpcg(system, multigrid=False).solve(
        b, wp.zeros(labels.shape, dtype=float, device=device), max_iterations=500
    )

    assert multigrid.converged and jacobi.converged
    assert multigrid.iterations <= 15
    assert jacobi.iterations >= 3 * multigrid.iterations


def test_a_zero_right_hand_side_gives_zero_at_once(device: str) -> None:
    shape = (8, 8, 8)
    solver = Mgpcg(system_from_labels(_labels(_random_labels(shape, 8), device), dx_m=0.1))
    x = wp.zeros(shape, dtype=float, device=device)

    result = solver.solve(wp.zeros(shape, dtype=float, device=device), x)

    assert result.converged and result.iterations == 0
    assert np.all(x.numpy() == 0.0)


@pytest.mark.parametrize("theta", [0.0, -0.5, 1.5, float("nan")])
def test_air_theta_must_be_a_fraction_of_a_cell(theta: float) -> None:
    labels = _labels(np.full((4, 4, 4), CELL_FLUID), "cpu")
    with pytest.raises(ValueError, match="air_theta"):
        system_from_labels(labels, dx_m=0.1, air_theta=theta)


def test_mismatched_arrays_are_refused() -> None:
    solver = Mgpcg(system_from_labels(_labels(_random_labels((8, 8, 8), 9), "cpu"), dx_m=0.1))
    with pytest.raises(ValueError, match="shape"):
        solver.solve(wp.zeros((8, 8, 7), dtype=float), wp.zeros((8, 8, 8), dtype=float))


@pytest.mark.gpu
def test_cpu_and_cuda_solve_alike() -> None:
    """SPEC 9.1 rule 4."""
    solutions = []
    for device in ("cpu", "cuda:0"):
        system, labels, _, rhs = _dirichlet_cube(16, device)
        x = wp.zeros(labels.shape, dtype=float, device=device)
        Mgpcg(system).solve(_field(np.where(labels == CELL_FLUID, rhs, 0.0), device), x)
        solutions.append(x.numpy())

    np.testing.assert_allclose(solutions[0], solutions[1], rtol=1e-4, atol=1e-5)

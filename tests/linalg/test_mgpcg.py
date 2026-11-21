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


def test_refresh_relabels_in_place_and_solves_the_new_problem(device: str) -> None:
    """The liquid relabels cells every substep: the solver must follow without new arrays,
    so CUDA graphs captured on those arrays stay valid (ADR 0016)."""
    shape = (16, 14, 12)
    labels = _labels(_random_labels(shape, 20), device)
    solver = Mgpcg(system_from_labels(labels, dx_m=0.1, air_theta=0.5))
    rng = np.random.default_rng(21)
    x = wp.zeros(shape, dtype=float, device=device)
    solver.solve(_field(rng.standard_normal(shape), device), x)
    diag_before = solver.system.diag

    # The free surface drops by 4 cells, so the coarse levels change too.
    new_labels = _random_labels(shape, 22)
    new_labels[:, :, -5:] = CELL_AIR
    wp.copy(labels, _labels(new_labels, device))
    solver.refresh()
    b = np.where(new_labels == CELL_FLUID, rng.standard_normal(shape), 0.0)
    x.zero_()
    result = solver.solve(_field(b, device), x, check_every=1)

    fresh = Mgpcg(system_from_labels(_labels(new_labels, device), dx_m=0.1, air_theta=0.5))
    expected = wp.zeros(shape, dtype=float, device=device)
    fresh_result = fresh.solve(_field(b, device), expected, check_every=1)
    assert result.converged and solver.system.diag is diag_before
    assert result.iterations == fresh_result.iterations
    if device == "cpu":  # CPU runs are bit-for-bit reproducible (SPEC 9.6)
        np.testing.assert_array_equal(x.numpy(), expected.numpy())
    else:
        np.testing.assert_allclose(x.numpy(), expected.numpy(), rtol=1e-4, atol=1e-5)


def test_multigrid_iterations_barely_grow_with_the_grid(device: str) -> None:
    """A random right-hand side has every frequency: one level alone would need ever more
    iterations as the grid grows; the V-cycle keeps the count about constant."""
    counts = []
    for n in (16, 32):
        labels = np.full((n, n, n), CELL_FLUID)
        labels[:, :, -1] = CELL_AIR
        solver = Mgpcg(system_from_labels(_labels(labels, device), dx_m=1.0 / n, air_theta=0.5))
        b = np.where(
            labels == CELL_FLUID, np.random.default_rng(n).standard_normal(labels.shape), 0.0
        )
        result = solver.solve(
            _field(b, device), wp.zeros(labels.shape, dtype=float, device=device), check_every=1
        )
        assert result.converged and solver.levels >= 3
        counts.append(result.iterations)

    assert counts[1] <= counts[0] + 2, counts


def test_a_single_sweep_on_a_single_level_is_damped_jacobi(device: str) -> None:
    """z = ω Δx² r / diag on fluid cells: an odd sweep count must still land in z."""
    shape = (8, 7, 6)
    labels = _random_labels(shape, 25)
    system = system_from_labels(_labels(labels, device), dx_m=0.1, air_theta=0.5)
    solver = Mgpcg(system, min_coarse_cells=10**6, coarsest_sweeps=1)
    r = np.where(labels == CELL_FLUID, np.random.default_rng(26).standard_normal(shape), 0.0)
    z = wp.zeros(shape, dtype=float, device=device)

    solver.precondition(_field(r, device), z)

    assert solver.levels == 1
    expected = solver.omega * 0.1**2 * r / system.diag.numpy()
    np.testing.assert_allclose(z.numpy(), np.where(labels == CELL_FLUID, expected, 0.0), rtol=1e-5)


@pytest.mark.parametrize("sweeps", [1, 2, 3])
def test_the_preconditioner_is_symmetric_for_any_sweep_count(device: str, sweeps: int) -> None:
    shape = (12, 10, 9)
    system = system_from_labels(_labels(_random_labels(shape, 4), device), dx_m=0.1)
    solver = Mgpcg(system, smoothing_sweeps=sweeps, coarsest_sweeps=2 * sweeps + 1)
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


@pytest.mark.parametrize("tolerance", [1e-8, 0.0, -1e-3, float("nan")])
def test_a_tolerance_below_float32_reach_is_refused(tolerance: float) -> None:
    solver = Mgpcg(system_from_labels(_labels(_random_labels((8, 8, 8), 23), "cpu"), dx_m=0.1))
    zeros = wp.zeros((8, 8, 8), dtype=float, device="cpu")
    with pytest.raises(ValueError, match="tolerance"):
        solver.solve(zeros, wp.zeros((8, 8, 8), dtype=float, device="cpu"), tolerance=tolerance)


def test_settings_cannot_change_after_construction() -> None:
    """On CUDA the V-cycle is captured once: a later change would be ignored there."""
    solver = Mgpcg(system_from_labels(_labels(_random_labels((8, 8, 8), 24), "cpu"), dx_m=0.1))
    with pytest.raises(AttributeError):
        solver.omega = 0.5  # type: ignore[misc]


def _face_thetas(shape: tuple[int, int, int], device: str, value=None, seed: int = 0):
    nx, ny, nz = shape
    rng = np.random.default_rng(seed)
    arrays = []
    for face in ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1)):
        data = np.full(face, value) if value is not None else rng.uniform(0.05, 1.0, face)
        arrays.append(_field(data, device))
    return tuple(arrays)


def test_one_theta_per_face_equal_everywhere_gives_the_single_theta_system(device: str) -> None:
    shape = (8, 7, 6)
    labels = _random_labels(shape, 40)
    single = system_from_labels(_labels(labels, device), dx_m=0.1, air_theta=0.5)
    per_face = system_from_labels(
        _labels(labels, device), dx_m=0.1, face_theta=_face_thetas(shape, device, value=0.5)
    )

    np.testing.assert_allclose(per_face.diag.numpy(), single.diag.numpy(), rtol=1e-6)


def test_each_face_theta_weighs_its_own_face(device: str) -> None:
    """A fluid cell in air: each of its six faces adds 1/θ of that face to the diagonal."""
    labels = np.full((3, 3, 3), CELL_AIR)
    labels[1, 1, 1] = CELL_FLUID
    thetas = [np.ones(face) for face in ((4, 3, 3), (3, 4, 3), (3, 3, 4))]
    own = {(0, 1): 0.2, (0, 2): 0.25, (1, 1): 0.4, (1, 2): 0.5, (2, 1): 0.8, (2, 2): 1.0}
    for (axis, index), value in own.items():
        face = [1, 1, 1]
        face[axis] = index
        thetas[axis][tuple(face)] = value
    dx_m = 0.1

    system = system_from_labels(
        _labels(labels, device),
        dx_m=dx_m,
        face_theta=(
            _field(thetas[0], device),
            _field(thetas[1], device),
            _field(thetas[2], device),
        ),
    )

    unit = np.zeros((3, 3, 3))
    unit[1, 1, 1] = 1.0
    out = wp.zeros((3, 3, 3), dtype=float, device=device)
    apply(system, _field(unit, device), out)

    expected = sum(1.0 / v for v in own.values()) / dx_m**2
    assert out.numpy()[1, 1, 1] == pytest.approx(expected, rel=1e-6)


def test_refresh_reads_the_face_thetas_again(device: str) -> None:
    """The liquid rewrites θ every substep from its level set, then refreshes."""
    shape = (8, 7, 6)
    labels = _random_labels(shape, 44)
    thetas = _face_thetas(shape, device, value=1.0)
    solver = Mgpcg(system_from_labels(_labels(labels, device), dx_m=0.1, face_theta=thetas))

    new = _face_thetas(shape, device, seed=45)
    for target, source in zip(thetas, new, strict=True):
        wp.copy(target, source)
    solver.refresh()

    fresh = system_from_labels(_labels(labels, device), dx_m=0.1, face_theta=new)
    np.testing.assert_allclose(solver.system.diag.numpy(), fresh.diag.numpy(), rtol=1e-6)


def test_face_thetas_must_match_the_faces() -> None:
    labels = _labels(np.full((4, 4, 4), CELL_FLUID), "cpu")
    thetas = _face_thetas((4, 4, 3), "cpu", value=0.5)
    with pytest.raises(ValueError, match="theta"):
        system_from_labels(labels, dx_m=0.1, face_theta=thetas)


def test_one_theta_for_all_and_one_per_face_cannot_both_be_given() -> None:
    labels = _labels(np.full((4, 4, 4), CELL_FLUID), "cpu")
    thetas = _face_thetas((4, 4, 4), "cpu", value=0.5)
    with pytest.raises(ValueError, match="air_theta or face_theta"):
        system_from_labels(labels, dx_m=0.1, air_theta=0.5, face_theta=thetas)


@pytest.mark.gpu
def test_face_thetas_must_be_on_the_labels_device() -> None:
    labels = _labels(np.full((4, 4, 4), CELL_FLUID), "cuda:0")
    thetas = _face_thetas((4, 4, 4), "cpu", value=0.5)
    with pytest.raises(ValueError, match="face_theta x is on cpu"):
        system_from_labels(labels, dx_m=0.1, face_theta=thetas)

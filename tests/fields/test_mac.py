"""Staggered (MAC) grid, interpolation, divergence and gradient (`entropy.fields.mac`)."""

import numpy as np
import pytest
import warp as wp
from hypothesis import given, settings
from hypothesis import strategies as st

from entropy.fields.mac import (
    GridSpec,
    MacVelocity,
    divergence,
    gradient,
    sample_cells,
    sample_velocity,
)

SPEC = GridSpec(shape=(7, 5, 4), dx_m=0.25, origin_m=(1.0, -2.0, 0.5))


def _face_positions_m(spec: GridSpec, axis: int) -> np.ndarray:
    """World positions of the faces normal to `axis`, shaped like that face array."""
    shape = spec.face_shape(axis)
    index = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing="ij"), axis=-1)
    offset = np.full(3, 0.5)
    offset[axis] = 0.0
    return np.asarray(spec.origin_m) + (index + offset) * spec.dx_m


def _cell_positions_m(spec: GridSpec) -> np.ndarray:
    index = np.stack(np.meshgrid(*(np.arange(n) for n in spec.shape), indexing="ij"), axis=-1)
    return np.asarray(spec.origin_m) + (index + 0.5) * spec.dx_m


def _linear_velocity(spec: GridSpec, device: str, a: np.ndarray, b: np.ndarray) -> MacVelocity:
    """Velocity component c = a[c] + b[c] · x, sampled on its faces."""
    components = [
        (a[axis] + _face_positions_m(spec, axis) @ b[axis]).astype(np.float32) for axis in range(3)
    ]
    return MacVelocity(spec, *(wp.array(c, dtype=float, device=device) for c in components))


def _interior_points_m(spec: GridSpec, count: int, seed: int) -> np.ndarray:
    """Points where every component has samples on both sides: half a cell inside."""
    rng = np.random.default_rng(seed)
    low = np.asarray(spec.origin_m) + 0.5 * spec.dx_m
    high = np.asarray(spec.origin_m) + (np.asarray(spec.shape) - 0.5) * spec.dx_m
    return rng.uniform(low, high, size=(count, 3)).astype(np.float32)


def _sample(velocity: MacVelocity, points_m: np.ndarray, device: str) -> np.ndarray:
    points = wp.array(points_m, dtype=wp.vec3, device=device)
    out = wp.zeros(len(points_m), dtype=wp.vec3, device=device)
    sample_velocity(velocity, points, out)
    return out.numpy()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (0, 4, 4), "dx_m": 0.1},
        {"shape": (4, 4), "dx_m": 0.1},
        {"shape": (4, 4, 4), "dx_m": 0.0},
        {"shape": (4, 4, 4), "dx_m": float("nan")},
        {"shape": (4.0, 4, 4), "dx_m": 0.1},
        {"shape": (True, 4, 4), "dx_m": 0.1},
    ],
)
def test_grid_spec_rejects_impossible_grids(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        GridSpec(**kwargs)


def test_grid_spec_normalises_to_tuples() -> None:
    """A list or NumPy integers work, and compare and hash like plain tuples."""
    spec = GridSpec(shape=[7, 5, np.int64(4)], dx_m=np.float32(0.25), origin_m=[1, -2, 0.5])  # type: ignore[arg-type]

    assert spec == SPEC and hash(spec) == hash(SPEC)
    assert type(spec.shape) is tuple and all(type(n) is int for n in spec.shape)


def test_velocity_components_live_on_their_faces(device: str) -> None:
    velocity = MacVelocity.zeros(SPEC, device)

    assert velocity.u.shape == (8, 5, 4)
    assert velocity.v.shape == (7, 6, 4)
    assert velocity.w.shape == (7, 5, 5)
    assert {str(a.device) for a in (velocity.u, velocity.v, velocity.w)} == {device}


def test_velocity_arrays_must_match_their_faces() -> None:
    good = MacVelocity.zeros(SPEC, "cpu")
    with pytest.raises(ValueError, match="v"):
        MacVelocity(SPEC, good.u, wp.zeros((7, 5, 4), dtype=float, device="cpu"), good.w)


def test_sampling_reproduces_a_linear_velocity_field(device: str) -> None:
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(3), rng.standard_normal((3, 3))
    points_m = _interior_points_m(SPEC, 200, seed=4)

    sampled = _sample(_linear_velocity(SPEC, device, a, b), points_m, device)

    np.testing.assert_allclose(sampled, a + points_m @ b.T, rtol=1e-4, atol=1e-4)


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    coefficients=st.lists(st.floats(-5, 5), min_size=12, max_size=12),
    seed=st.integers(0, 2**16),
)
def test_sampling_is_exact_for_any_linear_field(coefficients: list[float], seed: int) -> None:
    a, b = np.array(coefficients[:3]), np.array(coefficients[3:]).reshape(3, 3)
    points_m = _interior_points_m(SPEC, 16, seed)

    sampled = _sample(_linear_velocity(SPEC, "cpu", a, b), points_m, "cpu")

    np.testing.assert_allclose(sampled, a + points_m @ b.T, rtol=1e-4, atol=1e-3)


def test_sampling_cells_reproduces_a_linear_scalar(device: str) -> None:
    field = (3.0 + _cell_positions_m(SPEC) @ np.array([0.5, -1.0, 2.0])).astype(np.float32)
    points_m = _interior_points_m(SPEC, 200, seed=5)
    out = wp.zeros(len(points_m), dtype=float, device=device)

    sample_cells(
        SPEC,
        wp.array(field, dtype=float, device=device),
        wp.array(points_m, dtype=wp.vec3, device=device),
        out,
    )

    expected = 3.0 + points_m @ np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-4)


def _clamped_linear(
    spec: GridSpec, a: np.ndarray, b: np.ndarray, points_m: np.ndarray
) -> np.ndarray:
    """What clamped sampling of the linear field gives: each component at the point moved
    into the box of that component's own samples."""
    origin, dx, shape = np.asarray(spec.origin_m), spec.dx_m, np.asarray(spec.shape)
    out = np.empty_like(points_m, dtype=np.float64)
    for axis in range(3):
        low = origin + np.full(3, 0.5) * dx
        high = origin + (shape - 0.5) * dx
        low[axis], high[axis] = origin[axis], origin[axis] + shape[axis] * dx
        out[:, axis] = a[axis] + np.clip(points_m, low, high) @ b[axis]
    return out


def test_sampling_clamps_to_the_nearest_samples_outside(device: str) -> None:
    """Beyond its samples a component reads the nearest boundary value: no extrapolation."""
    rng = np.random.default_rng(10)
    a, b = rng.standard_normal(3), rng.standard_normal((3, 3))
    near = SPEC.origin_m[0] + 0.1 * SPEC.dx_m  # inside the grid, before the first v and w samples
    points_m = np.array(
        [
            [near, 0.0, 1.0],
            [100.0, 100.0, 100.0],
            [-100.0, -100.0, -100.0],
            [1e10, -1e10, 1e10],
            [np.inf, -np.inf, 3.0],
        ],
        dtype=np.float32,
    )

    sampled = _sample(_linear_velocity(SPEC, device, a, b), points_m, device)

    np.testing.assert_allclose(sampled, _clamped_linear(SPEC, a, b, points_m), rtol=1e-4, atol=1e-4)


def test_a_nan_point_reads_nan(device: str) -> None:
    """A NaN position is an upstream bug: it propagates so the NaN guard (SPEC 5.9) sees it."""
    velocity = _linear_velocity(SPEC, device, np.ones(3), np.zeros((3, 3)))

    sampled = _sample(velocity, np.array([[np.nan, 1.0, 1.0]], dtype=np.float32), device)

    assert np.isnan(sampled).all()


def test_divergence_of_a_linear_field_is_its_trace(device: str) -> None:
    b = np.diag([0.5, -2.0, 4.0])
    out = wp.zeros(SPEC.shape, dtype=float, device=device)

    divergence(_linear_velocity(SPEC, device, np.zeros(3), b), out)

    np.testing.assert_allclose(out.numpy(), np.full(SPEC.shape, 2.5), rtol=1e-4, atol=1e-4)


def test_divergence_of_the_gradient_is_the_7_point_laplacian(device: str) -> None:
    p = np.random.default_rng(6).standard_normal(SPEC.shape).astype(np.float32)
    velocity = MacVelocity.zeros(SPEC, device)
    out = wp.zeros(SPEC.shape, dtype=float, device=device)

    gradient(SPEC, wp.array(p, dtype=float, device=device), velocity)
    divergence(velocity, out)

    inner = (slice(1, -1),) * 3
    padded = np.pad(p, 1)
    laplacian = (
        padded[:-2, 1:-1, 1:-1]
        + padded[2:, 1:-1, 1:-1]
        + padded[1:-1, :-2, 1:-1]
        + padded[1:-1, 2:, 1:-1]
        + padded[1:-1, 1:-1, :-2]
        + padded[1:-1, 1:-1, 2:]
        - 6.0 * p
    ) / SPEC.dx_m**2
    np.testing.assert_allclose(out.numpy()[inner], laplacian[inner], rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gradient_is_zero_on_boundary_faces(device: str, axis: int) -> None:
    """The grid boundary is a closed wall (homogeneous Neumann) for every component."""
    p = np.random.default_rng(7).standard_normal(SPEC.shape).astype(np.float32)
    velocity = MacVelocity.zeros(SPEC, device)

    gradient(SPEC, wp.array(p, dtype=float, device=device), velocity)

    face = (velocity.u, velocity.v, velocity.w)[axis].numpy()
    first: list[slice | int] = [slice(None)] * 3
    last: list[slice | int] = [slice(None)] * 3
    first[axis], last[axis] = 0, -1
    inner = [slice(None)] * 3
    inner[axis] = slice(1, -1)
    assert np.all(face[tuple(first)] == 0.0) and np.all(face[tuple(last)] == 0.0)
    np.testing.assert_allclose(
        face[tuple(inner)], np.diff(p, axis=axis) / SPEC.dx_m, rtol=1e-5, atol=1e-5
    )


def test_divergence_of_the_gradient_is_the_neumann_laplacian_everywhere(device: str) -> None:
    """On boundary cells too: a missing neighbour adds nothing (no flux through the wall)."""
    p = np.random.default_rng(11).standard_normal(SPEC.shape).astype(np.float64)
    velocity = MacVelocity.zeros(SPEC, device)
    out = wp.zeros(SPEC.shape, dtype=float, device=device)

    gradient(SPEC, wp.array(p.astype(np.float32), dtype=float, device=device), velocity)
    divergence(velocity, out)

    neumann = np.zeros_like(p)
    for axis in range(3):
        flux = np.diff(p, axis=axis)  # p[i+1] - p[i] across each interior face
        before = [slice(None)] * 3
        after = [slice(None)] * 3
        before[axis], after[axis] = slice(0, -1), slice(1, None)
        neumann[tuple(before)] += flux
        neumann[tuple(after)] -= flux
    np.testing.assert_allclose(out.numpy(), neumann / SPEC.dx_m**2, rtol=1e-4, atol=1e-3)


def test_operators_refuse_mismatched_arrays() -> None:
    velocity = MacVelocity.zeros(SPEC, "cpu")
    with pytest.raises(ValueError, match="shape"):
        divergence(velocity, wp.zeros((7, 5, 3), dtype=float, device="cpu"))
    points = wp.zeros(3, dtype=wp.vec3, device="cpu")
    with pytest.raises(ValueError, match="one shape"):
        sample_velocity(velocity, points, wp.zeros(4, dtype=wp.vec3, device="cpu"))


@pytest.mark.gpu
def test_arrays_on_two_devices_are_refused() -> None:
    cpu = MacVelocity.zeros(SPEC, "cpu")
    cuda = MacVelocity.zeros(SPEC, "cuda:0")
    with pytest.raises(ValueError, match="one device"):
        MacVelocity(SPEC, cpu.u, cuda.v, cpu.w)
    points = wp.zeros(3, dtype=wp.vec3, device="cuda:0")
    with pytest.raises(ValueError, match="one device"):
        sample_velocity(cpu, points, wp.zeros(3, dtype=wp.vec3, device="cuda:0"))


@pytest.mark.gpu
def test_cpu_and_cuda_sample_the_same_values() -> None:
    """SPEC 9.1 rule 4."""
    rng = np.random.default_rng(8)
    a, b = rng.standard_normal(3), rng.standard_normal((3, 3))
    points_m = _interior_points_m(SPEC, 100, seed=9)

    on_cpu = _sample(_linear_velocity(SPEC, "cpu", a, b), points_m, "cpu")
    on_cuda = _sample(_linear_velocity(SPEC, "cuda:0", a, b), points_m, "cuda:0")

    np.testing.assert_allclose(on_cpu, on_cuda, rtol=1e-6, atol=1e-6)

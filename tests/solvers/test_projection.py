"""Pressure projection of the liquid: free surface and static solids (SPEC 5.2, ADR 0017)."""

import numpy as np
import pytest
import warp as wp
from hypothesis import given, settings
from hypothesis import strategies as st

from entropy.fields.mac import GridSpec, MacVelocity
from entropy.linalg.mgpcg import CELL_AIR, CELL_FLUID, CELL_SOLID
from entropy.solvers.liquid.projection import Projection
from entropy.world.maps import Environment
from entropy.world.materials import load_material

WATER = load_material("water")
GRAVITY_M_PER_S2 = -Environment().gravity_m_per_s2[2]
DT_S = 1.0 / 60.0


def _centres_z(spec: GridSpec) -> np.ndarray:
    return spec.origin_m[2] + (np.arange(spec.shape[2]) + 0.5) * spec.dx_m


def _still_water(spec: GridSpec, depth_m: float, device: str) -> tuple[Projection, MacVelocity]:
    """Water up to `depth_m` in a closed box (the grid's walls), air above, gravity applied."""
    projection = Projection(spec, WATER.density_kg_per_m3, device)
    z = np.broadcast_to(_centres_z(spec), spec.shape)
    labels = np.where(z < depth_m, CELL_FLUID, CELL_AIR).astype(np.int32)
    wp.copy(projection.labels, wp.array(labels, dtype=wp.int32, device=device))
    wp.copy(
        projection.phi_m, wp.array((z - depth_m).astype(np.float32), dtype=float, device=device)
    )
    velocity = MacVelocity.zeros(spec, device)
    velocity.w.fill_(-GRAVITY_M_PER_S2 * DT_S)  # one step of gravity on every face
    return projection, velocity


def test_v_liq_01_still_water_has_hydrostatic_pressure(device: str) -> None:
    """V-LIQ-01: still water 1 m deep, p = rho g d within 2 % at every depth."""
    spec = GridSpec(shape=(8, 8, 24), dx_m=1.0 / 16.0)
    projection, velocity = _still_water(spec, 1.0, device)

    result = projection.project(velocity, DT_S)

    assert result.converged
    z = _centres_z(spec)
    fluid = z < 1.0
    expected = WATER.density_kg_per_m3 * GRAVITY_M_PER_S2 * (1.0 - z[fluid])
    measured = projection.pressure_pa.numpy()[4, 4, fluid]
    np.testing.assert_allclose(measured, expected, rtol=0.02)
    assert (
        np.abs(velocity.w.numpy()[:, :, : int(fluid.sum())]).max() < 1e-4
    )  # the water stays still


def test_the_free_surface_need_not_lie_on_a_cell_face(device: str) -> None:
    """θ per face (Enright et al. 2003) puts p = 0 where the level set says, between centres."""
    spec = GridSpec(shape=(6, 6, 20), dx_m=1.0 / 16.0)
    depth_m = 0.71  # 11.36 cells: the surface cuts a cell
    projection, velocity = _still_water(spec, depth_m, device)

    projection.project(velocity, DT_S)

    z = _centres_z(spec)
    fluid = z < depth_m
    expected = WATER.density_kg_per_m3 * GRAVITY_M_PER_S2 * (depth_m - z[fluid])
    np.testing.assert_allclose(projection.pressure_pa.numpy()[3, 3, fluid], expected, rtol=0.02)


def _random_scene(seed: int, device: str) -> tuple[Projection, MacVelocity, np.ndarray]:
    rng = np.random.default_rng(seed)
    spec = GridSpec(shape=(12, 10, 11), dx_m=0.1)
    projection = Projection(spec, WATER.density_kg_per_m3, device)
    z = np.broadcast_to(_centres_z(spec), spec.shape)
    phi = z - 0.7 + 0.05 * rng.standard_normal(spec.shape)
    labels = np.where(phi < 0.0, CELL_FLUID, CELL_AIR)
    labels[rng.random(spec.shape) < 0.08] = CELL_SOLID
    wp.copy(projection.labels, wp.array(labels.astype(np.int32), dtype=wp.int32, device=device))
    wp.copy(projection.phi_m, wp.array(phi.astype(np.float32), dtype=float, device=device))
    faces = [rng.standard_normal(spec.face_shape(a)).astype(np.float32) for a in range(3)]
    velocity = MacVelocity(spec, *(wp.array(f, dtype=float, device=device) for f in faces))
    return projection, velocity, labels


def _divergence(velocity: MacVelocity) -> np.ndarray:
    u, v, w = (c.numpy().astype(np.float64) for c in (velocity.u, velocity.v, velocity.w))
    return (np.diff(u, axis=0) + np.diff(v, axis=1) + np.diff(w, axis=2)) / velocity.spec.dx_m


@settings(max_examples=12, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16))
def test_v_liq_04_projected_velocity_is_divergence_free(seed: int) -> None:
    """V-LIQ-04: after projection the fluid cells' divergence is ≤ 1e-5 of what it was."""
    projection, velocity, labels = _random_scene(seed, "cpu")
    projection.zero_solid_faces(velocity)
    before = _divergence(velocity)[labels == CELL_FLUID]

    result = projection.project(velocity, DT_S)

    after = _divergence(velocity)[labels == CELL_FLUID]
    assert result.converged
    assert np.linalg.norm(after) <= 1e-5 * np.linalg.norm(before) * 2  # float32 recomputation


def test_faces_against_solids_and_walls_do_not_move(device: str) -> None:
    projection, velocity, labels = _random_scene(7, device)

    projection.project(velocity, DT_S)

    u = velocity.u.numpy()
    assert np.all(u[0] == 0.0) and np.all(u[-1] == 0.0)  # the grid's walls (ADR 0015)
    solid = labels == CELL_SOLID
    touching = np.zeros(u.shape, dtype=bool)
    touching[:-1] |= solid
    touching[1:] |= solid
    assert np.all(u[touching] == 0.0)


def test_a_warm_start_from_the_last_pressure_needs_fewer_iterations(device: str) -> None:
    spec = GridSpec(shape=(8, 8, 24), dx_m=1.0 / 16.0)
    projection, velocity = _still_water(spec, 1.0, device)
    first = projection.project(velocity, DT_S)
    velocity.w.fill_(-GRAVITY_M_PER_S2 * DT_S)

    second = projection.project(velocity, DT_S)

    assert second.converged and second.iterations < first.iterations


@pytest.mark.parametrize(("dt_s", "density"), [(0.0, 1000.0), (DT_S, 0.0), (float("nan"), 1000.0)])
def test_impossible_steps_are_refused(dt_s: float, density: float) -> None:
    spec = GridSpec(shape=(4, 4, 4), dx_m=0.1)
    with pytest.raises(ValueError):
        Projection(spec, density, "cpu").project(MacVelocity.zeros(spec, "cpu"), dt_s)


@pytest.mark.gpu
def test_cpu_and_cuda_project_alike() -> None:
    results = []
    for device in ("cpu", "cuda:0"):
        projection, velocity, _ = _random_scene(11, device)
        projection.project(velocity, DT_S)
        results.append((velocity.w.numpy(), projection.pressure_pa.numpy()))

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-3, atol=1e-2)

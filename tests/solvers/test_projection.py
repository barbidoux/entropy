"""Pressure projection of the liquid: free surface and static solids (SPEC 5.2, ADR 0017)."""

import numpy as np
import pytest
import warp as wp
from hypothesis import HealthCheck, given, settings
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


def _height(spec: GridSpec, axis: int, floor_low: bool) -> np.ndarray:
    """Height (m) of every cell centre above the floor: the wall at the low or high end
    of `axis`, towards which gravity points."""
    along = (np.arange(spec.shape[axis]) + 0.5) * spec.dx_m
    if not floor_low:
        along = spec.shape[axis] * spec.dx_m - along
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.broadcast_to(along.reshape(shape), spec.shape)


def _water(
    spec: GridSpec,
    device: str,
    *,
    labels: np.ndarray,
    phi_m: np.ndarray,
    axis: int = 2,
    floor_low: bool = True,
) -> tuple[Projection, MacVelocity]:
    """A closed box (the grid's walls) with the given labels and level set, and one step
    of gravity towards the floor on every face along `axis`."""
    projection = Projection(spec, WATER.density_kg_per_m3, device)
    wp.copy(projection.labels, wp.array(labels.astype(np.int32), dtype=wp.int32, device=device))
    wp.copy(projection.phi_m, wp.array(phi_m.astype(np.float32), dtype=float, device=device))
    velocity = MacVelocity.zeros(spec, device)
    fall = GRAVITY_M_PER_S2 * DT_S
    (velocity.u, velocity.v, velocity.w)[axis].fill_(-fall if floor_low else fall)
    return projection, velocity


def _still_water(
    spec: GridSpec, depth_m: float, device: str, axis: int = 2, floor_low: bool = True
) -> tuple[Projection, MacVelocity]:
    """Water `depth_m` deep against the floor, air beyond, gravity applied."""
    height = _height(spec, axis, floor_low)
    labels = np.where(height < depth_m, CELL_FLUID, CELL_AIR)
    return _water(
        spec, device, labels=labels, phi_m=height - depth_m, axis=axis, floor_low=floor_low
    )


def _hydrostatic(
    projection: Projection, height: np.ndarray, surface_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Measured pressure and rho g (surface - height), on the fluid cells."""
    fluid = projection.labels.numpy() == CELL_FLUID
    expected = WATER.density_kg_per_m3 * GRAVITY_M_PER_S2 * (surface_m - height[fluid])
    return projection.pressure_pa.numpy()[fluid], expected


def test_v_liq_01_still_water_has_hydrostatic_pressure(device: str) -> None:
    """V-LIQ-01: still water 1 m deep, p = rho g d within 2 % at every depth."""
    spec = GridSpec(shape=(8, 8, 24), dx_m=1.0 / 16.0)
    projection, velocity = _still_water(spec, 1.0, device)

    result = projection.project(velocity, DT_S)

    assert result.converged
    measured, expected = _hydrostatic(projection, _height(spec, 2, True), 1.0)
    np.testing.assert_allclose(measured, expected, rtol=0.02)  # V-LIQ-01, every column
    np.testing.assert_allclose(measured, expected, rtol=1e-4)  # a flat surface is exact
    depth_cells = int((_centres_z(spec) < 1.0).sum())
    # The water stays still, up to and including the surface face, where the ghost
    # pressure acts.
    assert np.abs(velocity.w.numpy()[:, :, : depth_cells + 1]).max() < 1e-4


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("floor_low", [True, False], ids=["fluid-low", "fluid-high"])
def test_the_free_surface_need_not_lie_on_a_cell_face(
    device: str, axis: int, floor_low: bool
) -> None:
    """θ per face (Enright et al. 2003) puts p = 0 where the level set says, between
    centres, along every axis and with the water on either side of the surface."""
    shape = [6, 6, 6]
    shape[axis] = 20
    spec = GridSpec(shape=(shape[0], shape[1], shape[2]), dx_m=1.0 / 16.0)
    depth_m = 0.71  # 11.36 cells: the surface cuts a cell
    projection, velocity = _still_water(spec, depth_m, device, axis, floor_low)

    projection.project(velocity, DT_S)

    measured, expected = _hydrostatic(projection, _height(spec, axis, floor_low), depth_m)
    np.testing.assert_allclose(measured, expected, rtol=0.02)


def test_a_surface_just_above_a_centre_keeps_its_small_theta(device: str) -> None:
    """θ = 0.05: the surface sits 5 % of a cell above the top fluid centre."""
    spec = GridSpec(shape=(6, 6, 20), dx_m=1.0 / 16.0)
    depth_m = (11 + 0.5 + 0.05) / 16.0
    projection, velocity = _still_water(spec, depth_m, device)

    projection.project(velocity, DT_S)

    measured, expected = _hydrostatic(projection, _height(spec, 2, True), depth_m)
    np.testing.assert_allclose(measured, expected, rtol=0.02)


def test_a_surface_closer_than_one_percent_of_a_cell_is_put_at_one_percent(
    device: str,
) -> None:
    """θ is floored at 0.01 (ADR 0017) so that A stays well conditioned."""
    spec = GridSpec(shape=(6, 6, 20), dx_m=1.0 / 16.0)
    top_centre_m = (11 + 0.5) / 16.0
    projection, velocity = _still_water(spec, top_centre_m + 0.001 / 16.0, device)

    projection.project(velocity, DT_S)

    surface_m = top_centre_m + 0.01 / 16.0
    measured, expected = _hydrostatic(projection, _height(spec, 2, True), surface_m)
    np.testing.assert_allclose(measured, expected, rtol=0.02)


def test_where_labels_and_level_set_disagree_the_surface_is_first_order(device: str) -> None:
    """The labels say water up to 1 m, the level set up to 0.9 m: θ = 1, so p = 0 at the
    first air cell's centre (ADR 0017)."""
    spec = GridSpec(shape=(6, 6, 24), dx_m=1.0 / 16.0)
    height = _height(spec, 2, True)
    projection, velocity = _water(
        spec,
        device,
        labels=np.where(height < 1.0, CELL_FLUID, CELL_AIR),
        phi_m=height - 0.9,
    )

    projection.project(velocity, DT_S)

    first_air_centre_m = 1.0 + 0.5 / 16.0
    measured, expected = _hydrostatic(projection, height, first_air_centre_m)
    np.testing.assert_allclose(measured, expected, rtol=0.02)


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


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],  # `device` is a name
)
@given(seed=st.integers(0, 2**16))
def test_v_liq_04_projected_velocity_is_divergence_free(device: str, seed: int) -> None:
    """V-LIQ-04: after projection the fluid cells' divergence is ≤ 1e-5 of what it was."""
    projection, velocity, labels = _random_scene(seed, device)
    projection.zero_solid_faces(velocity)
    before = _divergence(velocity)[labels == CELL_FLUID]

    result = projection.project(velocity, DT_S)

    after = _divergence(velocity)[labels == CELL_FLUID]
    assert result.converged
    assert np.linalg.norm(after) <= 1e-5 * np.linalg.norm(before)


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

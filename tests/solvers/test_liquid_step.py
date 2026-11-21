"""The liquid's full step (SPEC 5.2, 5.9; ADR 0013, 0016, 0017)."""

import math
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from entropy.fields.mac import GridSpec
from entropy.sim.simulation import PacedSolver
from entropy.solvers.interface import Solver
from entropy.solvers.liquid.solver import (
    MAX_SUBSTEPS,
    LiquidSetup,
    LiquidSolver,
    NonFiniteStateError,
    fill_box,
    load_dump,
)
from entropy.world.maps import Environment
from entropy.world.materials import load_material

WATER = load_material("water")
GRAVITY = Environment().gravity_m_per_s2
DX_M = 1.0 / 16.0


def _setup(shape: tuple[int, int, int], dump_dir: Path | None = None) -> LiquidSetup:
    return LiquidSetup(
        spec=GridSpec(shape=shape, dx_m=DX_M),
        density_kg_per_m3=WATER.density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
        dump_dir=dump_dir,
    )


def test_fill_box_seeds_jittered_particles_per_cell() -> None:
    points = fill_box((0.0, 0.0, 0.0), (0.5, 0.25, 0.125), dx_m=DX_M, particles_per_cell=8, seed=1)

    assert len(points) == 8 * 8 * 4 * 2
    assert np.all(points.min(axis=0) >= 0.0)
    assert np.all(points.max(axis=0) <= (0.5, 0.25, 0.125))
    cells = np.floor(points / DX_M).astype(int)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    assert np.all(counts == 8)


def test_the_liquid_is_a_solver_the_simulation_can_pace(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)

    assert isinstance(liquid, Solver)
    assert isinstance(liquid, PacedSolver)


def test_a_falling_blob_accelerates_at_g(device: str) -> None:
    """Free fall, far from walls: after 0.1 s the water moves at g t downwards (2 %)."""
    liquid = LiquidSolver.create(_setup((16, 16, 32)), device)
    liquid.add_particles(fill_box((0.375, 0.375, 1.25), (0.625, 0.625, 1.5), DX_M, 8, seed=2))

    for _ in range(6):
        liquid.step(1.0 / 60.0)

    velocity = liquid.particle_velocities()
    assert velocity[:, 2].mean() == pytest.approx(GRAVITY[2] * 0.1, rel=0.02)
    assert np.abs(velocity[:, :2]).max() < 0.05  # no spurious sideways motion


def _surface_heights(liquid: LiquidSolver, columns: slice = slice(1, -1)) -> np.ndarray:
    """Height (m) where the level set crosses zero, in each inner column of the grid."""
    phi = liquid.fields()["phi_m"].numpy()[columns, columns]
    z = (np.arange(phi.shape[2]) + 0.5) * DX_M
    below = (phi[..., :-1] < 0.0) & (phi[..., 1:] >= 0.0)
    k = np.argmax(below, axis=2)
    lo = np.take_along_axis(phi, k[..., None], axis=2)[..., 0]
    hi = np.take_along_axis(phi, (k + 1)[..., None], axis=2)[..., 0]
    return z[k] + DX_M * lo / (lo - hi)


def test_still_water_stays_still(device: str) -> None:
    """A tank of still water with a flat surface: after 1 s nothing has moved by 1 mm
    (ADR 0017's falsifier), and the surface is where the water ends, within 1 mm."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.5), DX_M, 8, seed=3, jitter=0.0))
    start = liquid.particle_positions().copy()

    for _ in range(60):
        liquid.step(1.0 / 60.0)

    moved = np.linalg.norm(liquid.particle_positions() - start, axis=1)
    assert np.percentile(moved, 99) < 1e-3
    assert np.abs(liquid.particle_velocities()).max() < 5e-3
    heights = _surface_heights(liquid)
    assert np.abs(heights - 0.5).max() < 1e-3


def test_a_rough_surface_settles(device: str) -> None:
    """Jittered particles make a rough surface; gravity flattens it, as with real water."""
    liquid = LiquidSolver.create(_setup((12, 12, 16)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.75, 0.75, 0.5), DX_M, 8, seed=3))
    liquid.step(1.0 / 60.0)
    rough = _surface_heights(liquid).std()

    for _ in range(59):
        liquid.step(1.0 / 60.0)

    assert _surface_heights(liquid).std() < rough / 3
    assert _surface_heights(liquid).mean() == pytest.approx(0.5, abs=2e-3)


def test_particles_never_leave_the_grid(device: str) -> None:
    """The grid's walls are the tank's walls (ADR 0017); particles stay inside them."""
    setup = _setup((8, 8, 8))
    liquid = LiquidSolver.create(setup, device)
    points = fill_box((0.0, 0.0, 0.1), (0.5, 0.5, 0.3), DX_M, 8, seed=4)
    liquid.add_particles(points, velocities=np.tile([3.0, -2.0, 1.0], (len(points), 1)))

    for _ in range(20):
        liquid.step(1.0 / 60.0)

    positions = liquid.particle_positions()
    assert positions.min() >= 0.0
    assert np.all(positions.max(axis=0) <= np.asarray(setup.spec.shape) * DX_M)


def test_substeps_follow_the_cfl_condition_and_the_cap(device: str) -> None:
    """ADR 0013: n = ceil(dt (u + sqrt(g dx)) / dx), at most 4."""
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=5)
    liquid.add_particles(points, velocities=np.tile([2.0, 0.0, 0.0], (len(points), 1)))
    liquid.step(1e-4)  # learn the speed
    speed = np.linalg.norm(liquid.particle_velocities(), axis=1).max()

    cfl = speed + math.sqrt(abs(GRAVITY[2]) * DX_M)
    assert liquid.max_advance_s() == pytest.approx(MAX_SUBSTEPS * DX_M / cfl, rel=0.05)
    liquid.step(1.0 / 30.0)
    assert liquid.diagnostics()["substeps"] == min(MAX_SUBSTEPS, math.ceil(cfl / 30.0 / DX_M))


def test_a_nan_freezes_the_liquid_and_dumps_its_state(device: str, tmp_path: Path) -> None:
    """SPEC 5.9: a NaN stops the simulation and becomes a reproducible dump."""
    liquid = LiquidSolver.create(_setup((8, 8, 8), dump_dir=tmp_path), device)
    points = fill_box((0.125, 0.125, 0.125), (0.375, 0.375, 0.25), DX_M, 8, seed=6)
    velocities = np.zeros_like(points)
    velocities[0] = np.nan
    liquid.add_particles(points, velocities=velocities)

    with pytest.raises(NonFiniteStateError) as raised:
        for _ in range(3):  # detected at the latest one step later (no sync point)
            liquid.step(1.0 / 60.0)

    dump = load_dump(raised.value.dump_path)
    assert raised.value.dump_path.parent == tmp_path
    assert dump["positions_m"].shape == (len(points), 3)
    assert not np.isfinite(dump["velocities_m_per_s"]).all()


def test_diagnostics_count_particles_and_pressure_iterations(device: str) -> None:
    liquid = LiquidSolver.create(_setup((8, 8, 8)), device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (0.5, 0.5, 0.25), DX_M, 8, seed=7))

    liquid.step(1.0 / 60.0)

    diagnostics = liquid.diagnostics()
    assert diagnostics["particles"] == 8 * 8 * 8 * 4
    assert diagnostics["pressure_iterations"] > 0
    assert diagnostics["substeps"] >= 1


def test_adding_beyond_capacity_is_refused() -> None:
    setup = LiquidSetup(
        spec=GridSpec(shape=(4, 4, 4), dx_m=DX_M),
        density_kg_per_m3=WATER.density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
        max_particles=10,
    )
    liquid = LiquidSolver.create(setup, "cpu")
    with pytest.raises(ValueError, match="capacity"):
        liquid.add_particles(np.zeros((11, 3)))


@pytest.mark.gpu
def test_cpu_and_cuda_fall_alike() -> None:
    means = []
    for device in ("cpu", "cuda:0"):
        liquid = LiquidSolver.create(_setup((12, 12, 20)), device)
        liquid.add_particles(fill_box((0.25, 0.25, 0.6), (0.5, 0.5, 0.85), DX_M, 8, seed=8))
        for _ in range(6):
            liquid.step(1.0 / 60.0)
        means.append(liquid.particle_positions().mean(axis=0))
        wp.synchronize()

    np.testing.assert_allclose(means[0], means[1], atol=1e-3)

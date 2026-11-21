"""Validation of the liquid's full step: V-LIQ-02 and V-LIQ-03 (SPEC 9.4, ADR 0019)."""

import math
import tomllib

import numpy as np
import pytest
import warp as wp

from entropy.core.paths import content_dir
from entropy.fields.mac import GridSpec, sample_cells
from entropy.solvers.liquid.solver import LiquidSetup, LiquidSolver, fill_box
from entropy.world.maps import Environment
from entropy.world.materials import load_material

WATER = load_material("water")
GRAVITY = Environment().gravity_m_per_s2
FRAME_S = 1.0 / 60.0


def _liquid(shape: tuple[int, int, int], dx_m: float, device: str) -> LiquidSolver:
    setup = LiquidSetup(
        spec=GridSpec(shape=shape, dx_m=dx_m),
        density_kg_per_m3=WATER.density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
    )
    return LiquidSolver.create(setup, device)


class _Volume:
    """The water's volume (m³) from its level set: the share of 8³ points per cell where
    the level set, interpolated between cell centres, is negative (ADR 0019; 4³ points
    snap a flat surface to quarter cells, about 2 % of this tank's depth)."""

    def __init__(self, spec: GridSpec, device: str, per_axis: int = 8) -> None:
        offsets = (np.arange(per_axis) + 0.5) / per_axis
        axes = [
            spec.origin_m[a] + np.add.outer(np.arange(n), offsets).ravel() * spec.dx_m
            for a, n in enumerate(spec.shape)
        ]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self._spec = spec
        self._points = wp.array(points.astype(np.float32), dtype=wp.vec3, device=device)
        self._phi = wp.zeros(len(points), dtype=float, device=device)
        self._sample_m3 = (spec.dx_m / per_axis) ** 3

    def __call__(self, liquid: LiquidSolver) -> float:
        sample_cells(self._spec, liquid.fields()["phi_m"], self._points, self._phi)
        return float((self._phi.numpy() < 0.0).sum()) * self._sample_m3


@pytest.mark.slow
def test_v_liq_02_a_sloshing_tank_keeps_its_volume(device: str) -> None:
    """V-LIQ-02 (ADR 0019): a tank released with a tilted surface sloshes for 33 s
    (2,000 frames of 1/60 s). Its particles stay the same; its volume, from the level set
    and averaged over each second, stays within 1 % of the first second's.

    The level set of a few particles per cell wobbles by a few tenths of a per cent from
    frame to frame on this small grid; the second's mean keeps any drift and drops the
    wobble. The volume is measured every third frame, 20 times a second."""
    dx_m = 1.0 / 16.0
    shape = (16, 4, 16)
    liquid = _liquid(shape, dx_m, device)
    points = fill_box((0.0, 0.0, 0.0), (1.0, 0.25, 0.7), dx_m, 8, seed=9)
    points = points[points[:, 2] < 0.4 + 0.3 * (points[:, 0] - 0.5)]
    liquid.add_particles(points)
    volume = _Volume(GridSpec(shape=shape, dx_m=dx_m), device)
    nominal_m3 = len(points) * dx_m**3 / 8

    volumes = []
    for frame in range(2000):
        liquid.step(FRAME_S)
        if frame % 3 == 0:
            volumes.append(volume(liquid))

    assert liquid.diagnostics()["particles"] == len(points)
    assert volumes[0] == pytest.approx(nominal_m3, rel=0.01)  # the measure is sound
    per_second = np.asarray(volumes[: len(volumes) // 20 * 20]).reshape(-1, 20).mean(axis=1)
    assert np.abs(per_second / per_second[0] - 1.0).max() <= 0.01
    speeds = np.linalg.norm(liquid.particle_velocities(), axis=1)
    assert speeds.max() > 0.05  # it was still sloshing: the test saw moving water


CLOCK_DELAY = 0.25
"""V-LIQ-03's fixed delay of the simulated clock, in T (ADR 0019): Martin & Moyce's
clock starts from the heater current, before the water moved. Never fitted per run."""


def _martin_moyce() -> tuple[np.ndarray, np.ndarray]:
    """(T, Z): the means of Table 2's records for the 2¼ in column."""
    text = (content_dir() / "validation" / "martin_moyce_1952.toml").read_text("utf-8")
    data = tomllib.loads(text)
    assert data["height_over_width"] == 2.0
    column = data["column_2_25_in"]
    return np.asarray(column["time_dimensionless"]), np.asarray(column["front_dimensionless"])


def _floor_front_m(liquid: LiquidSolver, dx_m: float) -> float:
    """Where the level set crosses zero along the floor's row of cells (m from the wall
    behind the column), averaged across the tank's width."""
    phi = liquid.fields()["phi_m"].numpy()[:, :, 0].mean(axis=1)
    i = int(np.nonzero(phi < 0.0)[0].max())
    if i + 1 == len(phi):
        return len(phi) * dx_m
    return (i + 0.5) * dx_m + dx_m * phi[i] / (phi[i] - phi[i + 1])


def _dam_break(a_m: float, cells_per_a: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """A column a wide and 2a high released against the wall at x = 0; the front's Z = z/a
    after each frame, against T = t sqrt(2g/a)."""
    dx_m = a_m / cells_per_a
    shape = (math.ceil(4.5 * cells_per_a), 4, math.ceil(2.5 * cells_per_a))
    liquid = _liquid(shape, dx_m, device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (a_m, 4 * dx_m, 2 * a_m), dx_m, 8, seed=10))
    scale = math.sqrt(2.0 * abs(GRAVITY[2]) / a_m)
    times, fronts = [0.0], [1.0]
    while times[-1] < 3.2:
        liquid.step(FRAME_S)
        times.append(times[-1] + FRAME_S * scale)
        fronts.append(_floor_front_m(liquid, dx_m) / a_m)
    return np.asarray(times), np.asarray(fronts)


def test_v_liq_03_a_released_column_advances_like_martin_and_moyce(device: str) -> None:
    """V-LIQ-03 (ADR 0019), with the game's 6.25 cm cells: a 1 m column, 16 cells across,
    released at once. With the simulated clock delayed by `CLOCK_DELAY`, every mean of
    Table 2 with 1.19 <= T <= 3 is within 7 %. (Measured: -3 to +1.5 %. Gravity off by
    20 % either way misses by 2 points or more.)"""
    times, fronts = _dam_break(1.0, 16, device)
    t_ref, z_ref = _martin_moyce()
    window = (t_ref >= 1.19) & (t_ref <= 3.0)

    simulated = np.interp(t_ref[window] - CLOCK_DELAY, times, fronts)

    assert window.sum() == 11
    np.testing.assert_allclose(simulated, z_ref[window], rtol=0.07)

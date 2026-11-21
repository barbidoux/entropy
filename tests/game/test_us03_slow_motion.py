"""US-03, slow motion (SPEC 2, ADR 0008, ADR 0013): a dam break run at 0.1x matches the
same at 1x in simulated time. Both meet V-LIQ-03 (the front against Martin & Moyce,
ADR 0019) and keep their water's volume within V-LIQ-02's 1 %.

Gameplay test: frames as the game runs them, each advancing frame time x scale."""

import math

import numpy as np
import pytest

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid.solver import LiquidSetup, LiquidSolver, fill_box
from entropy.world.maps import Environment
from entropy.world.materials import load_material
from tests.solvers.test_liquid_validation import (
    CLOCK_DELAY,
    _floor_front_m,
    _martin_moyce,
    _Volume,
)

GRAVITY = Environment().gravity_m_per_s2
FRAME_S = 1.0 / 60.0


def _dam_break(time_scale: float, device: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    a_m, cells = 1.0, 16
    dx_m = a_m / cells
    spec = GridSpec(shape=(math.ceil(4.5 * cells), 4, math.ceil(2.5 * cells)), dx_m=dx_m)
    setup = LiquidSetup(
        spec=spec,
        density_kg_per_m3=load_material("water").density_kg_per_m3,
        gravity_m_per_s2=GRAVITY,
    )
    liquid = LiquidSolver.create(setup, device)
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (a_m, 4 * dx_m, 2 * a_m), dx_m, 8, seed=10))
    volume = _Volume(spec, device)
    scale = math.sqrt(2.0 * abs(GRAVITY[2]) / a_m)
    dt_s = FRAME_S * time_scale
    times, fronts, volumes = [0.0], [1.0], []
    while times[-1] < 3.2:
        liquid.step(dt_s)
        times.append(times[-1] + dt_s * scale)
        fronts.append(_floor_front_m(liquid, dx_m) / a_m)
        if len(times) % round(1 / time_scale) == 0:  # the same simulated instants
            volumes.append(volume(liquid))
    return np.asarray(times), np.asarray(fronts), volumes


@pytest.mark.slow
@pytest.mark.parametrize("time_scale", [1.0, 0.1])
def test_us_03_a_dam_break_in_slow_motion_meets_the_same_tests(time_scale: float) -> None:
    times, fronts, volumes = _dam_break(time_scale, "cpu")
    t_ref, z_ref = _martin_moyce()
    window = (t_ref >= 1.19) & (t_ref <= 3.0)

    simulated = np.interp(t_ref[window] - CLOCK_DELAY, times, fronts)

    np.testing.assert_allclose(simulated, z_ref[window], rtol=0.07)
    assert np.abs(np.asarray(volumes) / volumes[0] - 1.0).max() <= 0.01

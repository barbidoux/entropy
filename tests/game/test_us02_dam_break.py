"""US-02, dam break (SPEC 2): a water column released in the Test Tank advances like
Martin & Moyce's experiment (V-LIQ-03, ADR 0019).

Gameplay test: the bucket in its dam shape (right click), 2 m³ across the 4 m tank, so
a = 0.5 m and 8 cells across it; the toolbox and the liquid run frame by frame as in
the game. The front is where the level set crosses zero along the floor, averaged
across the tank. On the GPU: the Test Tank is too large for the CPU suite; V-LIQ-03
itself runs on both."""

import math

import numpy as np
import pytest

from entropy.core.paths import content_dir
from entropy.game.input import Action, FrameInput
from entropy.game.toolbox import Toolbox
from entropy.game.tools.base import Aim
from entropy.game.tools.jet import Domain
from entropy.sim.setup import liquid_setup
from entropy.solvers.liquid.solver import LiquidSolver
from entropy.world.maps import load_map
from entropy.world.sdf import OpenBox
from entropy.world.world import build_world
from tests.solvers.test_liquid_validation import CLOCK_DELAY, _martin_moyce

FRAME_S = 1.0 / 60.0


def _frame(**kw: object) -> FrameInput:
    return FrameInput(frozenset(), frozenset(), 0.0, 0.0, **kw)  # type: ignore[arg-type]


@pytest.mark.gpu
def test_us_02_a_dam_released_in_the_test_tank_follows_martin_and_moyce() -> None:
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    setup = liquid_setup(world)
    assert setup is not None
    spec = setup.spec
    liquid = LiquidSolver.create(setup, "cuda:0")
    (ox, oy, oz), (nx, ny, nz) = spec.origin_m, spec.shape
    high = (ox + nx * spec.dx_m, oy + ny * spec.dx_m, oz + nz * spec.dx_m)
    tank = next(shape for shape, _ in world.solids if isinstance(shape, OpenBox))
    domain = Domain(spec.origin_m, high, tank.inner_max_m[2], ground_z_m=-1.0)
    volume_m3 = spec.dx_m**3 / setup.particles_per_cell
    toolbox = Toolbox.standard(
        gravity_m_per_s2=world.environment.gravity_m_per_s2,
        domain=domain,
        liquid=liquid,
        particle_volume_m3=volume_m3,
    )
    toolbox.handle(FrameInput(frozenset(), frozenset({Action.TOOL_2}), 0.0, 0.0))
    toolbox.current.values["volume_m3"] = 2.0
    toolbox.handle(_frame(alternate_pressed=True))
    aim = Aim(origin_m=(0.0, -3.0, 3.0), direction=(0.0, 0.8, -0.6))  # facing +y
    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    toolbox.apply(aim, FRAME_S)
    toolbox.handle(_frame(tool_released=True))

    a_m = math.sqrt(2.0 / (2 * 4.0))
    scale = math.sqrt(2.0 * abs(world.environment.gravity_m_per_s2[2]) / a_m)
    wall_m = high[1]
    times, fronts = [0.0], [1.0]
    while times[-1] < 3.2:
        liquid.step(FRAME_S)
        times.append(times[-1] + FRAME_S * scale)
        phi = liquid.fields()["phi_m"].numpy()[:, :, 0].mean(axis=0)  # along y
        j = int(np.nonzero(phi < 0.0)[0].min())  # the front moves towards -y
        y = oy + (j + 0.5) * spec.dx_m
        if j > 0:
            y -= spec.dx_m * phi[j] / (phi[j] - phi[j - 1])
        fronts.append((wall_m - y) / a_m)

    t_ref, z_ref = _martin_moyce()
    window = (t_ref >= 1.19) & (t_ref <= 3.0)
    simulated = np.interp(t_ref[window] - CLOCK_DELAY, np.asarray(times), np.asarray(fronts))
    np.testing.assert_allclose(simulated, z_ref[window], rtol=0.07)

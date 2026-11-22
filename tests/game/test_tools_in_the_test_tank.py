"""The flood source and the eraser in the Test Tank (#54's gameplay tests), frame by frame
as the game runs them: the toolbox as the simulation's first step. On the GPU: the Test
Tank is too large for the CPU suite."""

from typing import cast

import numpy as np
import pytest

from entropy.core.paths import content_dir
from entropy.game.input import Action, FrameInput
from entropy.game.toolbox import Toolbox
from entropy.game.tools.base import Aim
from entropy.game.tools.eraser import Eraser, first_water_along
from entropy.game.tools.jet import Domain
from entropy.sim.setup import liquid_setup
from entropy.sim.simulation import Simulation
from entropy.solvers.liquid.solver import LiquidSolver
from entropy.world.maps import load_map
from entropy.world.sdf import OpenBox
from entropy.world.world import build_world

FRAME_S = 1.0 / 60.0
DOWN = Aim(origin_m=(0.0, 0.0, 2.8), direction=(0.0, 0.0, -1.0))


def _frame(**kw: object) -> FrameInput:
    return FrameInput(frozenset(), frozenset(), 0.0, 0.0, **kw)  # type: ignore[arg-type]


def _test_tank():
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    setup = liquid_setup(world)
    assert setup is not None
    spec = setup.spec
    liquid = LiquidSolver.create(setup, "cuda:0")
    (ox, oy, oz), (nx, ny, nz) = spec.origin_m, spec.shape
    high = (ox + nx * spec.dx_m, oy + ny * spec.dx_m, oz + nz * spec.dx_m)
    tank = next(shape for shape, _ in world.solids if isinstance(shape, OpenBox))
    domain = Domain(spec.origin_m, high, tank.inner_max_m[2], ground_z_m=-1.0)
    toolbox = Toolbox.standard(
        gravity_m_per_s2=world.environment.gravity_m_per_s2,
        domain=domain,
        liquid=liquid,
        particle_volume_m3=setup.particle_volume_m3,
    )
    return liquid, toolbox, Simulation([liquid]), setup.particle_volume_m3


def _run(simulation: Simulation, toolbox: Toolbox, aim: Aim, seconds: float, **held) -> None:
    simulated_s = 0.0
    while simulated_s < seconds - 1e-9:
        step_s = min(FRAME_S, seconds - simulated_s)
        simulated_s += simulation.step(step_s, player=lambda a: toolbox.apply(aim, a))
        toolbox.handle(_frame(**held))


@pytest.mark.gpu
def test_the_flood_source_fills_the_test_tank_at_its_rate() -> None:
    """0.5 m³/s for 4 s: 2 m³ of water, which settles 12.5 cm deep over the 16 m² floor."""
    liquid, toolbox, simulation, particle_m3 = _test_tank()
    toolbox.handle(FrameInput(frozenset(), frozenset({Action.TOOL_3}), 0.0, 0.0))
    toolbox.current.values["rate_m3_per_s"] = 0.5
    toolbox.handle(_frame(tool_pressed=True, tool_held=True))

    _run(simulation, toolbox, DOWN, 4.0, tool_held=True)
    toolbox.handle(_frame(tool_released=True))
    _run(simulation, toolbox, DOWN, 3.0)

    added_m3 = liquid.diagnostics()["particles"] * particle_m3
    assert added_m3 == pytest.approx(2.0, abs=particle_m3)
    # Not squeezed: the water's own volume (its level set) is nearly all of it (#76 has
    # the rest: the volume correction lags in shallow water).
    assert liquid.diagnostics()["water_volume_m3"] >= 0.94 * added_m3
    heights = liquid.particle_positions()[:, 2]
    assert np.percentile(heights, 99) == pytest.approx(0.125, abs=0.03)


@pytest.mark.gpu
def test_the_eraser_takes_water_out_of_the_test_tank() -> None:
    """A 2 m³ bucket, just landed; the eraser aimed down at its middle takes the water
    within its radius of where the aim meets it, and counts it."""
    liquid, toolbox, simulation, particle_m3 = _test_tank()
    toolbox.handle(FrameInput(frozenset(), frozenset({Action.TOOL_2}), 0.0, 0.0))
    toolbox.current.values["volume_m3"] = 2.0
    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    _run(simulation, toolbox, DOWN, FRAME_S)
    toolbox.handle(_frame(tool_released=True))
    _run(simulation, toolbox, DOWN, 0.1)
    before = liquid.particle_positions().copy()
    hit = first_water_along(liquid, DOWN)
    assert hit is not None

    toolbox.handle(FrameInput(frozenset(), frozenset({Action.TOOL_4}), 0.0, 0.0))
    toolbox.current.values["radius_m"] = 0.3
    toolbox.handle(_frame(tool_pressed=True, tool_held=True))
    toolbox.apply(DOWN, FRAME_S)

    within = np.linalg.norm(before - hit, axis=1) <= 0.3
    assert within.sum() > 1000
    assert liquid.diagnostics()["particles"] == len(before) - within.sum()
    eraser = toolbox.current
    assert isinstance(eraser, Eraser)
    eraser = cast(Eraser, eraser)
    assert eraser.removed_m3 == pytest.approx(within.sum() * particle_m3)

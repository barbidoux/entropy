"""US-01, first splash (SPEC 2): 10 s of hose at 20 L/s puts 200 L ± 2 % of water in the
tank, at 1x and at 0.1x (ADR 0013: slow motion gives each frame less simulated time).

Gameplay test: the toolbox, the jet and the real liquid, frame by frame as the game
runs them (`Simulation.step` with the tools as its first step), with the hose at its
default nozzle speed. The Test Tank runs on the GPU; the CPU runs a 1 m tank at 2 L/s. The
"at 30 fps or more" part is checked on the laptop at the gate (#57)."""

import math

import pytest

from entropy.core.paths import content_dir
from entropy.game.input import FrameInput
from entropy.game.toolbox import Toolbox
from entropy.game.tools.base import Aim
from entropy.game.tools.jet import Domain
from entropy.sim.setup import liquid_setup
from entropy.sim.simulation import Simulation
from entropy.solvers.liquid.solver import LiquidSolver
from entropy.world.maps import Environment, load_map
from entropy.world.materials import load_material
from entropy.world.sdf import OpenBox
from entropy.world.world import World, build_world

FRAME_S = 1.0 / 60.0


def _pour(world: World, flow_l_per_s: float, time_scale: float, device: str) -> float:
    """Hold the hose for 10 s of simulated time from above the rim, let the water land
    for 2 s more, and return the volume of water in the tank (m³)."""
    setup = liquid_setup(world)
    assert setup is not None
    liquid = LiquidSolver.create(setup, device)
    spec = setup.spec
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
    toolbox.current.values["flow_l_per_s"] = flow_l_per_s
    simulation = Simulation([liquid])
    top = tank.inner_max_m[2] + 0.6
    aim = Aim(origin_m=(0.0, 0.0, top), direction=(0.0, 0.6, -0.8))
    dt_s = FRAME_S * time_scale
    hold = FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_held=True)
    toolbox.handle(
        FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_held=True, tool_pressed=True)
    )

    def player(advance_s: float) -> None:
        toolbox.apply(aim, advance_s)

    simulated_s = 0.0
    while simulated_s < 10.0 - 1e-9:
        simulated_s += simulation.step(min(dt_s, 10.0 - simulated_s), player=player)
        toolbox.handle(hold)
    toolbox.handle(FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_released=True))
    while simulated_s < 12.0 - 1e-9:
        simulated_s += simulation.step(dt_s, player=player)
    assert len(toolbox.jet) == 0 and toolbox.jet.lost_parcels == 0
    return liquid.diagnostics()["particles"] * volume_m3


def _small_tank() -> World:
    tank = OpenBox(inner_min_m=(-0.5, -0.5, 0.0), inner_max_m=(0.5, 0.5, 1.0), wall_m=0.02)
    return World("small tank", Environment(), ((tank, load_material("glass")),))


@pytest.mark.slow
def test_us_01_a_small_tank_gets_its_water() -> None:
    """At 1x; 0.1x on the CPU takes a minute, and its volume is exact in test_tools."""
    poured = _pour(_small_tank(), 2.0, 1.0, "cpu")

    assert poured == pytest.approx(0.020, rel=0.02)


@pytest.mark.gpu
@pytest.mark.parametrize("time_scale", [1.0, 0.1])
def test_us_01_ten_seconds_of_hose_in_the_test_tank(time_scale: float) -> None:
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))

    poured = _pour(world, 20.0, time_scale, "cuda:0")

    assert poured == pytest.approx(0.200, rel=0.02)
    assert math.isfinite(poured)

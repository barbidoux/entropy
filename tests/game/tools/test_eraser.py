"""The eraser (SPEC 4.2, #54): it removes the water within its radius of the aim."""

import numpy as np
import pytest

from entropy.fields.mac import GridSpec
from entropy.game.tools.base import Aim, ToolContext, load_tool_spec
from entropy.game.tools.eraser import Eraser, first_water_along
from entropy.game.tools.jet import Domain, Jet
from entropy.solvers.liquid.solver import LiquidSetup, LiquidSolver, fill_box
from entropy.world.maps import Environment
from entropy.world.materials import load_material

DX_M = 0.0625
TANK = Domain(low_m=(0.0, 0.0, 0.0), high_m=(1.0, 1.0, 1.0), rim_z_m=0.75, ground_z_m=-0.1)


def _pool() -> LiquidSolver:
    setup = LiquidSetup(
        spec=GridSpec(shape=(16, 16, 16), dx_m=DX_M),
        density_kg_per_m3=load_material("water").density_kg_per_m3,
        gravity_m_per_s2=Environment().gravity_m_per_s2,
    )
    liquid = LiquidSolver.create(setup, "cpu")
    liquid.add_particles(fill_box((0.0, 0.0, 0.0), (1.0, 1.0, 0.25), DX_M, 8, seed=1))
    liquid.step(1e-5)  # builds the level set
    return liquid


DOWN = Aim(origin_m=(0.5, 0.5, 0.9), direction=(0.0, 0.0, -1.0))


def test_the_aim_finds_the_water_s_surface() -> None:
    point = first_water_along(_pool(), DOWN)

    assert point is not None
    np.testing.assert_allclose(point, (0.5, 0.5, 0.25), atol=DX_M / 4)


def test_an_aim_that_misses_the_water_finds_nothing() -> None:
    up = Aim(origin_m=(0.5, 0.5, 0.9), direction=(0.0, 0.0, 1.0))

    assert first_water_along(_pool(), up) is None


def test_the_eraser_removes_the_water_within_its_radius() -> None:
    liquid = _pool()
    before = liquid.particle_positions().copy()  # on the CPU, numpy() is a view
    eraser = Eraser(load_tool_spec("eraser"))
    eraser.values["radius_m"] = 0.2
    ctx = ToolContext(
        aim=DOWN, liquid=liquid, jet=Jet((0, 0, -9.81)), particle_volume_m3=DX_M**3 / 8, domain=TANK
    )

    eraser.on_hold(ctx, 1.0 / 60.0)

    hit = first_water_along(_pool(), DOWN)
    assert hit is not None
    within = np.linalg.norm(before - hit, axis=1) <= 0.2
    assert liquid.diagnostics()["particles"] == int((~within).sum())
    assert eraser.removed_m3 == pytest.approx(within.sum() * DX_M**3 / 8)

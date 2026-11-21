"""Solver setups built from the world (`entropy.sim.setup`; ADR 0017: the tank's walls
are the liquid grid's walls)."""

import numpy as np
import pytest

from entropy.core.paths import content_dir
from entropy.sim.setup import HEADROOM_M, LIQUID_CELL_M, liquid_setup
from entropy.world.maps import Environment, load_map
from entropy.world.materials import load_material
from entropy.world.sdf import Box, OpenBox
from entropy.world.world import World, build_world

GLASS = load_material("glass")
CONCRETE = load_material("concrete")


def _tank_world(*extra: Box, inner_max_m=(1.0, 1.0, 1.0)) -> World:
    tank = OpenBox(inner_min_m=(-1.0, -1.0, 0.0), inner_max_m=inner_max_m, wall_m=0.02)
    return World(
        name="tank",
        environment=Environment(),
        solids=((tank, GLASS), *((box, CONCRETE) for box in extra)),
    )


def test_the_test_tank_s_liquid_grid_spans_its_cavity_and_some_air_above() -> None:
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))

    setup = liquid_setup(world)

    assert setup is not None
    assert setup.spec.origin_m == (-2.0, -2.0, 0.0)
    assert setup.spec.dx_m == LIQUID_CELL_M == 0.0625
    height_cells = round((2.0 + HEADROOM_M) / LIQUID_CELL_M)
    assert setup.spec.shape == (64, 64, height_cells)
    assert setup.density_kg_per_m3 == load_material("water").density_kg_per_m3
    assert setup.gravity_m_per_s2 == world.environment.gravity_m_per_s2
    assert setup.solid_phi_m is None  # the floor lies under the grid, the glass is its wall


def test_a_solid_inside_the_tank_becomes_the_liquid_s_static_solid() -> None:
    world = _tank_world(Box(centre_m=(0.0, 0.0, 0.25), half_size_m=(0.25, 0.25, 0.25)))

    setup = liquid_setup(world, dx_m=0.125)

    assert setup is not None
    phi = setup.solid_phi_m
    assert phi is not None and phi.shape == setup.spec.shape
    inside = phi[7, 7, 1]  # cell centre (-0.0625, -0.0625, 0.1875), in the box
    outside = phi[1, 1, 1]
    assert inside < 0.0 < outside


def test_a_world_without_a_tank_has_no_liquid() -> None:
    world = World(
        name="open",
        environment=Environment(),
        solids=((Box(centre_m=(0.0, 0.0, -0.1), half_size_m=(5.0, 5.0, 0.1)), CONCRETE),),
    )

    assert liquid_setup(world) is None


def test_a_cavity_that_is_not_whole_cells_rounds_to_the_nearest_cell() -> None:
    world = _tank_world(inner_max_m=(1.03, 1.0, 1.0))  # 2.03 m: 32.48 cells of 6.25 cm

    setup = liquid_setup(world)

    assert setup is not None
    assert setup.spec.shape[0] == 32
    assert abs(setup.spec.shape[0] * setup.spec.dx_m - 2.03) <= setup.spec.dx_m / 2


def test_a_cavity_smaller_than_a_cell_is_refused() -> None:
    with pytest.raises(ValueError, match="cavity"):
        liquid_setup(_tank_world(inner_max_m=(-0.99, 1.0, 1.0)))


def test_the_grid_s_cells_are_where_the_solver_expects_them() -> None:
    """The static solid is sampled at the liquid grid's own cell centres (ADR 0015)."""
    world = _tank_world(Box(centre_m=(0.5, 0.5, 0.5), half_size_m=(0.1, 0.1, 0.1)))

    setup = liquid_setup(world, dx_m=0.1)

    assert setup is not None and setup.solid_phi_m is not None
    i, j, k = 15, 15, 5  # centre (0.55, 0.55, 0.55)
    np.testing.assert_allclose(setup.solid_phi_m[i, j, k], -0.05, atol=1e-6)

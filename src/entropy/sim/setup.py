"""Solver setups built from the world (SPEC 8.1: `sim` reads `world`; solvers do not).

The liquid (ADR 0017): the grid spans the cavity of the world's tank, extended
`HEADROOM_M` upward, so the tank's glass is the grid's wall. Every other solid is
sampled at the grid's cell centres as the liquid's static solid.

Named limits:
- One tank per world: the first open box holds the water. A world without one has
  no liquid (M1).
- A cavity that is not a whole number of cells is rounded to the nearest cell, so
  the grid's wall can sit up to half a cell from the glass.

The other solids' distance is sampled once, at setup, on the CPU; the liquid copies it
to its own device.
"""

from __future__ import annotations

import warp as wp

from entropy.fields.mac import GridSpec
from entropy.solvers.liquid.solver import LiquidSetup
from entropy.world.materials import load_material
from entropy.world.sdf import OpenBox, Solids, sample_on_grid
from entropy.world.world import World

LIQUID_CELL_M = 0.0625
"""The liquid's cell (SPEC 5.2 default): 6.25 cm."""
HEADROOM_M = 0.5
"""Air kept above the tank's rim inside the grid, so splashes can rise (ADR 0017: the
grid's top is closed)."""


def liquid_setup(
    world: World, *, dx_m: float = LIQUID_CELL_M, headroom_m: float = HEADROOM_M
) -> LiquidSetup | None:
    """The liquid of `world`, in its tank; None when the world has no tank."""
    tank = next((shape for shape, _ in world.solids if isinstance(shape, OpenBox)), None)
    if tank is None:
        return None
    low, high = tank.inner_min_m, tank.inner_max_m
    extent = (high[0] - low[0], high[1] - low[1], high[2] - low[2] + headroom_m)
    shape = tuple(round(length / dx_m) for length in extent)
    if min(shape) < 1:
        raise ValueError(f"the tank's cavity {extent} m is smaller than a {dx_m} m cell")
    spec = GridSpec(shape=(shape[0], shape[1], shape[2]), dx_m=dx_m, origin_m=low)
    others = tuple(solid for solid, _ in world.solids if solid is not tank)
    solid_phi = None
    if others:
        phi = wp.zeros(spec.shape, dtype=float, device="cpu")
        sample_on_grid(Solids(others).to_device("cpu"), spec.origin_m, dx_m, phi)
        values = phi.numpy()
        if (values < 0.0).any():
            solid_phi = values
    return LiquidSetup(
        spec=spec,
        density_kg_per_m3=load_material("water").density_kg_per_m3,
        gravity_m_per_s2=world.environment.gravity_m_per_s2,
        solid_phi_m=solid_phi,
    )

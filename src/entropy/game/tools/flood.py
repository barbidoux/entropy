"""The flood source (SPEC 4.2): water rising from the tank's whole floor at a set rate.

Each frame it adds rate * dt of water (dt is simulated time), carrying the fraction of
a particle it could not add to the next frame (#35). The particles appear at random
over the floor's area, in the half particle spacing under the level the tank's water
stands at (its volume over the floor's area), moving up at the rate over the floor's
area: the speed at which the level rises.

Named simplification (CLAUDE.md rule 7): the inflow comes in at the water's level, not
at the floor. Coming in at the floor, it must push the whole column up through the
liquid's volume correction (ADR 0018), which lags in shallow water: measured in the
Test Tank, 2 m³ flooded from the floor in 4 s stood at half its depth, the level set
stuck one cell deep. Over a flat surface the two raise the level alike; over uneven
water, some of it lands in air above the low parts and falls.
"""

from __future__ import annotations

import numpy as np

from entropy.game.tools.base import ToolBase, ToolContext, ToolSpec
from entropy.game.tools.registry import register_tool


@register_tool("flood")
class Flood(ToolBase):
    def __init__(self, spec: ToolSpec, seed: int = 0) -> None:
        super().__init__(spec)
        self._carried_m3 = 0.0
        self._rng = np.random.default_rng(seed)

    def on_press(self, ctx: ToolContext) -> None:
        self._carried_m3 = 0.0

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        domain = ctx.domain
        if dt_s <= 0.0 or domain is None or ctx.liquid is None:
            return
        rate = self.values["rate_m3_per_s"]
        self._carried_m3 += rate * dt_s
        count = int(self._carried_m3 // ctx.particle_volume_m3)
        self._carried_m3 -= count * ctx.particle_volume_m3
        if count == 0:
            return
        low, high = np.asarray(domain.low_m), np.asarray(domain.high_m)
        spacing = ctx.particle_volume_m3 ** (1.0 / 3.0)
        area = float(np.prod(high[:2] - low[:2]))
        held = ctx.liquid.particle_count() + count  # with this frame's water
        level_m = min(low[2] + held * ctx.particle_volume_m3 / area, high[2])
        points = np.empty((count, 3))
        points[:, :2] = low[:2] + self._rng.random((count, 2)) * (high[:2] - low[:2])
        floor_m = low[2] + 0.01 * spacing  # inside the grid, not on its wall
        points[:, 2] = np.maximum(level_m - self._rng.random(count) * 0.5 * spacing, floor_m)
        velocities = np.zeros_like(points)
        velocities[:, 2] = rate / area
        ctx.liquid.add_particles(points.astype(np.float32), velocities=velocities)

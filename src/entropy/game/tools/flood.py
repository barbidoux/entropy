"""The flood source (SPEC 4.2): water rising from the tank's whole floor at a set rate.

Each frame it adds rate * dt of water (dt is simulated time), carrying the fraction of
a particle it could not add to the next frame (#35). The particles appear at random
over the floor, in the lowest half particle spacing, moving up at the rate over the
floor's area: the speed at which the water level rises. The liquid's volume correction
(ADR 0018) makes room for them.
"""

from __future__ import annotations

import numpy as np

from entropy.game.tools.base import ToolBase, ToolContext, ToolSpec


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
        points = np.empty((count, 3))
        points[:, :2] = low[:2] + self._rng.random((count, 2)) * (high[:2] - low[:2])
        points[:, 2] = low[2] + self._rng.random(count) * 0.5 * spacing
        area = float(np.prod(high[:2] - low[:2]))
        velocities = np.zeros_like(points)
        velocities[:, 2] = rate / area
        ctx.liquid.add_particles(points.astype(np.float32), velocities=velocities)

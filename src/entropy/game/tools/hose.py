"""The hose (SPEC 4.2): a jet of water from the camera along the aim (#53).

Each frame it releases flow * dt of water (dt is simulated time: slow motion gives
less) as particles of the liquid's own volume. The fraction of a particle left over
is carried to the next frame (#35), so no water is lost to rounding at any time scale.
The frame's particles leave the nozzle at even times through the frame, so at the
frame's end they are strung along the first stretch of the jet; each starts at a
random point of the nozzle's opening, whose area is flow / speed.
"""

from __future__ import annotations

import math

import numpy as np

from entropy.game.tools.base import ToolBase, ToolContext, ToolSpec

NOZZLE_AHEAD_M = 0.3
"""How far in front of the camera the water leaves the nozzle."""


class Hose(ToolBase):
    def __init__(self, spec: ToolSpec, seed: int = 0) -> None:
        super().__init__(spec)
        self._carried_m3 = 0.0
        self._rng = np.random.default_rng(seed)

    def on_press(self, ctx: ToolContext) -> None:
        self._carried_m3 = 0.0

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        flow_m3_per_s = self.values["flow_l_per_s"] * 1e-3
        speed = self.values["nozzle_speed_m_per_s"]
        self._carried_m3 += flow_m3_per_s * dt_s
        count = int(self._carried_m3 // ctx.particle_volume_m3)
        self._carried_m3 -= count * ctx.particle_volume_m3
        if count == 0:
            return
        aim = np.asarray(ctx.aim.direction)
        nozzle = np.asarray(ctx.aim.origin_m) + NOZZLE_AHEAD_M * aim
        # A random point of the opening, in the plane across the aim.
        side = np.cross(aim, [0.0, 0.0, 1.0])
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(aim, [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(side, aim)
        radius_m = math.sqrt(flow_m3_per_s / (speed * math.pi))
        r = radius_m * np.sqrt(self._rng.random(count))
        angle = 2.0 * math.pi * self._rng.random(count)
        opening = nozzle + (r * np.cos(angle))[:, None] * side + (r * np.sin(angle))[:, None] * up
        # Ages at the frame's end: they left at even times through the frame.
        age = (np.arange(count) + 0.5) / count * dt_s
        gravity = ctx.jet.gravity
        velocity0 = speed * aim
        positions = opening + velocity0 * age[:, None] + 0.5 * gravity * (age**2)[:, None]
        velocities = velocity0 + gravity * age[:, None]
        ctx.jet.launch(positions, velocities)

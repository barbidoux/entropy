"""The eraser (SPEC 4.2, #54): removes the liquid within a radius of where the aim meets it.

While the button is held, each frame finds where the aim's ray first meets the water
(the liquid's level set turning negative, sampled every quarter cell along the ray,
the crossing interpolated between samples) and removes the particles within the radius
of that point. `removed_m3` adds up the volume taken. The level set is sampled on the
liquid's device, and only those samples and the new particle count come back: reading
them waits for the GPU once per frame of erasing, a tool's action, not part of the
step. Removing gas and objects comes with those solvers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from entropy.fields.mac import GridSpec, sample_cells
from entropy.game.tools.base import (
    Aim,
    ErasableLiquid,
    Inflow,
    ToolBase,
    ToolContext,
    ToolSpec,
)
from entropy.game.tools.registry import register_tool


def _sample(phi: wp.array, spec: GridSpec, points: np.ndarray) -> np.ndarray:
    """The cell-centred level set at `points` (m), interpolated on its own device."""
    where = wp.array(points.astype(np.float32), dtype=wp.vec3, device=phi.device)
    out = wp.zeros(len(points), dtype=float, device=phi.device)
    sample_cells(spec, phi, where, out)
    return out.numpy()


def first_water_along(liquid: Any, aim: Aim) -> np.ndarray | None:
    """Where the aim's ray first enters the liquid's water (m), or None."""
    fields = liquid.fields()
    spec = fields["velocity"].spec
    low = np.asarray(spec.origin_m)
    high = low + np.asarray(spec.shape) * spec.dx_m
    origin, direction = np.asarray(aim.origin_m, float), np.asarray(aim.direction, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low, t_high = (low - origin) / direction, (high - origin) / direction
    t_in = np.nanmax(np.minimum(t_low, t_high))
    t_out = np.nanmin(np.maximum(t_low, t_high))
    t_in, t_out = max(t_in, 0.0), t_out
    if not t_out > t_in:
        return None
    step = spec.dx_m / 4.0
    t = np.arange(t_in, t_out, step)
    if len(t) < 2:
        return None
    points = origin + t[:, None] * direction
    phi = _sample(fields["phi_m"], spec, points)
    inside = np.nonzero(phi < 0.0)[0]
    if not len(inside):
        return None
    k = int(inside[0])
    if k == 0:
        return points[0]
    a, b = phi[k - 1], phi[k]
    return points[k - 1] + (points[k] - points[k - 1]) * (a / (a - b))


@register_tool("eraser")
class Eraser(ToolBase):
    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(spec)
        self.removed_m3 = 0.0
        self.status = ""

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        liquid = ctx.liquid.liquid if isinstance(ctx.liquid, Inflow) else ctx.liquid
        if not isinstance(liquid, ErasableLiquid):
            return
        point = first_water_along(liquid, ctx.aim)
        if point is None:
            self.status = "visez l'eau"
            return
        centre = (float(point[0]), float(point[1]), float(point[2]))
        removed = liquid.remove_particles_within(centre, self.values["radius_m"])
        self.removed_m3 += removed * ctx.particle_volume_m3
        self.status = f"{self.removed_m3 * 1e3:.0f} L enlevés"

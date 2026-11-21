"""The eraser (SPEC 4.2, #54): removes the liquid within a radius of where the aim meets it.

While the button is held, each frame finds where the aim's ray first meets the water
(the liquid's level set turning negative, sampled every quarter cell along the ray,
the crossing interpolated between samples) and removes the particles within the radius
of that point. `removed_m3` adds up the volume taken. Reading the level set and the new
particle count waits for the GPU once per frame of erasing: a tool's action, not part of
the step. Removing gas and objects comes with those solvers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from entropy.game.tools.base import Aim, ToolBase, ToolContext, ToolSpec


def _sample(phi: np.ndarray, spec: Any, points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of a cell-centred field, clamped at the grid's edge."""
    g = (points - np.asarray(spec.origin_m)) / spec.dx_m - 0.5
    shape = np.asarray(phi.shape)
    g = np.clip(g, 0.0, shape - 1.0)
    i0 = np.minimum(np.floor(g).astype(int), shape - 2)
    f = g - i0
    out = np.zeros(len(points))
    for corner in range(8):
        offset = np.array([(corner >> axis) & 1 for axis in range(3)])
        weight = np.prod(np.where(offset, f, 1.0 - f), axis=1)
        idx = i0 + offset
        out += weight * phi[idx[:, 0], idx[:, 1], idx[:, 2]]
    return out


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
    phi = _sample(fields["phi_m"].numpy(), spec, points)
    inside = np.nonzero(phi < 0.0)[0]
    if not len(inside):
        return None
    k = int(inside[0])
    if k == 0:
        return points[0]
    a, b = phi[k - 1], phi[k]
    return points[k - 1] + (points[k] - points[k - 1]) * (a / (a - b))


class Eraser(ToolBase):
    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(spec)
        self.removed_m3 = 0.0
        self.status = ""

    def on_hold(self, ctx: ToolContext, dt_s: float) -> None:
        liquid = ctx.liquid
        remove = getattr(liquid, "remove_particles_within", None)
        if liquid is None or remove is None:
            return
        point = first_water_along(liquid, ctx.aim)
        if point is None:
            self.status = "visez l'eau"
            return
        removed = remove(tuple(float(c) for c in point), self.values["radius_m"])
        self.removed_m3 += removed * ctx.particle_volume_m3
        self.status = f"{self.removed_m3 * 1e3:.0f} L enlevés"

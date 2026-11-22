"""The bucket (SPEC 4.2): a volume of water dropped into the tank where the player aims.

One click drops one bucketful: exactly the chosen volume, as a whole number of the
liquid's particles (the last one rounded, under 0.01 % of the smallest bucket), at
rest. Right click switches between two shapes:
- "bloc": as wide as it is deep, hanging from the top of the liquid's grid, centred
  where the aim meets the rim's height over the tank; it falls in;
- "barrage" (a dam, US-02): Martin & Moyce's column, a wide and 2a high on the floor,
  across the tank's whole width against the wall the player faces, so a = sqrt(V/2W)
  for a tank W wide; released at once, it collapses like their dam break.
When the aim does not cross the tank, nothing falls, and `status` says why.
"""

from __future__ import annotations

import math

import numpy as np

from entropy.game.tools.base import ToolBase, ToolContext, ToolSpec
from entropy.game.tools.registry import register_tool


def aim_over_tank(ctx: ToolContext) -> np.ndarray | None:
    """Where the aim meets the rim's height over the tank (m), or None."""
    domain = ctx.domain
    if domain is None:
        return None
    origin, direction = np.asarray(ctx.aim.origin_m), np.asarray(ctx.aim.direction)
    if abs(direction[2]) < 1e-6:
        return None
    t = (domain.rim_z_m - origin[2]) / direction[2]
    if t <= 0.0:
        return None
    point = origin + t * direction
    low, high = np.asarray(domain.low_m), np.asarray(domain.high_m)
    if np.any(point[:2] <= low[:2]) or np.any(point[:2] >= high[:2]):
        return None
    return point


def block_of(count: int, spacing_m: float, centre_xy: np.ndarray, top_z_m: float) -> np.ndarray:
    """`count` particle positions filling a block on a lattice of `spacing_m`, as wide as
    deep, centred on `centre_xy` and hanging from `top_z_m`, bottom layer first."""
    side = max(1, int(np.ceil(count ** (1.0 / 3.0))))
    layers = int(np.ceil(count / side**2))
    i, j, k = np.meshgrid(np.arange(side), np.arange(side), np.arange(layers), indexing="ij")
    cells = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    cells = cells[np.lexsort((cells[:, 1], cells[:, 0], cells[:, 2]))][:count]
    xy = (cells[:, :2] + 0.5 - side / 2.0) * spacing_m + centre_xy
    z = top_z_m - (layers - cells[:, 2] - 0.5) * spacing_m
    return np.column_stack([xy, z])


def dam_of(
    count: int, spacing_m: float, low_m: np.ndarray, high_m: np.ndarray, aim: np.ndarray
) -> np.ndarray:
    """`count` particles in a column a wide and 2a high on the floor, across the tank's
    width, against the wall the horizontal aim points to, bottom layer first."""
    axis = 0 if abs(aim[0]) >= abs(aim[1]) else 1
    across = 1 - axis
    width_m = high_m[across] - low_m[across]
    a_m = math.sqrt(count * spacing_m**3 / (2.0 * width_m))
    n_a = max(1, round(a_m / spacing_m))
    n_w = max(1, round(width_m / spacing_m))
    layers = math.ceil(count / (n_a * n_w))
    i, j, k = np.meshgrid(np.arange(n_a), np.arange(n_w), np.arange(layers), indexing="ij")
    cells = np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1)
    cells = cells[np.lexsort((cells[:, 1], cells[:, 0], cells[:, 2]))][:count]
    points = np.empty((count, 3))
    if aim[axis] >= 0.0:  # the far wall is the high one
        points[:, axis] = high_m[axis] - (cells[:, 0] + 0.5) * spacing_m
    else:
        points[:, axis] = low_m[axis] + (cells[:, 0] + 0.5) * spacing_m
    points[:, across] = low_m[across] + (cells[:, 1] + 0.5) * spacing_m
    points[:, 2] = low_m[2] + (cells[:, 2] + 0.5) * spacing_m
    return points


@register_tool("bucket")
class Bucket(ToolBase):
    def __init__(self, spec: ToolSpec) -> None:
        super().__init__(spec)
        self.status = ""
        self.shape = "bloc"

    def on_alternate(self) -> None:
        self.shape = "barrage" if self.shape == "bloc" else "bloc"
        self.status = f"forme : {self.shape}"

    def on_press(self, ctx: ToolContext) -> None:
        target = aim_over_tank(ctx)
        if target is None or ctx.liquid is None or ctx.domain is None:
            self.status = "visez la cuve"
            return
        count = round(self.values["volume_m3"] / ctx.particle_volume_m3)
        spacing = ctx.particle_volume_m3 ** (1.0 / 3.0)
        fits = min(count, max(int(ctx.liquid.room_particles()), 0))
        dropped = f"{fits * ctx.particle_volume_m3:.3g} m³"
        full = f" (cuve pleine : {dropped} sur {count * ctx.particle_volume_m3:.3g} m³)"
        note = full if fits < count else ""
        if self.shape == "barrage":
            low, high = np.asarray(ctx.domain.low_m), np.asarray(ctx.domain.high_m)
            points = dam_of(count, spacing, low, high, np.asarray(ctx.aim.direction))
            ctx.liquid.add_particles(points.astype(np.float32), velocities=np.zeros_like(points))
            self.status = f"barrage de {dropped} lâché{note}"
            return
        points = block_of(count, spacing, target[:2], ctx.domain.high_m[2] - 0.5 * spacing)
        low = np.asarray(ctx.domain.low_m) + 0.5 * spacing
        high = np.asarray(ctx.domain.high_m) - 0.5 * spacing
        # Slide the block inside the tank where it would overlap a wall.
        shift = np.maximum(low - points.min(axis=0), 0.0) - np.maximum(
            points.max(axis=0) - high, 0.0
        )
        points = points + shift
        ctx.liquid.add_particles(points.astype(np.float32), velocities=np.zeros_like(points))
        self.status = f"{dropped} lâchés{note}"

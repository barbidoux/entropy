"""The signed distance of the static solids, on the device (SPEC 5.2, ADR 0031).

The shapes (boxes, open boxes, half-spaces), their conventions and their exactness are
defined in `entropy.world.sdf`, which builds `DeviceSolids` from a map. The maths lives
here, beside the grids, so that the solvers can query the solids too: the liquid samples
them on its grid, the rigid bodies query them at points of their surfaces (SPEC 8.1:
solvers may import `fields`, not `world`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from entropy.fields import _solid_kernels as _k
from entropy.fields._solid_kernels import nearest as nearest  # for solvers' kernels

Vec3 = tuple[float, float, float]


def _finite(v: Vec3) -> bool:
    return len(v) == 3 and all(math.isfinite(c) for c in v)


@dataclass(frozen=True)
class DeviceSolids:
    """The primitives of a `Solids` as device arrays (vec3 and float, m), one set per kind."""

    box_centres_m: wp.array
    box_halves_m: wp.array
    tank_lows_m: wp.array
    tank_highs_m: wp.array
    tank_walls_m: wp.array
    plane_points_m: wp.array
    plane_normals: wp.array

    def arrays(self) -> list[wp.array]:
        """The arrays in the order the kernels take them."""
        return [
            self.box_centres_m,
            self.box_halves_m,
            self.tank_lows_m,
            self.tank_highs_m,
            self.tank_walls_m,
            self.plane_points_m,
            self.plane_normals,
        ]


def signed_distance(
    solids: DeviceSolids, points_m: wp.array, distance_m: wp.array, normals: wp.array
) -> None:
    """Signed distance (m) and outward unit normal of the union at each point."""
    if not points_m.shape == distance_m.shape == normals.shape:
        raise ValueError("points, distances and normals must have one shape")
    device = solids.box_centres_m.device
    if not points_m.device == distance_m.device == normals.device == device:
        raise ValueError("solids, points and outputs must be on one device")
    wp.launch(
        _k.distance_kernel,
        dim=points_m.shape[0],
        inputs=[*solids.arrays(), points_m, distance_m, normals],
        device=device,
    )


def sample_on_grid(solids: DeviceSolids, origin_m: Vec3, dx_m: float, out: wp.array) -> None:
    """Signed distance (m) at the cell centres of a grid of `out.shape` cells of `dx_m`.

    Cell (i, j, k) is centred at origin + (i + ½, j + ½, k + ½)·dx (ADR 0015).
    Named limit: a solid thinner than a cell can miss every cell centre, so
    its samples may all be positive. A solver must then use point queries,
    face fractions or thicker walls.
    """
    if not (math.isfinite(dx_m) and dx_m > 0) or not _finite(origin_m):
        raise ValueError(f"need a finite origin and a positive cell size, got {origin_m}, {dx_m}")
    if out.device != solids.box_centres_m.device:
        raise ValueError("solids and the grid must be on one device")
    wp.launch(
        _k.grid_kernel,
        dim=out.shape,
        inputs=[*solids.arrays(), wp.vec3(*origin_m), dx_m, out],
        device=out.device,
    )

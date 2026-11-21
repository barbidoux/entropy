"""Static solids as signed distance fields (SPEC 5.2: static geometry enters as an SDF).

A solid is a union of primitives: axis-aligned boxes, open boxes (a tank: walls
and a floor around a cavity open at the top) and half-spaces (a floor). The
signed distance is negative inside a solid and positive outside, in metres. The
normal is the unit gradient of the distance: it points out of the nearest
solid surface, which is the way to push a particle out of it.

Exactness:
- A box's distance is exact (Quilez's formula), and so is a half-space's.
- An open box is its outer box minus its cavity, where the cavity extends
  upward without end: d = max(d_outer, -d_cavity). This is exact inside the
  walls and the floor, where push-out happens, and exact in the cavity and
  beside the tank. Above the rim, over the cavity, it is a lower bound (the
  nearest point is a rim edge).
- A union takes the nearest primitive. This is exact outside every primitive,
  and a bound inside where two primitives overlap. Build solids without
  overlaps, as the open box does internally.

Ties (a point equally near two faces of a box) resolve x first, then y, then z,
and towards + on a mid-plane. Between primitives the first kind wins (boxes,
then open boxes, then half-spaces), then the first of that kind.

This module knows grids only as plain numbers (origin, cell size): `world` sits
beside `fields`, not above it (SPEC 8.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from entropy.world import _sdf_kernels as _k

Vec3 = tuple[float, float, float]


def _finite(v: Sequence[float]) -> bool:
    return len(v) == 3 and all(math.isfinite(c) for c in v)


@dataclass(frozen=True)
class Box:
    """An axis-aligned solid box: centre and half sizes in metres."""

    centre_m: Vec3
    half_size_m: Vec3

    def __post_init__(self) -> None:
        if not _finite(self.centre_m):
            raise ValueError(f"centre_m must be a finite 3D point, got {self.centre_m}")
        if not _finite(self.half_size_m) or min(self.half_size_m) <= 0:
            raise ValueError(f"half_size_m must be positive and finite, got {self.half_size_m}")


@dataclass(frozen=True)
class OpenBox:
    """A tank: a floor and four walls `wall_m` thick around a cavity open at the top.

    The cavity spans `inner_min_m` to `inner_max_m`; the walls rise from the
    bottom of the floor to the cavity's top.
    """

    inner_min_m: Vec3
    inner_max_m: Vec3
    wall_m: float

    def __post_init__(self) -> None:
        lo, hi = self.inner_min_m, self.inner_max_m
        if not (_finite(lo) and _finite(hi)) or any(b <= a for a, b in zip(lo, hi, strict=True)):
            raise ValueError(f"inner_min_m {lo} must be below inner_max_m {hi}")
        if not (math.isfinite(self.wall_m) and self.wall_m > 0):
            raise ValueError(f"wall_m must be positive and finite, got {self.wall_m}")


@dataclass(frozen=True)
class HalfSpace:
    """Everything on the far side of the plane through `point_m` against `normal`."""

    point_m: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        if not (_finite(self.point_m) and _finite(self.normal)) or not any(self.normal):
            raise ValueError(f"need a finite point and a non-zero normal, got {self.normal}")


Shape = Box | OpenBox | HalfSpace


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


@dataclass(frozen=True)
class Solids:
    """A union of boxes, open boxes and half-spaces."""

    shapes: tuple[Shape, ...]

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("a solid needs at least one shape")

    def to_device(self, device: str) -> DeviceSolids:
        def vectors(values: Sequence[Sequence[float]] | Sequence[np.ndarray]) -> wp.array:
            array = np.asarray(values, dtype=np.float32).reshape(-1, 3)
            return wp.array(array, dtype=wp.vec3, device=device)

        boxes = [s for s in self.shapes if isinstance(s, Box)]
        tanks = [s for s in self.shapes if isinstance(s, OpenBox)]
        planes = [s for s in self.shapes if isinstance(s, HalfSpace)]
        walls = np.asarray([t.wall_m for t in tanks], dtype=np.float32)
        return DeviceSolids(
            box_centres_m=vectors([b.centre_m for b in boxes]),
            box_halves_m=vectors([b.half_size_m for b in boxes]),
            tank_lows_m=vectors([t.inner_min_m for t in tanks]),
            tank_highs_m=vectors([t.inner_max_m for t in tanks]),
            tank_walls_m=wp.array(walls, dtype=float, device=device),
            plane_points_m=vectors([p.point_m for p in planes]),
            plane_normals=vectors(
                [np.asarray(p.normal) / np.linalg.norm(p.normal) for p in planes]
            ),
        )


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

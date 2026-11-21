"""Static solids as signed distance fields (SPEC 5.2: static geometry enters as an SDF).

Solids are unions of axis-aligned boxes for now: enough for the Test Tank (an
open glass box on a concrete floor). The signed distance is negative inside a
solid and positive outside, in metres. The normal is the unit gradient of the
distance: it points out of the nearest solid, which is the way to push a
particle out of it.

Box distances are exact (Quilez's formula). For a union the distance is the
minimum over boxes. It is exact outside the union, but inside where boxes
overlap it is only a bound. The open box below builds its walls without
overlaps for that reason.

This module knows grids only as plain numbers (origin, cell size): `world`
sits beside `fields`, not above it (SPEC 8.1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

Vec3 = tuple[float, float, float]


def _finite(v: Sequence[float]) -> bool:
    return len(v) == 3 and all(math.isfinite(c) for c in v)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box: centre and half sizes in metres."""

    centre_m: Vec3
    half_size_m: Vec3

    def __post_init__(self) -> None:
        if not _finite(self.centre_m):
            raise ValueError(f"centre_m must be a finite 3D point, got {self.centre_m}")
        if not _finite(self.half_size_m) or min(self.half_size_m) <= 0:
            raise ValueError(f"half_size_m must be positive and finite, got {self.half_size_m}")


@dataclass(frozen=True)
class DeviceSolids:
    """Box centres and half sizes as device arrays (vec3, m)."""

    centres_m: wp.array
    half_sizes_m: wp.array


@dataclass(frozen=True)
class Solids:
    """A union of boxes."""

    boxes: tuple[Box, ...]

    def __post_init__(self) -> None:
        if not self.boxes:
            raise ValueError("a solid needs at least one box")

    def to_device(self, device: str) -> DeviceSolids:
        centres = np.array([b.centre_m for b in self.boxes], dtype=np.float32)
        halves = np.array([b.half_size_m for b in self.boxes], dtype=np.float32)
        return DeviceSolids(
            wp.array(centres, dtype=wp.vec3, device=device),
            wp.array(halves, dtype=wp.vec3, device=device),
        )


def open_box(*, inner_min_m: Vec3, inner_max_m: Vec3, wall_m: float) -> tuple[Box, ...]:
    """A box open at the top: a floor and four walls `wall_m` thick around the cavity.

    The cavity spans `inner_min_m` to `inner_max_m`. The floor lies under it,
    and the walls stand on the floor up to the cavity's top. The boxes do not
    overlap, so the union distance is exact inside the walls too.
    """
    lo, hi = np.asarray(inner_min_m, float), np.asarray(inner_max_m, float)
    if not (_finite(inner_min_m) and _finite(inner_max_m)) or np.any(hi <= lo):
        raise ValueError(f"inner_min_m {inner_min_m} must be below inner_max_m {inner_max_m}")
    if not (math.isfinite(wall_m) and wall_m > 0):
        raise ValueError(f"wall_m must be positive and finite, got {wall_m}")
    t = wall_m
    outer_lo = lo - np.array([t, t, t])
    outer_hi = np.array([hi[0] + t, hi[1] + t, hi[2]])

    def box(a: np.ndarray, b: np.ndarray) -> Box:
        centre, half = (a + b) / 2, (b - a) / 2
        return Box(tuple(map(float, centre)), tuple(map(float, half)))  # pyright: ignore[reportArgumentType]

    floor = box(outer_lo, np.array([outer_hi[0], outer_hi[1], lo[2]]))
    z0, z1 = lo[2], hi[2]
    x_low = box(np.array([outer_lo[0], outer_lo[1], z0]), np.array([lo[0], outer_hi[1], z1]))
    x_high = box(np.array([hi[0], outer_lo[1], z0]), np.array([outer_hi[0], outer_hi[1], z1]))
    y_low = box(np.array([lo[0], outer_lo[1], z0]), np.array([hi[0], lo[1], z1]))
    y_high = box(np.array([lo[0], hi[1], z0]), np.array([hi[0], outer_hi[1], z1]))
    return (floor, x_low, x_high, y_low, y_high)


@wp.func
def _box_distance(p: wp.vec3, centre: wp.vec3, half: wp.vec3) -> float:
    d = p - centre
    q = wp.vec3(wp.abs(d[0]), wp.abs(d[1]), wp.abs(d[2])) - half
    outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    return wp.length(outside) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)


@wp.func
def _sign(x: float) -> float:
    if x < 0.0:
        return -1.0
    return 1.0


@wp.func
def _box_normal(p: wp.vec3, centre: wp.vec3, half: wp.vec3) -> wp.vec3:
    """Unit gradient of `_box_distance`: away from the box, or to its nearest face inside."""
    d = p - centre
    q = wp.vec3(wp.abs(d[0]), wp.abs(d[1]), wp.abs(d[2])) - half
    s = wp.vec3(_sign(d[0]), _sign(d[1]), _sign(d[2]))
    if wp.max(q[0], wp.max(q[1], q[2])) > 0.0:  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)
        ox = wp.max(q[0], 0.0) * s[0]  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)
        oy = wp.max(q[1], 0.0) * s[1]  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)
        oz = wp.max(q[2], 0.0) * s[2]  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)
        outside = wp.vec3(ox, oy, oz)
        return wp.normalize(outside)  # pyright: ignore[reportCallIssue, reportArgumentType, reportOperatorIssue] (Warp kernel types)
    if q[0] >= q[1] and q[0] >= q[2]:
        return wp.vec3(s[0], 0.0, 0.0)
    if q[1] >= q[2]:
        return wp.vec3(0.0, s[1], 0.0)
    return wp.vec3(0.0, 0.0, s[2])


@wp.func
def _nearest(p: wp.vec3, centres: wp.array(dtype=wp.vec3), halves: wp.array(dtype=wp.vec3)) -> int:
    best = wp.int32(0)  # a typed constructor makes it mutable in the loop
    best_d = _box_distance(p, centres[0], halves[0])
    for b in range(1, centres.shape[0]):
        d = _box_distance(p, centres[b], halves[b])
        if d < best_d:
            best_d = d
            best = b
    return best  # pyright: ignore[reportReturnType] (Warp int32 is the kernel int)


@wp.kernel
def _distance_kernel(
    centres: wp.array(dtype=wp.vec3),
    halves: wp.array(dtype=wp.vec3),
    points_m: wp.array(dtype=wp.vec3),
    distance_m: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    p = points_m[t]
    b = _nearest(p, centres, halves)
    distance_m[t] = _box_distance(p, centres[b], halves[b])
    normals[t] = _box_normal(p, centres[b], halves[b])


@wp.kernel
def _grid_kernel(
    centres: wp.array(dtype=wp.vec3),
    halves: wp.array(dtype=wp.vec3),
    origin_m: wp.vec3,
    dx_m: float,
    out: wp.array3d(dtype=float),
):
    # 3D launch: tid() returns three indices, which Warp's stubs do not express.
    i, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    p = origin_m + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx_m
    b = _nearest(p, centres, halves)
    out[i, j, k] = _box_distance(p, centres[b], halves[b])


def signed_distance(
    solids: DeviceSolids, points_m: wp.array, distance_m: wp.array, normals: wp.array
) -> None:
    """Signed distance (m) and outward unit normal of the union at each point."""
    if not points_m.shape == distance_m.shape == normals.shape:
        raise ValueError("points, distances and normals must have one shape")
    device = solids.centres_m.device
    if not points_m.device == distance_m.device == normals.device == device:
        raise ValueError("solids, points and outputs must be on one device")
    wp.launch(
        _distance_kernel,
        dim=points_m.shape[0],
        inputs=[solids.centres_m, solids.half_sizes_m, points_m, distance_m, normals],
        device=device,
    )


def sample_on_grid(solids: DeviceSolids, origin_m: Vec3, dx_m: float, out: wp.array) -> None:
    """Signed distance (m) at the cell centres of a grid of `out.shape` cells of `dx_m`."""
    if not (math.isfinite(dx_m) and dx_m > 0) or not _finite(origin_m):
        raise ValueError(f"need a finite origin and a positive cell size, got {origin_m}, {dx_m}")
    if out.device != solids.centres_m.device:
        raise ValueError("solids and the grid must be on one device")
    wp.launch(
        _grid_kernel,
        dim=out.shape,
        inputs=[solids.centres_m, solids.half_sizes_m, wp.vec3(*origin_m), dx_m, out],
        device=out.device,
    )

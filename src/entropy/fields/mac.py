"""Staggered (MAC) grid: velocity on cell faces, scalars at cell centres (SPEC 5.2).

A grid of `shape` = (nx, ny, nz) cubic cells of `dx_m` metres starts at
`origin_m`. Cell (i, j, k) is centred at origin + (i + ½, j + ½, k + ½)·dx.
The x component u lives on the faces normal to x, at origin + (i, j + ½,
k + ½)·dx, so it has (nx + 1, ny, nz) values; v and w likewise along y and z.

Interpolation is trilinear between the samples of each component. Beyond the
outermost samples it clamps to the boundary values, so a point outside the
grid reads the nearest value rather than garbage.

Units follow the caller: with velocity in m/s and positions in m, divergence
is in 1/s; the gradient of a pressure in Pa is in Pa/m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class GridSpec:
    """A uniform grid of `shape` cubic cells of `dx_m` metres, from `origin_m`."""

    shape: tuple[int, int, int]
    dx_m: float
    origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(int(n) != n or n < 1 for n in self.shape):
            raise ValueError(f"shape must be three positive cell counts, got {self.shape}")
        if not (math.isfinite(self.dx_m) and self.dx_m > 0):
            raise ValueError(f"dx_m must be a positive length, got {self.dx_m}")
        if len(self.origin_m) != 3 or not all(math.isfinite(c) for c in self.origin_m):
            raise ValueError(f"origin_m must be a finite 3D point, got {self.origin_m}")

    def face_shape(self, axis: int) -> tuple[int, int, int]:
        """Shape of the face array normal to `axis` (0 = x, 1 = y, 2 = z)."""
        shape = list(self.shape)
        shape[axis] += 1
        return (shape[0], shape[1], shape[2])


@dataclass(frozen=True)
class MacVelocity:
    """The three face-centred velocity components of a grid, on one device."""

    spec: GridSpec
    u: wp.array
    v: wp.array
    w: wp.array

    def __post_init__(self) -> None:
        for axis, (name, component) in enumerate(zip("uvw", (self.u, self.v, self.w), strict=True)):
            if component.shape != self.spec.face_shape(axis):
                raise ValueError(
                    f"{name} has shape {component.shape}, its faces {self.spec.face_shape(axis)}"
                )
        if not self.u.device == self.v.device == self.w.device:
            raise ValueError("u, v and w must be on one device")

    @classmethod
    def zeros(cls, spec: GridSpec, device: str) -> MacVelocity:
        u, v, w = (wp.zeros(spec.face_shape(axis), dtype=float, device=device) for axis in range(3))
        return cls(spec, u, v, w)

    @property
    def device(self) -> wp.Device | None:
        return self.u.device


@wp.func
def _base(x: float, n: int) -> int:
    """Lower sample index of the interval holding fractional index `x`, clamped."""
    return wp.clamp(int(wp.floor(x)), 0, wp.max(n - 2, 0))


@wp.func
def _frac(x: float, base: int, n: int) -> float:
    if n == 1:
        return 0.0
    return wp.clamp(x - float(base), 0.0, 1.0)


@wp.func
def _trilinear(a: wp.array3d(dtype=float), x: float, y: float, z: float) -> float:
    """`a` at fractional indices (x, y, z), clamped to its samples."""
    nx = a.shape[0]
    ny = a.shape[1]
    nz = a.shape[2]
    i = _base(x, nx)
    j = _base(y, ny)
    k = _base(z, nz)
    fx = _frac(x, i, nx)
    fy = _frac(y, j, ny)
    fz = _frac(z, k, nz)
    i1 = wp.min(i + 1, nx - 1)
    j1 = wp.min(j + 1, ny - 1)
    k1 = wp.min(k + 1, nz - 1)
    c00 = wp.lerp(a[i, j, k], a[i1, j, k], fx)
    c10 = wp.lerp(a[i, j1, k], a[i1, j1, k], fx)
    c01 = wp.lerp(a[i, j, k1], a[i1, j, k1], fx)
    c11 = wp.lerp(a[i, j1, k1], a[i1, j1, k1], fx)
    return wp.lerp(wp.lerp(c00, c10, fy), wp.lerp(c01, c11, fy), fz)


@wp.kernel
def _sample_velocity_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    origin_m: wp.vec3,
    inv_dx: float,
    points_m: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    g = (points_m[t] - origin_m) * inv_dx  # position in cell units
    out[t] = wp.vec3(
        _trilinear(u, g[0], g[1] - 0.5, g[2] - 0.5),
        _trilinear(v, g[0] - 0.5, g[1], g[2] - 0.5),
        _trilinear(w, g[0] - 0.5, g[1] - 0.5, g[2]),
    )


@wp.kernel
def _sample_cells_kernel(
    field: wp.array3d(dtype=float),
    origin_m: wp.vec3,
    inv_dx: float,
    points_m: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    t = wp.tid()
    g = (points_m[t] - origin_m) * inv_dx
    out[t] = _trilinear(field, g[0] - 0.5, g[1] - 0.5, g[2] - 0.5)


@wp.kernel
def _divergence_kernel(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    inv_dx: float,
    out: wp.array3d(dtype=float),
):
    # 3D launch: tid() returns three indices, which Warp's stubs do not express.
    i, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    out[i, j, k] = (
        (u[i + 1, j, k] - u[i, j, k])
        + (v[i, j + 1, k] - v[i, j, k])
        + (w[i, j, k + 1] - w[i, j, k])
    ) * inv_dx


@wp.kernel
def _gradient_kernel(
    p: wp.array3d(dtype=float),
    di: int,
    dj: int,
    dk: int,
    inv_dx: float,
    out: wp.array3d(dtype=float),
):
    """(di, dj, dk) is the unit offset along the face normal."""
    i, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    along = i * di + j * dj + k * dk
    n = p.shape[0] * di + p.shape[1] * dj + p.shape[2] * dk
    if along == 0 or along == n:
        out[i, j, k] = 0.0  # a boundary face has a cell on one side only
        return
    out[i, j, k] = (p[i, j, k] - p[i - di, j - dj, k - dk]) * inv_dx


def _check_points(points_m: wp.array, out: wp.array, device: wp.Device | None) -> None:
    if points_m.shape != out.shape:
        raise ValueError(f"points {points_m.shape} and out {out.shape} must have one shape")
    if not points_m.device == out.device == device:
        raise ValueError(f"points, out and the grid must be on one device, not {points_m.device}")


def sample_velocity(velocity: MacVelocity, points_m: wp.array, out: wp.array) -> None:
    """Interpolated velocity (vec3) at each point of `points_m` (vec3, m) into `out`."""
    _check_points(points_m, out, velocity.device)
    spec = velocity.spec
    wp.launch(
        _sample_velocity_kernel,
        dim=points_m.shape[0],
        inputs=[
            velocity.u,
            velocity.v,
            velocity.w,
            wp.vec3(*spec.origin_m),
            1.0 / spec.dx_m,
            points_m,
            out,
        ],
        device=velocity.device,
    )


def sample_cells(spec: GridSpec, field: wp.array, points_m: wp.array, out: wp.array) -> None:
    """Interpolated cell-centred `field` at each point of `points_m` (vec3, m) into `out`."""
    if field.shape != spec.shape:
        raise ValueError(f"field has shape {field.shape}, the grid {spec.shape}")
    _check_points(points_m, out, field.device)
    wp.launch(
        _sample_cells_kernel,
        dim=points_m.shape[0],
        inputs=[field, wp.vec3(*spec.origin_m), 1.0 / spec.dx_m, points_m, out],
        device=field.device,
    )


def divergence(velocity: MacVelocity, out: wp.array) -> None:
    """Divergence of `velocity` in each cell into `out` (cell-centred)."""
    spec = velocity.spec
    if out.shape != spec.shape:
        raise ValueError(f"out has shape {out.shape}, the grid {spec.shape}")
    if out.device != velocity.device:
        raise ValueError(f"out is on {out.device}, the velocity on {velocity.device}")
    wp.launch(
        _divergence_kernel,
        dim=spec.shape,
        inputs=[velocity.u, velocity.v, velocity.w, 1.0 / spec.dx_m, out],
        device=velocity.device,
    )


def gradient(spec: GridSpec, p: wp.array, out: MacVelocity) -> None:
    """Gradient of cell-centred `p` on every face of `out`; boundary faces get 0."""
    if p.shape != spec.shape or out.spec != spec:
        raise ValueError(f"p has shape {p.shape} and out grid {out.spec}, expected {spec}")
    if p.device != out.device:
        raise ValueError(f"p is on {p.device}, out on {out.device}")
    for offset, component in zip(
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (out.u, out.v, out.w), strict=True
    ):
        wp.launch(
            _gradient_kernel,
            dim=component.shape,
            inputs=[p, *offset, 1.0 / spec.dx_m, component],
            device=p.device,
        )

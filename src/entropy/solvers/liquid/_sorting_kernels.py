"""Warp kernels of `entropy.solvers.liquid.sorting`.

pyright cannot type Warp's kernel maths: its stubs give vector components union
and index types. This file holds kernels only, so the rules below are relaxed here and nowhere
else; the host code in `sorting.py` stays fully checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false

import warp as wp


@wp.func
def _cell(g: float, n: int) -> int:
    """The cell of fractional index g, clamped to the grid; g is clamped first, so huge
    values and infinities give the same cell on CPU and CUDA."""
    return wp.clamp(int(wp.floor(wp.clamp(g, -1.0, float(n)))), 0, n - 1)


@wp.kernel
def cell_keys(
    positions_m: wp.array(dtype=wp.vec3),
    origin_m: wp.vec3,
    dx_m: float,
    shape: wp.vec3i,
    keys: wp.array(dtype=wp.int32),
    order: wp.array(dtype=wp.int32),
):
    """keys[t] = the x-major index of particle t's bin, bins of `dx_m` on a grid of
    `shape` (the sort passes half-cells; 0 if the position is not finite: the NaN guard
    still sees it), order[t] = t."""
    t = wp.tid()
    p = positions_m[t]
    key = wp.int32(0)  # a variable the branch below may change
    if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
        g = (p - origin_m) / dx_m
        i = _cell(g[0], shape[0])
        j = _cell(g[1], shape[1])
        k = _cell(g[2], shape[2])
        key = wp.int32((i * shape[1] + j) * shape[2] + k)
    keys[t] = key
    order[t] = t


@wp.kernel
def gather(
    order: wp.array(dtype=wp.int32),
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
    out_positions_m: wp.array(dtype=wp.vec3),
    out_velocities: wp.array(dtype=wp.vec3),
    out_affine: wp.array(dtype=wp.mat33),
):
    t = wp.tid()
    s = order[t]
    out_positions_m[t] = positions_m[s]
    out_velocities[t] = velocities[s]
    out_affine[t] = affine[s]

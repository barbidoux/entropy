"""Warp kernels of `entropy.world.sdf`: the distance maths (conventions are there).

pyright cannot type Warp's kernel maths: its stubs give vector components and
operators union types. This file holds kernels only, so the rules below are
relaxed here and nowhere else; the host code in `sdf.py` stays fully checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false

import warp as wp


@wp.func
def _sign(x: float) -> float:
    if x < 0.0:
        return -1.0
    return 1.0


@wp.func
def _from_q(q: wp.vec3) -> float:
    """Distance from the per-axis excess q = |p - c| - h (Quilez)."""
    outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
    inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
    return wp.length(outside) + inside


@wp.func
def _normal_from_q(q: wp.vec3, s: wp.vec3) -> wp.vec3:
    """Unit gradient of `_from_q`, with s the sign of d|p - c|/dp on each axis."""
    if wp.max(q[0], wp.max(q[1], q[2])) > 0.0:
        ox = wp.max(q[0], 0.0) * s[0]
        oy = wp.max(q[1], 0.0) * s[1]
        oz = wp.max(q[2], 0.0) * s[2]
        return wp.normalize(wp.vec3(ox, oy, oz))
    if q[0] >= q[1] and q[0] >= q[2]:
        return wp.vec3(s[0], 0.0, 0.0)
    if q[1] >= q[2]:
        return wp.vec3(0.0, s[1], 0.0)
    return wp.vec3(0.0, 0.0, s[2])


@wp.func
def _box_q(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    d = p - (lo + hi) * 0.5
    return wp.vec3(wp.abs(d[0]), wp.abs(d[1]), wp.abs(d[2])) - (hi - lo) * 0.5


@wp.func
def _box_s(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    d = p - (lo + hi) * 0.5
    return wp.vec3(_sign(d[0]), _sign(d[1]), _sign(d[2]))


@wp.func
def _cavity_q(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    """Like `_box_q` for the box [lo, hi] extended upward without end."""
    return wp.vec3(
        wp.abs(p[0] - (lo[0] + hi[0]) * 0.5) - (hi[0] - lo[0]) * 0.5,
        wp.abs(p[1] - (lo[1] + hi[1]) * 0.5) - (hi[1] - lo[1]) * 0.5,
        lo[2] - p[2],
    )


@wp.func
def _cavity_s(p: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> wp.vec3:
    return wp.vec3(_sign(p[0] - (lo[0] + hi[0]) * 0.5), _sign(p[1] - (lo[1] + hi[1]) * 0.5), -1.0)


@wp.func
def _outer_lo(lo: wp.vec3, wall: float) -> wp.vec3:
    return lo - wp.vec3(wall, wall, wall)


@wp.func
def _outer_hi(hi: wp.vec3, wall: float) -> wp.vec3:
    return wp.vec3(hi[0] + wall, hi[1] + wall, hi[2])


@wp.func
def nearest(
    p: wp.vec3,
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
) -> wp.vec4:
    """(normal x, y, z, distance) of the nearest primitive; the first wins a tie."""
    best = wp.vec4(0.0, 0.0, 1.0, 1.0e30)
    for b in range(box_centres.shape[0]):
        lo = box_centres[b] - box_halves[b]
        hi = box_centres[b] + box_halves[b]
        d = _from_q(_box_q(p, lo, hi))
        if d < best[3]:
            n = _normal_from_q(_box_q(p, lo, hi), _box_s(p, lo, hi))
            best = wp.vec4(n[0], n[1], n[2], d)
    for t in range(tank_lows.shape[0]):
        lo = tank_lows[t]
        hi = tank_highs[t]
        olo = _outer_lo(lo, tank_walls[t])
        ohi = _outer_hi(hi, tank_walls[t])
        d_outer = _from_q(_box_q(p, olo, ohi))
        d_cavity = _from_q(_cavity_q(p, lo, hi))
        d = wp.max(d_outer, -d_cavity)
        if d < best[3]:
            n = -_normal_from_q(_cavity_q(p, lo, hi), _cavity_s(p, lo, hi))
            if d_outer >= -d_cavity:
                n = _normal_from_q(_box_q(p, olo, ohi), _box_s(p, olo, ohi))
            best = wp.vec4(n[0], n[1], n[2], d)
    for h in range(plane_points.shape[0]):
        n = plane_normals[h]
        d = wp.dot(p - plane_points[h], n)
        if d < best[3]:
            best = wp.vec4(n[0], n[1], n[2], d)
    return best


@wp.kernel
def distance_kernel(
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
    points_m: wp.array(dtype=wp.vec3),
    distance_m: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    best = nearest(
        points_m[t],
        box_centres,
        box_halves,
        tank_lows,
        tank_highs,
        tank_walls,
        plane_points,
        plane_normals,
    )
    distance_m[t] = best[3]
    normals[t] = wp.vec3(best[0], best[1], best[2])


@wp.kernel
def grid_kernel(
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
    origin_m: wp.vec3,
    dx_m: float,
    out: wp.array3d(dtype=float),
):
    # 3D launch: tid() returns three indices, which Warp's stubs do not express.
    i, j, k = wp.tid()
    p = origin_m + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx_m
    out[i, j, k] = nearest(
        p, box_centres, box_halves, tank_lows, tank_highs, tank_walls, plane_points, plane_normals
    )[3]

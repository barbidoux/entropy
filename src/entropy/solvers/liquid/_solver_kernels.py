"""Warp kernels of `entropy.solvers.liquid.solver` (labels, level set, extrapolation, advection).

pyright cannot type Warp's kernel maths: its stubs give 3D thread indices and
vector components union types. This file holds kernels only, so the rules below
are relaxed here and nowhere else; the host code in `solver.py` stays fully checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false

import warp as wp

AIR = wp.constant(0)
FLUID = wp.constant(1)
SOLID = wp.constant(2)


@wp.func
def _cell(p: wp.vec3, origin_m: wp.vec3, inv_dx: float, n: wp.vec3i) -> wp.vec3i:
    g = (p - origin_m) * inv_dx
    return wp.vec3i(
        wp.clamp(int(wp.floor(wp.clamp(g[0], -1.0, float(n[0])))), 0, n[0] - 1),
        wp.clamp(int(wp.floor(wp.clamp(g[1], -1.0, float(n[1])))), 0, n[1] - 1),
        wp.clamp(int(wp.floor(wp.clamp(g[2], -1.0, float(n[2])))), 0, n[2] - 1),
    )


@wp.kernel
def label_cells(
    phi_m: wp.array3d(dtype=float),
    solid_phi_m: wp.array3d(dtype=float),
    has_solids: int,
    labels: wp.array3d(dtype=wp.int32),
):
    """Solid where the static solids' distance is negative, fluid where the liquid's level
    set is: labels and level set then agree, so every free-surface face gets its θ."""
    i, j, k = wp.tid()
    if has_solids == 1 and solid_phi_m[i, j, k] < 0.0:
        labels[i, j, k] = SOLID
    elif phi_m[i, j, k] < 0.0:
        labels[i, j, k] = FLUID
    else:
        labels[i, j, k] = AIR


@wp.kernel
def splat_weights(
    positions_m: wp.array(dtype=wp.vec3),
    origin_m: wp.vec3,
    dx_m: float,
    kernel_radius_m: float,
    weight: wp.array3d(dtype=float),
    weighted_x: wp.array3d(dtype=wp.vec3),
):
    """Zhu & Bridson (2005): each particle adds k(|x_c - x_p| / R) and k x_p to the cell
    centres within R, with k(s) = (1 - s²)³."""
    t = wp.tid()
    p = positions_m[t]
    n = wp.vec3i(weight.shape[0], weight.shape[1], weight.shape[2])
    c = _cell(p, origin_m, 1.0 / dx_m, n)
    reach = int(wp.ceil(kernel_radius_m / dx_m))
    for a in range(-reach, reach + 1):
        for b in range(-reach, reach + 1):
            for d in range(-reach, reach + 1):
                i = c[0] + a
                j = c[1] + b
                k = c[2] + d
                if i >= 0 and j >= 0 and k >= 0 and i < n[0] and j < n[1] and k < n[2]:
                    centre = (
                        origin_m + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx_m
                    )
                    s2 = wp.length_sq(centre - p) / (kernel_radius_m * kernel_radius_m)
                    if s2 < 1.0:
                        w = (1.0 - s2) * (1.0 - s2) * (1.0 - s2)
                        wp.atomic_add(weight, i, j, k, w)
                        wp.atomic_add(weighted_x, i, j, k, w * p)


@wp.kernel
def level_set(
    weight: wp.array3d(dtype=float),
    weighted_x: wp.array3d(dtype=wp.vec3),
    origin_m: wp.vec3,
    dx_m: float,
    radius_m: float,
    far_m: float,
    phi_m: wp.array3d(dtype=float),
):
    """phi = |x_c - x̄| - r with x̄ the weighted mean of nearby particles; `far_m` where none."""
    i, j, k = wp.tid()
    w = weight[i, j, k]
    if w <= 0.0:
        phi_m[i, j, k] = far_m
        return
    centre = origin_m + wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5) * dx_m
    phi_m[i, j, k] = wp.length(centre - weighted_x[i, j, k] / w) - radius_m


@wp.kernel
def add_constant(a: wp.array3d(dtype=float), value: float):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] + value


@wp.kernel
def valid_faces(
    labels: wp.array3d(dtype=wp.int32), di: int, dj: int, dk: int, valid: wp.array3d(dtype=wp.int32)
):
    """A face is valid when a fluid cell touches it: the projection set its velocity."""
    i, j, k = wp.tid()
    along = i * di + j * dj + k * dk
    n = labels.shape[0] * di + labels.shape[1] * dj + labels.shape[2] * dk
    ok = 0
    if along > 0 and labels[i - di, j - dj, k - dk] == FLUID:
        ok = 1
    if along < n and labels[i, j, k] == FLUID:
        ok = 1
    valid[i, j, k] = ok


@wp.kernel
def extrapolate(
    u: wp.array3d(dtype=float),
    valid: wp.array3d(dtype=wp.int32),
    u_out: wp.array3d(dtype=float),
    valid_out: wp.array3d(dtype=wp.int32),
):
    """One layer: an invalid face takes the mean of its valid neighbours along the axes."""
    i, j, k = wp.tid()
    u_out[i, j, k] = u[i, j, k]
    valid_out[i, j, k] = valid[i, j, k]
    if valid[i, j, k] == 1:
        return
    total = wp.float32(0.0)
    count = wp.int32(0)
    for axis in range(3):
        for sign in range(2):
            ii = i
            jj = j
            kk = k
            step = 2 * sign - 1
            if axis == 0:
                ii = i + step
            elif axis == 1:
                jj = j + step
            else:
                kk = k + step
            inside = ii >= 0 and jj >= 0 and kk >= 0
            inside = inside and ii < u.shape[0] and jj < u.shape[1] and kk < u.shape[2]
            if inside and valid[ii, jj, kk] == 1:
                total += u[ii, jj, kk]
                count += 1
    if count > 0:
        u_out[i, j, k] = total / float(count)
        valid_out[i, j, k] = 1


@wp.kernel
def midpoint(
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    half_dt_s: float,
    out: wp.array(dtype=wp.vec3),
):
    t = wp.tid()
    out[t] = positions_m[t] + velocities[t] * half_dt_s


@wp.kernel
def move(
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt_s: float,
    low_m: wp.vec3,
    high_m: wp.vec3,
):
    """x += dt v, then kept inside the grid: its walls are the tank's walls (ADR 0017)."""
    t = wp.tid()
    p = positions_m[t] + velocities[t] * dt_s
    positions_m[t] = wp.vec3(
        wp.clamp(p[0], low_m[0], high_m[0]),
        wp.clamp(p[1], low_m[1], high_m[1]),
        wp.clamp(p[2], low_m[2], high_m[2]),
    )


@wp.kernel
def push_out(
    positions_m: wp.array(dtype=wp.vec3),
    distance_m: wp.array(dtype=float),
    plus: wp.array(dtype=float),
    step_m: float,
):
    """A particle inside a static solid moves out along the distance's gradient. `plus`
    holds the distance at the point moved by `step_m` along x, y and z (3 per particle)."""
    t = wp.tid()
    d = distance_m[t]
    if d >= 0.0:
        return
    g = wp.vec3(plus[3 * t] - d, plus[3 * t + 1] - d, plus[3 * t + 2] - d) / step_m
    length = wp.length(g)
    if length > 0.0:
        positions_m[t] = positions_m[t] - g / length * d


@wp.kernel
def offsets(positions_m: wp.array(dtype=wp.vec3), step_m: float, out: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    p = positions_m[t]
    out[3 * t] = p + wp.vec3(step_m, 0.0, 0.0)
    out[3 * t + 1] = p + wp.vec3(0.0, step_m, 0.0)
    out[3 * t + 2] = p + wp.vec3(0.0, 0.0, step_m)


@wp.kernel
def status(
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """out[0] = largest speed (m/s); out[1] = 1 if any position or velocity is not finite."""
    t = wp.tid()
    p = positions_m[t]
    v = velocities[t]
    finite = wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
    finite = finite and wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])
    if not finite:
        out[1] = 1.0
        return
    wp.atomic_max(out, 0, wp.length(v))

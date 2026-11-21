"""Warp kernels of `entropy.linalg.mgpcg` (the operator, the V-cycle and CG steps).

pyright cannot type Warp's kernel maths: its stubs give 3D thread indices and
vector components union types. This file holds kernels only, so the rules
below are relaxed here and nowhere else; the host code in `mgpcg.py` stays
fully checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false

import warp as wp

AIR = wp.constant(0)
FLUID = wp.constant(1)
SOLID = wp.constant(2)

# Slots of the float64 scalar array that keeps CG's scalars on the device.
RZ = wp.constant(0)  # r·z of the previous iteration
PQ = wp.constant(1)  # p·Ap
RZ_NEW = wp.constant(2)  # r·z of this iteration
RR = wp.constant(3)  # r·r, for the convergence check
BB = wp.constant(4)  # b·b over fluid cells, for the relative residual
ANY_AIR = wp.constant(5)  # 1 if some cell is air, else 0: A is then singular (ADR 0021)
FLUID_CELLS = wp.constant(6)  # how many cells are fluid, counted only with no air
FLUID_SUM = wp.constant(7)  # a vector's sum over the fluid cells, for its mean
SCALARS = 8


@wp.kernel
def face_weights(
    labels: wp.array3d(dtype=wp.int32), di: int, dj: int, dk: int, w: wp.array3d(dtype=float)
):
    """1 on a face between two non-solid cells inside the grid, 0 elsewhere (walls)."""
    i, j, k = wp.tid()
    along = i * di + j * dj + k * dk
    n = labels.shape[0] * di + labels.shape[1] * dj + labels.shape[2] * dk
    if along == 0 or along == n:
        w[i, j, k] = 0.0  # the grid boundary is a wall (ADR 0015)
        return
    if labels[i, j, k] == SOLID or labels[i - di, j - dj, k - dk] == SOLID:
        w[i, j, k] = 0.0
        return
    w[i, j, k] = 1.0


@wp.func
def _term(labels: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, w: float, inv_theta: float):
    """Contribution of one face to the diagonal: w to a fluid cell, w/θ to air."""
    label = labels[i, j, k]
    if label == FLUID:
        return w
    if label == AIR:
        return w * inv_theta
    return 0.0


@wp.kernel
def diagonal(
    labels: wp.array3d(dtype=wp.int32),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    inv_theta: float,
    diag: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        diag[i, j, k] = 1.0  # unused; keeps divisions safe
        return
    nx = labels.shape[0]
    ny = labels.shape[1]
    nz = labels.shape[2]
    d = wp.float32(0.0)
    if i > 0:
        d += _term(labels, i - 1, j, k, wx[i, j, k], inv_theta)
    if i < nx - 1:
        d += _term(labels, i + 1, j, k, wx[i + 1, j, k], inv_theta)
    if j > 0:
        d += _term(labels, i, j - 1, k, wy[i, j, k], inv_theta)
    if j < ny - 1:
        d += _term(labels, i, j + 1, k, wy[i, j + 1, k], inv_theta)
    if k > 0:
        d += _term(labels, i, j, k - 1, wz[i, j, k], inv_theta)
    if k < nz - 1:
        d += _term(labels, i, j, k + 1, wz[i, j, k + 1], inv_theta)
    diag[i, j, k] = wp.max(d, 1.0e-12)  # an isolated fluid cell: no flux at all


@wp.func
def _face_term(labels: wp.array3d(dtype=wp.int32), i: int, j: int, k: int, w: float, theta: float):
    """Like `_term` with this face's own θ, kept in [1e-3, 1] so 1/θ stays finite."""
    return _term(labels, i, j, k, w, 1.0 / wp.clamp(theta, 1.0e-3, 1.0))


@wp.kernel
def diagonal_face_theta(
    labels: wp.array3d(dtype=wp.int32),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    tx: wp.array3d(dtype=float),
    ty: wp.array3d(dtype=float),
    tz: wp.array3d(dtype=float),
    diag: wp.array3d(dtype=float),
):
    """`diagonal` with one θ per face (tx, ty, tz shaped like wx, wy, wz)."""
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        diag[i, j, k] = 1.0
        return
    nx = labels.shape[0]
    ny = labels.shape[1]
    nz = labels.shape[2]
    d = wp.float32(0.0)
    if i > 0:
        d += _face_term(labels, i - 1, j, k, wx[i, j, k], tx[i, j, k])
    if i < nx - 1:
        d += _face_term(labels, i + 1, j, k, wx[i + 1, j, k], tx[i + 1, j, k])
    if j > 0:
        d += _face_term(labels, i, j - 1, k, wy[i, j, k], ty[i, j, k])
    if j < ny - 1:
        d += _face_term(labels, i, j + 1, k, wy[i, j + 1, k], ty[i, j + 1, k])
    if k > 0:
        d += _face_term(labels, i, j, k - 1, wz[i, j, k], tz[i, j, k])
    if k < nz - 1:
        d += _face_term(labels, i, j, k + 1, wz[i, j, k + 1], tz[i, j, k + 1])
    diag[i, j, k] = wp.max(d, 1.0e-12)


@wp.func
def _neighbour(
    labels: wp.array3d(dtype=wp.int32), x: wp.array3d(dtype=float), i: int, j: int, k: int, w: float
):
    if labels[i, j, k] == FLUID:
        return w * x[i, j, k]
    return 0.0


@wp.func
def _w(w: wp.array3d(dtype=float), i: int, j: int, k: int, unit: int):
    """A face weight; 1 without reading memory when the system's weights are all 0 or 1
    (between two fluid cells, the only faces this is asked for, it is then 1)."""
    if unit == 1:
        return 1.0
    return w[i, j, k]


@wp.func
def _apply_at(
    labels: wp.array3d(dtype=wp.int32),
    diag: wp.array3d(dtype=float),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    unit: int,
    x: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
):
    """(A x) at a fluid cell, before the 1/Δx² factor."""
    nx = labels.shape[0]
    ny = labels.shape[1]
    nz = labels.shape[2]
    s = diag[i, j, k] * x[i, j, k]
    if i > 0:
        s -= _neighbour(labels, x, i - 1, j, k, _w(wx, i, j, k, unit))
    if i < nx - 1:
        s -= _neighbour(labels, x, i + 1, j, k, _w(wx, i + 1, j, k, unit))
    if j > 0:
        s -= _neighbour(labels, x, i, j - 1, k, _w(wy, i, j, k, unit))
    if j < ny - 1:
        s -= _neighbour(labels, x, i, j + 1, k, _w(wy, i, j + 1, k, unit))
    if k > 0:
        s -= _neighbour(labels, x, i, j, k - 1, _w(wz, i, j, k, unit))
    if k < nz - 1:
        s -= _neighbour(labels, x, i, j, k + 1, _w(wz, i, j, k + 1, unit))
    return s


@wp.kernel
def apply(
    labels: wp.array3d(dtype=wp.int32),
    diag: wp.array3d(dtype=float),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    unit: int,
    inv_dx2: float,
    x: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        out[i, j, k] = 0.0
        return
    out[i, j, k] = _apply_at(labels, diag, wx, wy, wz, unit, x, i, j, k) * inv_dx2


@wp.kernel
def residual(
    labels: wp.array3d(dtype=wp.int32),
    diag: wp.array3d(dtype=float),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    unit: int,
    inv_dx2: float,
    x: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    r: wp.array3d(dtype=float),
):
    """r = b - A x on fluid cells, 0 elsewhere."""
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        r[i, j, k] = 0.0
        return
    r[i, j, k] = b[i, j, k] - _apply_at(labels, diag, wx, wy, wz, unit, x, i, j, k) * inv_dx2


@wp.kernel
def jacobi(
    labels: wp.array3d(dtype=wp.int32),
    diag: wp.array3d(dtype=float),
    wx: wp.array3d(dtype=float),
    wy: wp.array3d(dtype=float),
    wz: wp.array3d(dtype=float),
    unit: int,
    inv_dx2: float,
    omega: float,
    x: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    """One damped Jacobi sweep: out = x + ω (b - A x) / D on fluid cells, 0 elsewhere."""
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        out[i, j, k] = 0.0
        return
    ax = _apply_at(labels, diag, wx, wy, wz, unit, x, i, j, k) * inv_dx2
    out[i, j, k] = x[i, j, k] + omega * (b[i, j, k] - ax) / (diag[i, j, k] * inv_dx2)


@wp.kernel
def scale_by_inverse_diagonal(
    labels: wp.array3d(dtype=wp.int32),
    diag: wp.array3d(dtype=float),
    inv_dx2: float,
    r: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    """The Jacobi preconditioner z = D⁻¹ r, for comparison with multigrid."""
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        out[i, j, k] = 0.0
        return
    out[i, j, k] = r[i, j, k] / (diag[i, j, k] * inv_dx2)


@wp.kernel
def coarsen_labels(fine: wp.array3d(dtype=wp.int32), coarse: wp.array3d(dtype=wp.int32)):
    """McAdams et al. (2010): air if any child is air, else fluid if any is, else solid."""
    i, j, k = wp.tid()
    any_air = wp.int32(0)
    any_fluid = wp.int32(0)
    for a in range(2):
        for b in range(2):
            for c in range(2):
                fi = 2 * i + a
                fj = 2 * j + b
                fk = 2 * k + c
                if fi < fine.shape[0] and fj < fine.shape[1] and fk < fine.shape[2]:
                    label = fine[fi, fj, fk]
                    if label == AIR:
                        any_air = 1
                    if label == FLUID:
                        any_fluid = 1
    if any_air == 1:
        coarse[i, j, k] = AIR
    elif any_fluid == 1:
        coarse[i, j, k] = FLUID
    else:
        coarse[i, j, k] = SOLID


@wp.func
def _lower(f: int) -> int:
    """Lower coarse neighbour of fine cell `f` along an axis, for trilinear transfer.

    Fine cell f sits at coarse position f/2 - 1/4: between coarse f/2 - 1 and f/2
    when f is even (weights 1/4, 3/4), between (f - 1)/2 and (f + 1)/2 when odd
    (3/4, 1/4). Out-of-grid neighbours are clamped by the caller.
    """
    if f % 2 == 0:
        return f // 2 - 1
    return (f - 1) // 2


@wp.func
def _upper_weight(f: int) -> float:
    """Weight of the upper coarse neighbour: 3/4 for even fine cells, 1/4 for odd."""
    if f % 2 == 0:
        return 0.75
    return 0.25


@wp.func
def _clamp(c: int, n: int) -> int:
    return wp.clamp(c, 0, n - 1)


@wp.kernel
def prolong_add(
    fine_labels: wp.array3d(dtype=wp.int32),
    coarse: wp.array3d(dtype=float),
    fine: wp.array3d(dtype=float),
):
    """fine += P coarse (trilinear between cell centres, clamped at the edges) on fluid cells."""
    i, j, k = wp.tid()
    if fine_labels[i, j, k] != FLUID:
        return
    ci = _lower(i)
    cj = _lower(j)
    ck = _lower(k)
    ti = _upper_weight(i)
    tj = _upper_weight(j)
    tk = _upper_weight(k)
    total = wp.float32(0.0)
    for a in range(2):
        wa = 1.0 - ti
        if a == 1:
            wa = ti
        for b in range(2):
            wb = 1.0 - tj
            if b == 1:
                wb = tj
            for c in range(2):
                wc = 1.0 - tk
                if c == 1:
                    wc = tk
                ii = _clamp(ci + a, coarse.shape[0])
                jj = _clamp(cj + b, coarse.shape[1])
                kk = _clamp(ck + c, coarse.shape[2])
                total += wa * wb * wc * coarse[ii, jj, kk]
    fine[i, j, k] = fine[i, j, k] + total


@wp.kernel
def restrict(
    fine_labels: wp.array3d(dtype=wp.int32),
    fine: wp.array3d(dtype=float),
    coarse: wp.array3d(dtype=float),
):
    """coarse += Pᵀ fine / 8: the exact transpose of `prolong_add`, by scattering."""
    i, j, k = wp.tid()
    if fine_labels[i, j, k] != FLUID:
        return
    value = fine[i, j, k] * 0.125
    ci = _lower(i)
    cj = _lower(j)
    ck = _lower(k)
    ti = _upper_weight(i)
    tj = _upper_weight(j)
    tk = _upper_weight(k)
    for a in range(2):
        wa = 1.0 - ti
        if a == 1:
            wa = ti
        for b in range(2):
            wb = 1.0 - tj
            if b == 1:
                wb = tj
            for c in range(2):
                wc = 1.0 - tk
                if c == 1:
                    wc = tk
                ii = _clamp(ci + a, coarse.shape[0])
                jj = _clamp(cj + b, coarse.shape[1])
                kk = _clamp(ck + c, coarse.shape[2])
                wp.atomic_add(coarse, ii, jj, kk, wa * wb * wc * value)


@wp.kernel
def mask(labels: wp.array3d(dtype=wp.int32), x: wp.array3d(dtype=float)):
    """Zero on every cell that is not fluid."""
    i, j, k = wp.tid()
    if labels[i, j, k] != FLUID:
        x[i, j, k] = 0.0


@wp.kernel
def dot_partial(
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    stride: int,
    partials: wp.array(dtype=wp.float64),
):
    """partials[t] = Σ a[i]·b[i] over i = t, t + stride, ... in float64 (coalesced reads)."""
    t = wp.tid()
    s = wp.float64(0.0)
    n = a.shape[0]
    i = t
    while i < n:
        s += wp.float64(a[i]) * wp.float64(b[i])
        i += stride
    partials[t] = s


@wp.kernel
def sum_partials(
    partials: wp.array(dtype=wp.float64),
    group: int,
    scalars: wp.array(dtype=wp.float64),
    slot: int,
):
    """scalars[slot] += the sum of `group` consecutive partials per thread: few atomics."""
    g = wp.tid()
    s = wp.float64(0.0)
    for i in range(g * group, wp.min((g + 1) * group, partials.shape[0])):
        s += partials[i]
    wp.atomic_add(scalars, slot, s)


@wp.kernel
def clear(scalars: wp.array(dtype=wp.float64), slot: int):
    scalars[slot] = wp.float64(0.0)


@wp.kernel
def step_x_and_r(
    p: wp.array(dtype=float),
    q: wp.array(dtype=float),
    scalars: wp.array(dtype=wp.float64),
    x: wp.array(dtype=float),
    r: wp.array(dtype=float),
):
    """x += alpha p and r -= alpha q, alpha = r·z / p·Ap.

    alpha is 0 once converged, so iterations past convergence change nothing."""
    i = wp.tid()
    alpha = wp.float64(0.0)
    if scalars[PQ] > wp.float64(0.0):
        alpha = scalars[RZ] / scalars[PQ]
    x[i] = x[i] + float(alpha) * p[i]
    r[i] = r[i] - float(alpha) * q[i]


@wp.kernel
def step_p(z: wp.array(dtype=float), scalars: wp.array(dtype=wp.float64), p: wp.array(dtype=float)):
    """p = z + beta p with beta = r·z new / r·z old (0 once converged)."""
    i = wp.tid()
    beta = wp.float64(0.0)
    if scalars[RZ] > wp.float64(0.0):
        beta = scalars[RZ_NEW] / scalars[RZ]
    p[i] = z[i] + float(beta) * p[i]


@wp.kernel
def roll(scalars: wp.array(dtype=wp.float64)):
    scalars[RZ] = scalars[RZ_NEW]


@wp.kernel
def mask_finding_air(
    labels: wp.array3d(dtype=wp.int32),
    x: wp.array3d(dtype=float),
    scalars: wp.array(dtype=wp.float64),
):
    """Zero on every cell that is not fluid, and scalars[ANY_AIR] = 1 if any cell is air
    (every air cell writes the same 1; the caller clears it first)."""
    i, j, k = wp.tid()
    label = labels[i, j, k]
    if label != FLUID:
        x[i, j, k] = 0.0
    if label == AIR:
        scalars[ANY_AIR] = wp.float64(1.0)


@wp.kernel
def count_partial(
    labels: wp.array(dtype=wp.int32),
    label: int,
    stride: int,
    partials: wp.array(dtype=wp.float64),
):
    """partials[t] = how many of the cells t, t + stride, ... carry `label`."""
    t = wp.tid()
    s = wp.float64(0.0)
    i = t
    while i < labels.shape[0]:
        if labels[i] == label:
            s += wp.float64(1.0)
        i += stride
    partials[t] = s


@wp.kernel
def fluid_sum_partial(
    labels: wp.array(dtype=wp.int32),
    a: wp.array(dtype=float),
    stride: int,
    scalars: wp.array(dtype=wp.float64),
    partials: wp.array(dtype=wp.float64),
):
    """partials[t] = Σ a over the fluid cells t, t + stride, ... in float64, and FLUID_SUM
    cleared for `sum_partials` to fill (no thread of this kernel reads it)."""
    t = wp.tid()
    if t == 0:
        scalars[FLUID_SUM] = wp.float64(0.0)
    s = wp.float64(0.0)
    i = t
    while i < labels.shape[0]:
        if labels[i] == FLUID:
            s += wp.float64(a[i])
        i += stride
    partials[t] = s


@wp.kernel
def subtract_fluid_mean(
    labels: wp.array(dtype=wp.int32), scalars: wp.array(dtype=wp.float64), a: wp.array(dtype=float)
):
    """a -= FLUID_SUM / FLUID_CELLS on the fluid cells, the difference taken in float64."""
    i = wp.tid()
    if labels[i] != FLUID:
        return
    a[i] = float(wp.float64(a[i]) - scalars[FLUID_SUM] / scalars[FLUID_CELLS])

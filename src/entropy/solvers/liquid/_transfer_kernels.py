"""Warp kernels of `entropy.solvers.liquid.transfer` (APIC with quadratic B-splines).

pyright cannot type Warp's kernel maths: its stubs give vector and matrix
components union types. This file holds kernels only, so the rules below are
relaxed here and nowhere else; the host code in `transfer.py` stays fully
checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false

import warp as wp


@wp.func
def _weight(t: float, node: int) -> float:
    """Quadratic B-spline weight of node 0, 1 or 2 at offset t = g - base, t in [0.5, 1.5)."""
    if node == 0:
        return 0.5 * (1.5 - t) * (1.5 - t)
    if node == 1:
        return 0.75 - (t - 1.0) * (t - 1.0)
    return 0.5 * (t - 0.5) * (t - 0.5)


@wp.func
def _base(g: float) -> int:
    """First of the three nodes around fractional index g."""
    return int(wp.floor(g - 0.5))


@wp.func
def _inside(i: int, j: int, k: int, a: wp.array3d(dtype=float)) -> bool:
    return i >= 0 and j >= 0 and k >= 0 and i < a.shape[0] and j < a.shape[1] and k < a.shape[2]


@wp.func
def _splat(
    momentum: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    g: wp.vec3,
    offset: wp.vec3,
    p_m: wp.vec3,
    origin_m: wp.vec3,
    dx_m: float,
    mass_kg: float,
    v: float,
    c: wp.vec3,
):
    """One component: mass and APIC momentum m (v + c·(x_i - x_p)) onto its 27 nodes."""
    bi = _base(g[0])
    bj = _base(g[1])
    bk = _base(g[2])
    for a in range(3):
        wa = _weight(g[0] - float(bi), a)
        for b in range(3):
            wb = _weight(g[1] - float(bj), b)
            for d in range(3):
                i = bi + a
                j = bj + b
                k = bk + d
                if _inside(i, j, k, mass):
                    w = wa * wb * _weight(g[2] - float(bk), d)
                    x_i = origin_m + (wp.vec3(float(i), float(j), float(k)) + offset) * dx_m
                    wp.atomic_add(mass, i, j, k, w * mass_kg)
                    wp.atomic_add(momentum, i, j, k, w * mass_kg * (v + wp.dot(c, x_i - p_m)))


@wp.kernel
def particles_to_grid(
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
    origin_m: wp.vec3,
    dx_m: float,
    mass_kg: float,
    mu: wp.array3d(dtype=float),
    mv: wp.array3d(dtype=float),
    mw: wp.array3d(dtype=float),
    m_u: wp.array3d(dtype=float),
    m_v: wp.array3d(dtype=float),
    m_w: wp.array3d(dtype=float),
):
    t = wp.tid()
    p = positions_m[t]
    vel = velocities[t]
    cm = affine[t]
    g = (p - origin_m) / dx_m  # cell units
    _splat(
        mu,
        m_u,
        wp.vec3(g[0], g[1] - 0.5, g[2] - 0.5),
        wp.vec3(0.0, 0.5, 0.5),
        p,
        origin_m,
        dx_m,
        mass_kg,
        vel[0],
        cm[0],
    )
    _splat(
        mv,
        m_v,
        wp.vec3(g[0] - 0.5, g[1], g[2] - 0.5),
        wp.vec3(0.5, 0.0, 0.5),
        p,
        origin_m,
        dx_m,
        mass_kg,
        vel[1],
        cm[1],
    )
    _splat(
        mw,
        m_w,
        wp.vec3(g[0] - 0.5, g[1] - 0.5, g[2]),
        wp.vec3(0.5, 0.5, 0.0),
        p,
        origin_m,
        dx_m,
        mass_kg,
        vel[2],
        cm[2],
    )


@wp.kernel
def momentum_to_velocity(momentum: wp.array3d(dtype=float), mass: wp.array3d(dtype=float)):
    """Velocity = momentum / mass in place; 0 where a face got no mass."""
    i, j, k = wp.tid()
    m = mass[i, j, k]
    if m > 0.0:
        momentum[i, j, k] = momentum[i, j, k] / m
    else:
        momentum[i, j, k] = 0.0


@wp.func
def _gather(
    a: wp.array3d(dtype=float),
    g: wp.vec3,
    offset: wp.vec3,
    p_m: wp.vec3,
    origin_m: wp.vec3,
    dx_m: float,
) -> wp.vec4:
    """(v, c) of one component: v = Σ w v_i and c = (4/Δx²) Σ w v_i (x_i - x_p)."""
    bi = _base(g[0])
    bj = _base(g[1])
    bk = _base(g[2])
    v = wp.float32(0.0)
    c = wp.vec3(0.0, 0.0, 0.0)
    for a_ in range(3):
        wa = _weight(g[0] - float(bi), a_)
        for b in range(3):
            wb = _weight(g[1] - float(bj), b)
            for d in range(3):
                i = bi + a_
                j = bj + b
                k = bk + d
                if _inside(i, j, k, a):
                    w = wa * wb * _weight(g[2] - float(bk), d)
                    x_i = origin_m + (wp.vec3(float(i), float(j), float(k)) + offset) * dx_m
                    value = a[i, j, k]
                    v += w * value
                    c += w * value * (x_i - p_m)
    c = c * (4.0 / (dx_m * dx_m))
    return wp.vec4(v, c[0], c[1], c[2])


@wp.kernel
def grid_to_particles(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    origin_m: wp.vec3,
    dx_m: float,
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
):
    t = wp.tid()
    p = positions_m[t]
    g = (p - origin_m) / dx_m
    gu = _gather(
        u, wp.vec3(g[0], g[1] - 0.5, g[2] - 0.5), wp.vec3(0.0, 0.5, 0.5), p, origin_m, dx_m
    )
    gv = _gather(
        v, wp.vec3(g[0] - 0.5, g[1], g[2] - 0.5), wp.vec3(0.5, 0.0, 0.5), p, origin_m, dx_m
    )
    gw = _gather(
        w, wp.vec3(g[0] - 0.5, g[1] - 0.5, g[2]), wp.vec3(0.5, 0.5, 0.0), p, origin_m, dx_m
    )
    velocities[t] = wp.vec3(gu[0], gv[0], gw[0])
    affine[t] = wp.mat33(gu[1], gu[2], gu[3], gv[1], gv[2], gv[3], gw[1], gw[2], gw[3])

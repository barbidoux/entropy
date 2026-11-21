"""Warp kernels of `entropy.solvers.liquid.transfer` (APIC with quadratic B-splines).

pyright cannot type Warp's kernel maths: its stubs give vector and matrix
components union types. This file holds kernels only, so the rules below are
relaxed here and nowhere else; the host code in `transfer.py` stays fully
checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false

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
def _clamped(g: float, n: int) -> float:
    """g kept within two nodes of the samples before any int conversion, so huge values
    and infinities give the same nodes on CPU and CUDA (as `entropy.fields.mac`)."""
    return wp.clamp(g, -1.0, float(n))


@wp.func
def _base(g: float) -> int:
    """First of the three nodes around fractional index g (already clamped)."""
    return int(wp.floor(g - 0.5))


@wp.func
def _node(i: int, n: int) -> int:
    """The sample a node reads or writes: nodes outside the grid take the nearest one
    (ADR 0015: values clamp to the nearest samples outside the grid)."""
    return wp.clamp(i, 0, n - 1)


@wp.func
def _splat(
    momentum: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    g_in: wp.vec3,
    dx_m: float,
    mass_kg: float,
    v: float,
    c: wp.vec3,
):
    """One component: mass and APIC momentum m (v + c·(x_i - x_p)) onto its 27 nodes.

    Nodes outside the grid deposit on the nearest sample (no mass is lost); their
    affine term uses their own position, so the momentum is what they carry.
    """
    g = wp.vec3(
        _clamped(g_in[0], mass.shape[0]),
        _clamped(g_in[1], mass.shape[1]),
        _clamped(g_in[2], mass.shape[2]),
    )
    bi = _base(g[0])
    bj = _base(g[1])
    bk = _base(g[2])
    for a in range(3):
        wa = _weight(g[0] - float(bi), a)
        for b in range(3):
            wb = _weight(g[1] - float(bj), b)
            for d in range(3):
                w = wa * wb * _weight(g[2] - float(bk), d)
                # Node minus particle, in metres (g_in already holds the staggering).
                rel = (wp.vec3(float(bi + a), float(bj + b), float(bk + d)) - g_in) * dx_m
                i = _node(bi + a, mass.shape[0])
                j = _node(bj + b, mass.shape[1])
                k = _node(bk + d, mass.shape[2])
                wp.atomic_add(mass, i, j, k, w * mass_kg)
                wp.atomic_add(momentum, i, j, k, w * mass_kg * (v + wp.dot(c, rel)))


@wp.kernel
def particles_to_grid(
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
    origin_m: wp.vec3,
    dx_m: float,
    mass_kg: float,
    momentum_u: wp.array3d(dtype=float),
    momentum_v: wp.array3d(dtype=float),
    momentum_w: wp.array3d(dtype=float),
    mass_u: wp.array3d(dtype=float),
    mass_v: wp.array3d(dtype=float),
    mass_w: wp.array3d(dtype=float),
):
    t = wp.tid()
    vel = velocities[t]
    cm = affine[t]
    g = (positions_m[t] - origin_m) / dx_m  # cell units
    # A NaN position: the clamped g keeps the weights finite, and NaN reaches the
    # momentum through the node-particle offset, so the face velocities become NaN.
    _splat(momentum_u, mass_u, wp.vec3(g[0], g[1] - 0.5, g[2] - 0.5), dx_m, mass_kg, vel[0], cm[0])
    _splat(momentum_v, mass_v, wp.vec3(g[0] - 0.5, g[1], g[2] - 0.5), dx_m, mass_kg, vel[1], cm[1])
    _splat(momentum_w, mass_w, wp.vec3(g[0] - 0.5, g[1] - 0.5, g[2]), dx_m, mass_kg, vel[2], cm[2])


@wp.kernel
def momentum_to_velocity(momentum: wp.array3d(dtype=float), mass: wp.array3d(dtype=float)):
    """Velocity = momentum / mass in place; 0 where a face got no mass.

    Dividing by a tiny mass is safe: momentum and mass carry the same weights."""
    i, j, k = wp.tid()
    m = mass[i, j, k]
    if m > 0.0:
        momentum[i, j, k] = momentum[i, j, k] / m
    else:
        momentum[i, j, k] = 0.0


@wp.func
def _gather(a: wp.array3d(dtype=float), g_in: wp.vec3, dx_m: float) -> wp.vec4:
    """(v, c) of one component: v = Σ w v_i and c = (4/Δx²) Σ w v_i (x_i - x_p).

    Nodes outside the grid read the nearest sample at their own position, so a
    uniform field gives v = U and c = 0 right up to the walls."""
    g = wp.vec3(
        _clamped(g_in[0], a.shape[0]),
        _clamped(g_in[1], a.shape[1]),
        _clamped(g_in[2], a.shape[2]),
    )
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
                w = wa * wb * _weight(g[2] - float(bk), d)
                rel = (wp.vec3(float(bi + a_), float(bj + b), float(bk + d)) - g_in) * dx_m
                value = a[
                    _node(bi + a_, a.shape[0]), _node(bj + b, a.shape[1]), _node(bk + d, a.shape[2])
                ]
                v += w * value
                c += w * value * rel
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
    if wp.isnan(p[0]) or wp.isnan(p[1]) or wp.isnan(p[2]):
        nan = p[0] + p[1] + p[2]  # NaN: the NaN guard must see this particle
        velocities[t] = wp.vec3(nan, nan, nan)
        affine[t] = wp.mat33(nan, nan, nan, nan, nan, nan, nan, nan, nan)
        return
    g = (p - origin_m) / dx_m
    gu = _gather(u, wp.vec3(g[0], g[1] - 0.5, g[2] - 0.5), dx_m)
    gv = _gather(v, wp.vec3(g[0] - 0.5, g[1], g[2] - 0.5), dx_m)
    gw = _gather(w, wp.vec3(g[0] - 0.5, g[1] - 0.5, g[2]), dx_m)
    velocities[t] = wp.vec3(gu[0], gv[0], gw[0])
    affine[t] = wp.mat33(gu[1], gu[2], gu[3], gv[1], gv[2], gv[3], gw[1], gw[2], gw[3])

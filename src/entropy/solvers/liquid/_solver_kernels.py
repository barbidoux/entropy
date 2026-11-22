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
    """x += dt v, then kept inside the grid: its walls are the tank's walls (ADR 0017).
    A non-finite position stays as it is, so the NaN guard sees it (a clamp would hide it)."""
    t = wp.tid()
    p = positions_m[t] + velocities[t] * dt_s
    if wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]):
        p = wp.vec3(
            wp.clamp(p[0], low_m[0], high_m[0]),
            wp.clamp(p[1], low_m[1], high_m[1]),
            wp.clamp(p[2], low_m[2], high_m[2]),
        )
    positions_m[t] = p


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
def copy_totals(totals: wp.array(dtype=wp.float64), out: wp.array(dtype=float)):
    """out[2:5] = the last substep's water volume (cells), particles in the water and
    fluid cells."""
    out[2] = float(totals[0])
    out[3] = float(totals[1])
    out[4] = float(totals[2])


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


@wp.func
def _cell_mass(
    mass_u: wp.array3d(dtype=float),
    mass_v: wp.array3d(dtype=float),
    mass_w: wp.array3d(dtype=float),
    i: int,
    j: int,
    k: int,
) -> float:
    """Mean of the cell's six face masses (kg)."""
    total = mass_u[i, j, k] + mass_u[i + 1, j, k] + mass_v[i, j, k] + mass_v[i, j + 1, k]
    total = total + mass_w[i, j, k] + mass_w[i, j, k + 1]
    return total / 6.0


@wp.kernel
def cell_mass(
    mass_u: wp.array3d(dtype=float),
    mass_v: wp.array3d(dtype=float),
    mass_w: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    out[i, j, k] = _cell_mass(mass_u, mass_v, mass_w, i, j, k)


@wp.kernel
def density_and_level_set(
    mass_u: wp.array3d(dtype=float),
    mass_v: wp.array3d(dtype=float),
    mass_w: wp.array3d(dtype=float),
    rest_mass: wp.array3d(dtype=float),
    surface_fill: float,
    phi_per_fill_m: float,
    density: wp.array3d(dtype=float),
    phi_m: wp.array3d(dtype=float),
):
    """density = cell mass over the mass of water at rest there (1 inside still water);
    phi = (surface_fill - density) * phi_per_fill_m, negative in the water (ADR 0018)."""
    i, j, k = wp.tid()
    rest = rest_mass[i, j, k]
    d = 0.0
    if rest > 0.0:
        d = _cell_mass(mass_u, mass_v, mass_w, i, j, k) / rest
    density[i, j, k] = d
    phi_m[i, j, k] = (surface_fill - d) * phi_per_fill_m


SLOTS = wp.constant(1024)
"""Sums are spread over this many slots, then added up: one address for every thread
would make them wait on each other."""


@wp.kernel
def fluid_volume(
    phi_m: wp.array3d(dtype=float),
    labels: wp.array3d(dtype=wp.int32),
    dx_m: float,
    slots: wp.array(dtype=wp.float64),
    cell_slots: wp.array(dtype=wp.float64),
):
    """The water's volume in cells: each cell counts clamp(1/2 - phi/dx, 0, 1), its
    share below the surface when phi is the distance to it (ADR 0018). `cell_slots`
    counts the fluid cells, for the HUD."""
    i, j, k = wp.tid()
    if labels[i, j, k] == SOLID:
        return
    slot = ((i * labels.shape[1] + j) * labels.shape[2] + k) % SLOTS
    share = wp.clamp(0.5 - phi_m[i, j, k] / dx_m, 0.0, 1.0)
    if share > 0.0:
        wp.atomic_add(slots, slot, wp.float64(share))
    if labels[i, j, k] == FLUID:
        wp.atomic_add(cell_slots, slot, wp.float64(1.0))


@wp.kernel
def count_inside(phi_at: wp.array(dtype=float), slots: wp.array(dtype=wp.float64)):
    """How many particles lie inside the water (the level set negative where they are)."""
    t = wp.tid()
    if phi_at[t] < 0.0:
        wp.atomic_add(slots, t % SLOTS, wp.float64(1.0))


@wp.kernel
def volume_error(
    volume_slots: wp.array(dtype=wp.float64),
    inside_slots: wp.array(dtype=wp.float64),
    cell_slots: wp.array(dtype=wp.float64),
    particles_per_cell: float,
    dead_band: float,
    integral_limit_s: float,
    dt_s: float,
    error: wp.array(dtype=wp.float64),
    totals: wp.array(dtype=wp.float64),
):
    """One thread. Adds up the slots: totals = [V (cells), particles inside the water].
    e = (V_target - V) / V_target, with V_target the volume of the particles inside the
    water. error[0] is e less the dead band (0 within it); error[1] adds error[0] dt,
    clamped to the limit (no wind-up against a bias of the measure smaller than the
    dead band)."""
    volume = wp.float64(0.0)
    inside = wp.float64(0.0)
    cells = wp.float64(0.0)
    for s in range(SLOTS):
        volume = volume + volume_slots[s]
        inside = inside + inside_slots[s]
        cells = cells + cell_slots[s]
    totals[0] = volume
    totals[1] = inside
    totals[2] = cells
    volume_cells = volume
    target = inside / wp.float64(particles_per_cell)
    e = wp.float64(0.0)
    if target > wp.float64(0.0):
        e = (target - volume_cells) / target
    band = wp.float64(dead_band)
    outside = wp.float64(0.0)
    if e > band:
        outside = e - band
    elif e < -band:
        outside = e + band
    error[0] = outside
    limit = wp.float64(integral_limit_s)
    error[1] = wp.clamp(error[1] + outside * wp.float64(dt_s), -limit, limit)


@wp.kernel
def volume_correction(
    density: wp.array3d(dtype=float),
    labels: wp.array3d(dtype=wp.int32),
    rate_per_s: float,
    error: wp.array(dtype=wp.float64),
    proportional_per_s: float,
    integral_per_s2: float,
    target: wp.array3d(dtype=float),
):
    """Target divergence of each fluid cell (ADR 0018), the sum of two terms:

    - volume control (Kim et al. 2007), proportional and integral: the same in every
      fluid cell, from the level set's volume error (`volume_error`);
    - local density: rate * (density - 1) in fluid cells with no air within two cells,
      so crowded cells expand and sparse ones contract. A cell's faces gather particles
      up to 2 cells away (quadratic B-splines), so nearer the surface the density reads
      low for lack of water: those cells are left alone. Walls and solids are allowed:
      the density is measured against the rest mass there."""
    i, j, k = wp.tid()
    target[i, j, k] = 0.0
    if labels[i, j, k] != FLUID:
        return
    e = float(error[0])
    accumulated = float(error[1])
    target[i, j, k] = proportional_per_s * e + integral_per_s2 * accumulated
    nx = labels.shape[0]
    ny = labels.shape[1]
    nz = labels.shape[2]
    for a in range(-2, 3):
        for b in range(-2, 3):
            for c in range(-2, 3):
                ii = i + a
                jj = j + b
                kk = k + c
                inside = ii >= 0 and jj >= 0 and kk >= 0 and ii < nx and jj < ny and kk < nz
                if inside and labels[ii, jj, kk] == AIR:
                    return
    target[i, j, k] = target[i, j, k] + rate_per_s * (density[i, j, k] - 1.0)


@wp.kernel
def outside_sphere(
    positions_m: wp.array(dtype=wp.vec3),
    centre_m: wp.vec3,
    radius_sq_m2: float,
    keep: wp.array(dtype=wp.int32),
):
    """keep = 1 for particles farther than the radius from the centre, 0 within it."""
    t = wp.tid()
    keep[t] = wp.where(wp.length_sq(positions_m[t] - centre_m) > radius_sq_m2, 1, 0)


@wp.kernel
def compact(
    keep: wp.array(dtype=wp.int32),
    index: wp.array(dtype=wp.int32),
    positions_m: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    affine: wp.array(dtype=wp.mat33),
    positions_out: wp.array(dtype=wp.vec3),
    velocities_out: wp.array(dtype=wp.vec3),
    affine_out: wp.array(dtype=wp.mat33),
):
    """Kept particles move to their exclusive prefix-sum index, in their order."""
    t = wp.tid()
    if keep[t] == 1:
        i = index[t]
        positions_out[i] = positions_m[t]
        velocities_out[i] = velocities[t]
        affine_out[i] = affine[t]

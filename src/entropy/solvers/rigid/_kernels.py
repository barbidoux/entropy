"""Warp kernels of the rigid bodies (SPEC 5.4).

pyright cannot type Warp's kernel maths (its stubs do not cover vectors and quaternions
in kernels), so those checks are off in this file, as in the liquid's kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false, reportReturnType=false

import warp as wp

from entropy.fields import _solid_kernels as _solid

GYROSCOPIC_ITERATIONS = 4
"""Fixed-point iterations of the implicit midpoint rule for the gyroscopic torque."""


@wp.func
def _unit(q: wp.quat) -> wp.quat:
    """q scaled to unit length. Not Warp's normalize, which turns a zero or NaN quaternion
    into the identity: a NaN must stay NaN for the guard to see it."""
    return q * (1.0 / wp.sqrt(wp.dot(q, q)))


@wp.func
def _rotation_quat(omega_rad_per_s: wp.vec3, dt_s: float) -> wp.quat:
    """The rotation by |omega| dt about omega's direction (the exact exponential)."""
    speed = wp.length(omega_rad_per_s)
    if speed * dt_s < 1.0e-12:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(omega_rad_per_s / speed, speed * dt_s)


@wp.func
def _gyroscopic(w_body: wp.vec3, inertia: wp.vec3, dt_s: float) -> wp.vec3:
    """The body-frame angular velocity after `dt_s` of torque-free rotation, by the
    implicit midpoint rule, I(w1 - w0)/dt = -wm x (I wm) with wm = (w0 + w1)/2, solved
    by fixed-point iteration. It keeps |L| and the energy where explicit Euler gains
    both every substep (#98's review). It holds to about 100 rad/s at 1/480 s
    (|w| dt about 0.2); faster spins lose energy, never gain it (a 2 m plank at 200 rad/s
    lost 26 % in 10 s), which is safe for a game."""
    w_next = w_body
    for _k in range(GYROSCOPIC_ITERATIONS):
        w_mid = 0.5 * (w_body + w_next)
        w_next = w_body - dt_s * wp.cw_div(wp.cross(w_mid, wp.cw_mul(inertia, w_mid)), inertia)
    return w_next


@wp.kernel
def predict(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    gravity_m_per_s2: wp.vec3,
    dt_s: float,
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    rotated: wp.array(dtype=wp.vec3),
):
    """The first half of an XPBD substep (Müller et al. 2020): move every body with
    gravity and the gyroscopic torque, and keep the motion as increments: a displacement,
    a turn (the rotation, a quaternion) and the rotation vector it was made of.
    Constraints add their corrections to all three; `recover` turns them into the
    velocities, not the difference of two nearby positions or orientations, which
    float32 does not keep exact (ADR 0030)."""
    i = wp.tid()
    q0 = orientation[i]
    v = velocity_m_per_s[i] + gravity_m_per_s2 * dt_s
    # The gyroscopic torque, in the body's frame where the inertia is diagonal. Only its
    # change of the angular velocity goes through the rotations: a round trip of the
    # velocity itself would scale it by |q|^4, which float32 leaves 1e-7 off 1.
    w = angular_velocity_rad_per_s[i]
    w_body = wp.quat_rotate_inv(q0, w)
    w = w + wp.quat_rotate(q0, _gyroscopic(w_body, inertia_kg_m2[i], dt_s) - w_body)
    displacement = v * dt_s
    rotation = _rotation_quat(w, dt_s)
    position_m[i] = position_m[i] + displacement
    orientation[i] = _unit(rotation * q0)
    displacement_m[i] = displacement
    turn[i] = rotation
    rotated[i] = w * dt_s


@wp.kernel
def recover(
    displacement_m: wp.array(dtype=wp.vec3),
    rotated: wp.array(dtype=wp.vec3),
    dt_s: float,
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
):
    """The second half of a substep: the velocities that made this substep's motion. The
    angular one is the sum of the rotation vectors over h, not the logarithm of the turn:
    a correction's rotation I⁻¹(r x p) then changes I ω by exactly r x p / h, so contacts
    keep the angular momentum as they keep the linear one (#101's review)."""
    i = wp.tid()
    velocity_m_per_s[i] = displacement_m[i] / dt_s
    angular_velocity_rad_per_s[i] = rotated[i] / dt_s


# Contacts (ADR 0031) ----------------------------------------------------------------

BOX_SAMPLES = wp.constant(26)
"""A box's sample points: its 8 corners, 12 edge midpoints and 6 face centres. Two
equal boxes face to face put every corner and edge midpoint on the other's edges, at
distance 0: only the face centres sink in."""
CAPSULE_MAX_SAMPLES = wp.constant(25)
"""A capsule's sample spheres lie along its axis at most a radius apart, an odd number
of them so that one is at its middle, so nothing thinner than about its radius slips
between them, up to 25 spheres: a capsule longer than 24 radii has wider gaps (#101's
review)."""
EDGE_POINTS = wp.constant(16)
"""The most points tested along one edge of a static box against a body."""
TOUCH_M = wp.constant(0.005)
"""A sample within 5 mm of a surface counts towards its patch's width (`_wide`), sunk in
or not: lifted by its bottom face's centre, a box resting flat has one contact only,
yet the whole face holds it (ADR 0032)."""
STATIC_PATCHES = wp.constant(3)
"""Room for a body's patches with the static solids: a floor and two walls."""
PATCH_COS = wp.constant(0.9)
"""A body's contacts with the static solids share a patch when their normals lie within
about 25° of the patch's first one: a floor and a wall make two patches."""


@wp.func
def _samples(shape: int, size: wp.vec3) -> int:
    if shape == 0:
        return BOX_SAMPLES
    if shape == 1:
        return 1
    gaps = int(wp.ceil(2.0 * size[1] / size[0]))
    gaps += gaps % 2  # an even number of gaps: a sample at the middle (`_order`)
    return wp.clamp(gaps + 1, 1, CAPSULE_MAX_SAMPLES)


@wp.func
def _sample(shape: int, size: wp.vec3, k: int, signs: wp.array(dtype=wp.vec3)) -> wp.vec4:
    """(x, y, z, radius) of a body's sample k, in its frame: a sphere about the point
    that the surface wraps (radius 0 for a box's corners and edges)."""
    if shape == 0:
        c = wp.cw_mul(signs[k], size)
        return wp.vec4(c[0], c[1], c[2], 0.0)
    if shape == 1:
        return wp.vec4(0.0, 0.0, 0.0, size[0])
    n = _samples(shape, size)
    z = 0.0
    if n > 1:
        z = size[1] * (2.0 * float(k) / float(n - 1) - 1.0)
    return wp.vec4(0.0, 0.0, z, size[0])


@wp.func
def _order(shape: int, count: int, step: int) -> int:
    """The sample a sweep tests at its `step`-th turn, out of `count`: a box's face
    centres first and its corners last; a capsule's middle first, then outwards, one
    side then the other. The first contact of a body lying flat (or a capsule lying
    along a surface) then lifts it whole, through its centre of mass, with no turn; ends
    first would tilt it a little every substep, and a pair of parallel capsules hit side
    on came apart spinning."""
    if shape == 0:
        return count - 1 - step
    middle = (count - 1) // 2
    reach = (step + 1) // 2
    if step % 2 == 1:
        return middle + reach
    return middle - reach


@wp.func
def _body_distance(shape: int, size: wp.vec3, p: wp.vec3) -> wp.vec4:
    """(outward normal x, y, z, signed distance) of a body's surface at a point of its
    own frame: exact for the box (Quilez), the sphere and the capsule. A point on a
    capsule's axis or at a sphere's centre gets the normal +z (any would do)."""
    if shape == 0:
        q = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2])) - size
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        d = wp.length(outside) + inside
        sign = wp.vec3(wp.sign(p[0]), wp.sign(p[1]), wp.sign(p[2]))  # Warp's sign(0) is 1
        n = wp.vec3(0.0, 0.0, 1.0)
        if wp.length(outside) > 0.0:
            n = wp.normalize(wp.cw_mul(outside, sign))
        else:
            axis = 0
            if q[1] > q[axis]:
                axis = 1
            if q[2] > q[axis]:
                axis = 2
            n = wp.vec3(0.0, 0.0, 0.0)
            n[axis] = sign[axis]
        return wp.vec4(n[0], n[1], n[2], d)
    centre = wp.vec3(0.0, 0.0, 0.0)
    if shape == 2:
        centre = wp.vec3(0.0, 0.0, wp.clamp(p[2], -size[1], size[1]))
    v = p - centre
    length = wp.length(v)
    n = wp.vec3(0.0, 0.0, 1.0)
    if length > 1.0e-9:
        n = v / length
    return wp.vec4(n[0], n[1], n[2], length - size[0])


@wp.func
def _inverse_inertia_world(q: wp.quat, inertia: wp.vec3, v: wp.vec3) -> wp.vec3:
    """I⁻¹ v in the world's frame, from the principal moments in the body's."""
    return wp.quat_rotate(q, wp.cw_div(wp.quat_rotate_inv(q, v), inertia))


# Against the static solids -------------------------------------------------------------


@wp.func
def _plane(
    c: int,
    start: wp.vec3,
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
) -> wp.vec4:
    """The c-th static plane a sample starting the substep at `start` may sink past:
    (outward normal n, offset d) of the plane n·p = d, or a zero normal when there is
    none. Each half-space is its plane. Each box gives the tangent plane of its surface
    nearest to `start`. Each open box (tank) gives six candidates: from inside its
    cavity's column, its floor's and its four walls' inner planes (the walls up to their
    top), each on its own, so a sample near an inner edge or corner meets both or all
    three (#101's review: small boxes left the tank there); from outside, the tangent
    plane of its surface nearest to `start`."""
    planes = plane_points.shape[0]
    boxes = box_centres.shape[0]
    if c < planes:
        n = plane_normals[c]
        return wp.vec4(n[0], n[1], n[2], wp.dot(plane_points[c], n))
    if c < planes + boxes:
        lo = box_centres[c - planes] - box_halves[c - planes]
        hi = box_centres[c - planes] + box_halves[c - planes]
        q = _solid._box_q(start, lo, hi)
        n = _solid._normal_from_q(q, _solid._box_s(start, lo, hi))
        surface = start - _solid._from_q(q) * n
        return wp.vec4(n[0], n[1], n[2], wp.dot(surface, n))
    t = (c - planes - boxes) // 6
    f = (c - planes - boxes) % 6
    lo = tank_lows[t]
    hi = tank_highs[t]
    wall = tank_walls[t]
    inside = start[0] >= lo[0] and start[0] <= hi[0] and start[1] >= lo[1] and start[1] <= hi[1]
    inside = inside and start[2] >= lo[2] - 0.5 * wall
    none = wp.vec4(0.0, 0.0, 0.0, 0.0)
    if inside:
        if f == 0:
            return wp.vec4(0.0, 0.0, 1.0, lo[2])
        if f == 5 or start[2] > hi[2]:
            return none
        if f == 1:
            return wp.vec4(1.0, 0.0, 0.0, lo[0])
        if f == 2:
            return wp.vec4(-1.0, 0.0, 0.0, -hi[0])
        if f == 3:
            return wp.vec4(0.0, 1.0, 0.0, lo[1])
        return wp.vec4(0.0, -1.0, 0.0, -hi[1])
    if f != 5:
        return none
    olo = _solid._outer_lo(lo, wall)
    ohi = _solid._outer_hi(hi, wall)
    outer = _solid._box_q(start, olo, ohi)
    cavity = _solid._cavity_q(start, lo, hi)
    d_outer = _solid._from_q(outer)
    n = -_solid._normal_from_q(cavity, _solid._cavity_s(start, lo, hi))
    d = wp.max(d_outer, -_solid._from_q(cavity))
    if d_outer >= -_solid._from_q(cavity):
        n = _solid._normal_from_q(outer, _solid._box_s(start, olo, ohi))
    surface = start - d * n
    return wp.vec4(n[0], n[1], n[2], wp.dot(surface, n))


@wp.func
def _lift(
    x: wp.vec3,
    spin: wp.vec3,
    q: wp.quat,
    mass: float,
    inertia: wp.vec3,
    point: wp.vec3,
    n: wp.vec3,
    depth: float,
):
    """One correction of a body against the static solids: the XPBD impulse (kg·m) that
    closes `depth` along `n` at `point`, compliance 0. Returns the body's new place, its
    new small rotation since the sweep began, and the impulse."""
    lever = wp.cross(point - x, n)
    weight = 1.0 / mass + wp.dot(lever, _inverse_inertia_world(q, inertia, lever))
    impulse = depth / weight
    return (
        x + impulse / mass * n,
        spin + _inverse_inertia_world(q, inertia, impulse * lever),
        impulse,
    )


@wp.func
def _slot(n: wp.vec3, used: int, n0: wp.vec3, n1: wp.vec3, n2: wp.vec3):
    """The static patch a contact of normal `n` joins: the first whose first normal lies
    within about 25° of it, else a new one while there is room, else the closest.
    Returns the slot and the patches' count and first normals."""
    if used > 0 and wp.dot(n, n0) > PATCH_COS:
        return 0, used, n0, n1, n2
    if used > 1 and wp.dot(n, n1) > PATCH_COS:
        return 1, used, n0, n1, n2
    if used > 2 and wp.dot(n, n2) > PATCH_COS:
        return 2, used, n0, n1, n2
    if used == 0:
        return 0, 1, n, n1, n2
    if used == 1:
        return 1, 2, n0, n, n2
    if used == 2:
        return 2, 3, n0, n1, n
    slot = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    best = wp.dot(n, n0)
    if wp.dot(n, n1) > best:
        slot = 1
        best = wp.dot(n, n1)
    if wp.dot(n, n2) > best:
        slot = 2
    return slot, used, n0, n1, n2


@wp.func
def _match(n: wp.vec3, used: int, n0: wp.vec3, n1: wp.vec3, n2: wp.vec3) -> int:
    """The static patch whose first normal lies within about 25° of `n`, or -1."""
    if used > 0 and wp.dot(n, n0) > PATCH_COS:
        return 0
    if used > 1 and wp.dot(n, n1) > PATCH_COS:
        return 1
    if used > 2 and wp.dot(n, n2) > PATCH_COS:
        return 2
    return -1


@wp.func
def _mean(a: float, b: float) -> float:
    """A pair's friction: the geometric mean of the two coefficients (ADR 0032)."""
    return wp.sqrt(a * b)


@wp.func
def _spread(second: wp.mat33, count: float, mean: wp.vec3) -> wp.mat33:
    """A patch's spread: the covariance of its touching points (m²), from their second
    moment `second` about a reference point, their number and their mean's offset from
    it."""
    return second / count - wp.outer(mean, mean)


@wp.func
def _wide(spread: wp.mat33, direction: wp.vec3, height_m: float, mu: float) -> bool:
    """Whether a patch is wide enough along `direction` that friction there cannot tip
    the body above it: its points reach √3 sigma along it (the half length of an even
    spread), at least μ times the height of that body's centre above the patch, so the
    normal forces can shift forward and hold the friction's torque (ADR 0032)."""
    reach2 = 3.0 * wp.dot(direction, spread * direction)
    return reach2 >= (mu * height_m) * (mu * height_m)


@wp.func
def _torsion_radius(spread: wp.mat33, n: wp.vec3) -> float:
    """The root-mean-square distance of a patch's points from their mean, across the
    normal (m): the lever of its friction against turning about the normal."""
    across = spread[0, 0] + spread[1, 1] + spread[2, 2] - wp.dot(n, spread * n)
    return wp.sqrt(wp.max(across, 0.0))


@wp.kernel
def solve_static(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    rotated: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    bounds_m: wp.array(dtype=float),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    signs: wp.array(dtype=wp.vec3),
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
    static_contacts: wp.array(dtype=wp.int32),
    patch_impulse: wp.array(dtype=float),
    patch_moment: wp.array(dtype=wp.vec3),
    patch_normal: wp.array(dtype=wp.vec3),
    patch_spread: wp.array(dtype=wp.mat33),
    patch_touch: wp.array(dtype=wp.vec4),
    static_friction: wp.array(dtype=float),
    solids_static_friction: float,
):
    """Body i against the static solids, by its own thread: one Gauss-Seidel sweep, each
    correction applied before the next is tested, compliance 0.

    First its sample spheres (`_order`: a box's face centres first, a capsule's middle
    first) against every static plane `_plane` gives from where the sample started the
    substep: one sinks in when its place now passes the plane, so a thin wall keeps its
    inner normal however fast a body comes, and a sample meets each wall of an inner
    corner. Then the edges of the static boxes near the body, at points no farther apart
    than half the body's smallest size, against the body's exact distance: a table's
    edge under a box's face holds it, where its samples alone would leave it balanced on
    the row under its centre of mass, and it would topple off (#101's review).

    Within the sweep the body turns to first order (a point moves by spin x r) and by
    the exact rotation once at the end: composing the small rotations one by one would
    add a second-order turn about the normal, and a box at rest would slowly start to
    spin. The contacts gather into up to `STATIC_PATCHES` patches by direction (`_slot`),
    entries 3i to 3i + 2: their normal impulses (kg·m), the moments of their points and
    their normals, and the spread of the points within `TOUCH_M` (`patch_spread`), for
    the velocity pass.

    Then static friction (Müller et al. 2020, §3.5; ADR 0032), once per patch, one after
    the other: the correction that undoes the sliding of the patch's centre of pressure
    this substep, kept only when it is within μs times the patch's normal impulse. On a
    patch wide enough along the sliding (`_wide`) it moves the body without turning it:
    the normal forces would shift and hold the torque, which one point cannot show, and
    a box on a slope crept by rolling about its centre of pressure."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    x0 = x - displacement_m[i]
    q0 = wp.quat_inverse(turn[i]) * q
    origin = x
    mass = masses_kg[i]
    inertia = inertia_kg_m2[i]
    shape = shapes[i]
    size = sizes_m[i]
    spin = wp.vec3(0.0, 0.0, 0.0)
    used = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    found = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    n0 = wp.vec3(0.0, 0.0, 0.0)
    n1 = wp.vec3(0.0, 0.0, 0.0)
    n2 = wp.vec3(0.0, 0.0, 0.0)
    for s in range(STATIC_PATCHES):
        patch_impulse[STATIC_PATCHES * i + s] = 0.0
        patch_moment[STATIC_PATCHES * i + s] = wp.vec3(0.0, 0.0, 0.0)
        patch_normal[STATIC_PATCHES * i + s] = wp.vec3(0.0, 0.0, 0.0)
        patch_spread[STATIC_PATCHES * i + s] = wp.mat33()
        patch_touch[STATIC_PATCHES * i + s] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    candidates = plane_points.shape[0] + box_centres.shape[0] + 6 * tank_lows.shape[0]
    count = _samples(shape, size)
    for step in range(count):
        sample = _sample(shape, size, _order(shape, count, step), signs)
        local = wp.vec3(sample[0], sample[1], sample[2])
        start = x0 + wp.quat_rotate(q0, local)
        for c in range(candidates):
            plane = _plane(c, start, box_centres, box_halves, tank_lows, tank_highs,
                           tank_walls, plane_points, plane_normals)  # fmt: skip
            n = wp.vec3(plane[0], plane[1], plane[2])
            if wp.length(n) > 0.0:
                offset = wp.quat_rotate(q, local)
                centre = x + offset + wp.cross(spin, offset)
                depth = sample[3] + plane[3] - wp.dot(centre, n)
                if depth > -TOUCH_M:
                    point = centre - sample[3] * n
                    slot = _match(n, used, n0, n1, n2)
                    if slot < 0 and depth > 0.0:
                        slot, used, n0, n1, n2 = _slot(n, used, n0, n1, n2)
                    if slot >= 0:
                        entry = STATIC_PATCHES * i + slot
                        rel = point - origin
                        patch_touch[entry] = patch_touch[entry] + wp.vec4(
                            rel[0], rel[1], rel[2], 1.0
                        )
                        patch_spread[entry] = patch_spread[entry] + wp.outer(rel, rel)
                        if depth > 0.0:
                            x, spin, impulse = _lift(x, spin, q, mass, inertia, point, n, depth)
                            patch_impulse[entry] = patch_impulse[entry] + impulse
                            patch_moment[entry] = patch_moment[entry] + impulse * point
                            patch_normal[entry] = patch_normal[entry] + impulse * n
                            found += 1
    # The static boxes' edges against the body (spheres need none: their sample is exact).
    reach = bounds_m[i]
    spacing = 0.5 * size[0]
    if shape == 0:
        spacing = 0.5 * wp.min(size[0], wp.min(size[1], size[2]))
    if shape != 1:
        for b in range(box_centres.shape[0]):
            centre_b = box_centres[b]
            half = box_halves[b]
            for e in range(12):
                axis = e // 4
                one = (axis + 1) % 3
                two = (axis + 2) % 3
                corner = centre_b
                along = wp.vec3(0.0, 0.0, 0.0)
                along[axis] = 1.0
                corner[one] = corner[one] + (float((e % 4) % 2) * 2.0 - 1.0) * half[one]
                corner[two] = corner[two] + (float((e % 4) // 2) * 2.0 - 1.0) * half[two]
                rel = x - corner
                across = rel - wp.dot(rel, along) * along
                if wp.length(across) < reach:
                    middle = wp.dot(rel, along)
                    lo = wp.max(middle - reach, -half[axis])
                    hi = wp.min(middle + reach, half[axis])
                    if hi > lo:
                        points = wp.clamp(int(wp.ceil((hi - lo) / spacing)) + 1, 2, EDGE_POINTS)
                        for j in range(points):
                            p = corner + (lo + (hi - lo) * float(j) / float(points - 1)) * along
                            rp = p - x
                            hit = _body_distance(
                                shape, size, wp.quat_rotate_inv(q, rp - wp.cross(spin, rp))
                            )
                            if hit[3] < TOUCH_M:
                                outward = wp.quat_rotate(q, wp.vec3(hit[0], hit[1], hit[2]))
                                n = -(outward + wp.cross(spin, outward))
                                n = n / wp.length(n)
                                slot = _match(n, used, n0, n1, n2)
                                if slot < 0 and hit[3] < 0.0:
                                    slot, used, n0, n1, n2 = _slot(n, used, n0, n1, n2)
                                if slot >= 0:
                                    entry = STATIC_PATCHES * i + slot
                                    rel = p - origin
                                    patch_touch[entry] = patch_touch[entry] + wp.vec4(
                                        rel[0], rel[1], rel[2], 1.0
                                    )
                                    patch_spread[entry] = patch_spread[entry] + wp.outer(rel, rel)
                                    if hit[3] < 0.0:
                                        x, spin, impulse = _lift(
                                            x, spin, q, mass, inertia, p, n, -hit[3]
                                        )
                                        patch_impulse[entry] = patch_impulse[entry] + impulse
                                        patch_moment[entry] = patch_moment[entry] + impulse * p
                                        patch_normal[entry] = patch_normal[entry] + impulse * n
                                        found += 1
    mu = _mean(static_friction[i], solids_static_friction)
    undo = wp.quat_inverse(turn[i])
    for s in range(STATIC_PATCHES):
        entry = STATIC_PATCHES * i + s
        total = patch_impulse[entry]
        normal_sum = patch_normal[entry]
        if total > 0.0 and wp.length(normal_sum) > 0.0:
            touch = patch_touch[entry]
            spread = _spread(
                patch_spread[entry], touch[3], wp.vec3(touch[0], touch[1], touch[2]) / touch[3]
            )
            patch_spread[entry] = spread
            point = patch_moment[entry] / total
            n = wp.normalize(normal_sum)
            arm = point - x
            # Where the body's material point now at `point` was when the substep began.
            slip = point - (x0 + wp.quat_rotate(undo, arm - wp.cross(spin, arm)))
            tangential = slip - wp.dot(slip, n) * n
            sliding = wp.length(tangential)
            if mu > 0.0 and sliding > 0.0:
                direction = tangential / sliding
                lever = wp.cross(arm, direction)
                wide = _wide(spread, direction, wp.abs(wp.dot(arm, n)), mu)
                weight = 1.0 / mass
                if not wide:
                    weight = weight + wp.dot(lever, _inverse_inertia_world(q, inertia, lever))
                held = sliding / weight
                if held <= mu * total:
                    x = x - held / mass * direction
                    if not wide:
                        spin = spin - _inverse_inertia_world(q, inertia, held * lever)
    if found > 0:
        correction = _rotation_quat(spin, 1.0)
        position_m[i] = x
        orientation[i] = _unit(correction * q)
        displacement_m[i] = displacement_m[i] + (x - origin)
        turn[i] = _unit(correction * turn[i])  # the substep's whole turn, for later passes
        rotated[i] = rotated[i] + spin
        wp.atomic_add(static_contacts, 0, found)


@wp.kernel
def solve_static_velocities(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    velocity_before: wp.array(dtype=wp.vec3),
    angular_before: wp.array(dtype=wp.vec3),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    patch_impulse: wp.array(dtype=float),
    patch_moment: wp.array(dtype=wp.vec3),
    patch_normal: wp.array(dtype=wp.vec3),
    patch_spread: wp.array(dtype=wp.mat33),
    dynamic_friction: wp.array(dtype=float),
    restitution: wp.array(dtype=float),
    solids_dynamic_friction: float,
    solids_restitution: float,
    slow_m_per_s: float,
    dt_s: float,
):
    """The velocity pass of body i's static patches (Müller et al. 2020, §3.6; ADR 0031,
    0032), by its own thread, one after the other, at each patch's centre of pressure.

    Restitution: the separating speed there becomes e times the approaching speed it had
    before the substep, or the separating speed it had if it was separating; no bounce
    below `slow_m_per_s` (2|g|h), so resting bodies do not jitter. The depth corrections
    then never turn into speed. The pair's e is the larger of the body's and the
    solids'.

    Dynamic friction: an impulse against the sliding there, of at most μd times the
    patch's normal impulse in the substep (its depth corrections over h plus the
    restitution's impulse, so a contact met at its surface still rubs); on a wide patch
    (`_wide`) it slows the body without turning it. Then the friction against turning
    about the normal: at most μd times the normal impulse times the patch's torsion
    radius. The pair's μd is the geometric mean of the body's and the solids'."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    v = velocity_m_per_s[i]
    w = angular_velocity_rad_per_s[i]
    mass = masses_kg[i]
    inertia = inertia_kg_m2[i]
    mu = _mean(dynamic_friction[i], solids_dynamic_friction)
    e = wp.max(restitution[i], solids_restitution)
    changed = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    for s in range(STATIC_PATCHES):
        entry = STATIC_PATCHES * i + s
        total = patch_impulse[entry]
        normal_sum = patch_normal[entry]
        if total > 0.0 and wp.length(normal_sum) > 0.0:
            changed = 1
            point = patch_moment[entry] / total
            n = wp.normalize(normal_sum)
            arm = point - x
            approach = wp.dot(velocity_before[i] + wp.cross(angular_before[i], arm), n)
            bounce = e
            if wp.abs(approach) <= slow_m_per_s:
                bounce = 0.0
            lever = wp.cross(arm, n)
            weight = 1.0 / mass + wp.dot(lever, _inverse_inertia_world(q, inertia, lever))
            now = v + wp.cross(w, arm)
            impulse = (wp.max(-bounce * approach, approach) - wp.dot(now, n)) / weight
            v = v + impulse / mass * n
            w = w + _inverse_inertia_world(q, inertia, impulse * lever)
            pressing = wp.max(total / dt_s + impulse, 0.0)
            now = v + wp.cross(w, arm)
            tangential = now - wp.dot(now, n) * n
            sliding = wp.length(tangential)
            if mu > 0.0 and sliding > 0.0:
                direction = tangential / sliding
                lever = wp.cross(arm, direction)
                wide = _wide(patch_spread[entry], direction, wp.abs(wp.dot(arm, n)), mu)
                weight = 1.0 / mass
                if not wide:
                    weight = weight + wp.dot(lever, _inverse_inertia_world(q, inertia, lever))
                rub = wp.min(mu * pressing, sliding / weight)
                v = v - rub / mass * direction
                if not wide:
                    w = w - _inverse_inertia_world(q, inertia, rub * lever)
            twirl = wp.dot(w, n)
            radius = _torsion_radius(patch_spread[entry], n)
            if mu > 0.0 and twirl != 0.0 and radius > 0.0:
                resist = wp.dot(n, _inverse_inertia_world(q, inertia, n))
                torque = wp.min(mu * pressing * radius, wp.abs(twirl) / resist)
                w = w - wp.sign(twirl) * torque * _inverse_inertia_world(q, inertia, n)
    if changed > 0:
        velocity_m_per_s[i] = v
        angular_velocity_rad_per_s[i] = w


# Between bodies ---------------------------------------------------------------------------


# The pairs of bodies in contact in a substep (ADR 0031), one patch each: all the
# contacts of one pair. Friction and the velocity pass act once per patch, at its centre
# of pressure (ADR 0032); `upper` is the body resting on the other (or -1), for the
# wide-patch rule.
@wp.struct
class Patches:
    count: wp.array(dtype=wp.int32)
    a: wp.array(dtype=wp.int32)
    b: wp.array(dtype=wp.int32)
    impulse: wp.array(dtype=float)
    moment: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    spread: wp.array(dtype=wp.mat33)
    upper: wp.array(dtype=wp.int32)
    per_body: wp.array(dtype=wp.int32)


@wp.func
def _column(m: wp.mat33, k: int) -> wp.vec3:
    return wp.vec3(m[0, k], m[1, k], m[2, k])


@wp.func
def _box_radius(axes: wp.mat33, half: wp.vec3, direction: wp.vec3) -> float:
    """How far a box reaches along a unit `direction` from its centre (m)."""
    reach = float(0.0)  # noqa: UP018 (a mutable float for Warp)
    for k in range(3):
        reach += half[k] * wp.abs(wp.dot(_column(axes, k), direction))
    return reach


@wp.func
def _box_axis(
    xa: wp.vec3, qa: wp.quat, ha: wp.vec3, xb: wp.vec3, qb: wp.quat, hb: wp.vec3
) -> wp.vec3:
    """The separating axis of least overlap between boxes a and b, pointing from b to a:
    one of the 3 + 3 face normals or the 9 edge directions' cross products (an edge axis
    wins only when its overlap is under 95 % of the best face's, so resting boxes keep a
    face normal). All the pair's contacts share it (#101's review): the boxes' own
    distances give a corner that lies on the other's edge a sideways normal, and a box
    resting on an equal one stayed tilted at 7°."""
    ra = wp.quat_to_matrix(qa)
    rb = wp.quat_to_matrix(qb)
    d = xa - xb
    best = float(1.0e30)  # noqa: UP018 (a mutable float for Warp)
    axis = wp.vec3(0.0, 0.0, 1.0)
    for k in range(15):
        candidate = wp.vec3(0.0, 0.0, 0.0)
        if k < 3:
            candidate = _column(ra, k)
        elif k < 6:
            candidate = _column(rb, k - 3)
        else:
            candidate = wp.cross(_column(ra, (k - 6) // 3), _column(rb, (k - 6) % 3))
        length = wp.length(candidate)
        if length > 1.0e-6:
            candidate = candidate / length
            overlap = _box_radius(ra, ha, candidate) + _box_radius(rb, hb, candidate)
            overlap = overlap - wp.abs(wp.dot(d, candidate))
            if (k < 6 and overlap < best) or (k >= 6 and overlap < 0.95 * best):
                best = overlap
                axis = candidate
                if wp.dot(d, candidate) < 0.0:
                    axis = -candidate
    return axis


@wp.func
def _tested(shape_a: int, size_a: wp.vec3, shape_b: int) -> int:
    """How many of body a's samples to test against body b. None against a sphere unless
    a is one too: the sphere's own sample against a's exact distance is the exact
    contact, and a box's face points against a curved surface would push sideways and
    tip a resting sphere off. A box against a capsule tests its corners and edge
    midpoints only: the capsule's samples cover its faces."""
    if shape_b == 1 and shape_a != 1:
        return 0
    if shape_a == 0 and shape_b == 2:
        return 20
    return _samples(shape_a, size_a)


@wp.func
def _sink(
    centre: wp.vec3,
    radius: float,
    xo: wp.vec3,
    qo: wp.quat,
    spin_o: wp.vec3,
    shape_o: int,
    size_o: wp.vec3,
    axis: wp.vec3,
) -> wp.vec4:
    """(normal pushing the sample out of body o, depth) for a sample sphere at `centre`
    against body o, placed at xo, qo and turned since by the small rotation `spin_o`: the
    depth is o's exact distance, and a non-zero `axis` (two boxes) is the shared
    normal."""
    rel = centre - xo
    hit = _body_distance(shape_o, size_o, wp.quat_rotate_inv(qo, rel - wp.cross(spin_o, rel)))
    n = wp.quat_rotate(qo, wp.vec3(hit[0], hit[1], hit[2]))
    n = n + wp.cross(spin_o, n)
    n = n / wp.length(n)
    if wp.length(axis) > 0.0:
        n = axis
    return wp.vec4(n[0], n[1], n[2], radius - hit[3])


@wp.kernel
def detect(
    grid: wp.uint64,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    bounds_m: wp.array(dtype=float),
    max_bound_m: float,
    signs: wp.array(dtype=wp.vec3),
    overflows: wp.array(dtype=wp.int32),
    patches: Patches,
):
    """Body i's pairs: each neighbour j > i within reach (the hash grid) whose sample
    spheres and i's (as `_tested` says) touch gets a patch; each pair is met once. A pair
    past the room is dropped and counted."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    none = wp.vec3(0.0, 0.0, 0.0)
    query = wp.hash_grid_query(grid, x, bounds_m[i] + max_bound_m)
    j = int(0)  # noqa: UP018, RUF046 (Warp needs a mutable int)
    while wp.hash_grid_query_next(query, j):
        if j > i and wp.length(position_m[j] - x) < bounds_m[i] + bounds_m[j]:
            xj = position_m[j]
            qj = orientation[j]
            axis = wp.vec3(0.0, 0.0, 0.0)
            if shapes[i] == 0 and shapes[j] == 0:
                axis = _box_axis(x, q, sizes_m[i], xj, qj, sizes_m[j])
            touching = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
            for k in range(_tested(shapes[i], sizes_m[i], shapes[j])):
                sample = _sample(shapes[i], sizes_m[i], k, signs)
                centre = x + wp.quat_rotate(q, wp.vec3(sample[0], sample[1], sample[2]))
                if _sink(centre, sample[3], xj, qj, none, shapes[j], sizes_m[j], axis)[3] > 0.0:
                    touching = 1
            back = _tested(shapes[j], sizes_m[j], shapes[i])
            if shapes[i] == 1 and shapes[j] == 1:
                back = 0
            if touching == 0:
                for k in range(back):
                    sample = _sample(shapes[j], sizes_m[j], k, signs)
                    centre = xj + wp.quat_rotate(qj, wp.vec3(sample[0], sample[1], sample[2]))
                    if _sink(centre, sample[3], x, q, none, shapes[i], sizes_m[i], -axis)[3] > 0.0:
                        touching = 1
            if touching == 1:
                slot = wp.atomic_add(patches.count, 0, 1)
                if slot < patches.a.shape[0]:
                    patches.a[slot] = i
                    patches.b[slot] = j
                    wp.atomic_add(patches.per_body, i, 1)
                    wp.atomic_add(patches.per_body, j, 1)
                else:
                    wp.atomic_add(overflows, 0, 1)


@wp.kernel
def solve_pairs(
    patches: Patches,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    signs: wp.array(dtype=wp.vec3),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    static_friction: wp.array(dtype=float),
    gravity_m_per_s2: wp.vec3,
    contact_count: wp.array(dtype=wp.int32),
    shift_m: wp.array(dtype=wp.vec3),
    spin_rad: wp.array(dtype=wp.vec3),
):
    """One pair of bodies, by its own thread: a Gauss-Seidel sweep of a's samples against
    b, then b's against a (in `_order`, as `solve_static` does), each correction applied
    to the thread's copies of both bodies before the next sample is tested, compliance 0.
    A correction is the XPBD impulse that closes the sample's depth, given to both bodies
    with opposite signs, so momentum is kept. The pair's net corrections are then scaled
    by 1 / (the larger pair count of its two bodies), Jacobi between pairs, and added to
    `shift_m` and `spin_rad`. Within the sweep the bodies turn to first order. Solved
    contact by contact in parallel (the first version), nine boxes on one slab jittered
    and sank, and the step cost half as much again. The patch gets the pair's normal
    impulse (scaled), centre of pressure and normal, and the spread of its points within
    `TOUCH_M`.

    Then the pair's static friction (ADR 0032), as `solve_static`'s, with μs the
    geometric mean of the two. When one body rests on the other (the normal within 60°
    of the vertical) and the patch is wide (`_wide`), the upper body moves without
    turning, and the lower one takes the torque that the shift of the normal forces
    would bring: (x_lower - x_upper) x P, so both momenta are kept."""
    p = wp.tid()
    if p >= wp.min(patches.count[0], patches.a.shape[0]):
        return
    a = patches.a[p]
    b = patches.b[p]
    xa = position_m[a]
    xb = position_m[b]
    qa = orientation[a]
    qb = orientation[b]
    axis = wp.vec3(0.0, 0.0, 0.0)
    if shapes[a] == 0 and shapes[b] == 0:
        axis = _box_axis(xa, qa, sizes_m[a], xb, qb, sizes_m[b])
    place_a = xa
    place_b = xb
    spin_a = wp.vec3(0.0, 0.0, 0.0)
    spin_b = wp.vec3(0.0, 0.0, 0.0)
    total = float(0.0)  # noqa: UP018 (a mutable float for Warp)
    moment = wp.vec3(0.0, 0.0, 0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    touched = float(0.0)  # noqa: UP018 (a mutable float for Warp)
    touch_sum = wp.vec3(0.0, 0.0, 0.0)
    second = wp.mat33()
    found = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    for side in range(2):
        # The sampled body s against the other o: a against b, then b against a.
        s = a
        o = b
        xs = place_a
        xo = place_b
        ss = spin_a
        so = spin_b
        direction = axis
        sign = 1.0
        tested = _tested(shapes[a], sizes_m[a], shapes[b])
        if side == 1:
            s = b
            o = a
            xs = place_b
            xo = place_a
            ss = spin_b
            so = spin_a
            direction = -axis
            sign = -1.0
            tested = _tested(shapes[b], sizes_m[b], shapes[a])
            if shapes[a] == 1 and shapes[b] == 1:
                tested = 0
        qs = orientation[s]
        qo = orientation[o]
        ms = masses_kg[s]
        mo = masses_kg[o]
        inertia_s = inertia_kg_m2[s]
        inertia_o = inertia_kg_m2[o]
        for back in range(tested):
            k = _order(shapes[s], tested, back)
            sample = _sample(shapes[s], sizes_m[s], k, signs)
            offset = wp.quat_rotate(qs, wp.vec3(sample[0], sample[1], sample[2]))
            centre = xs + offset + wp.cross(ss, offset)
            hit = _sink(centre, sample[3], xo, qo, so, shapes[o], sizes_m[o], direction)
            if hit[3] > -TOUCH_M:
                rel = centre - sample[3] * wp.vec3(hit[0], hit[1], hit[2]) - xa
                touched += 1.0
                touch_sum = touch_sum + rel
                second = second + wp.outer(rel, rel)
            if hit[3] > 0.0:
                n = wp.vec3(hit[0], hit[1], hit[2])
                point = centre - sample[3] * n
                ts = wp.cross(point - xs, n)
                to = wp.cross(point - xo, n)
                weight = 1.0 / ms + wp.dot(ts, _inverse_inertia_world(qs, inertia_s, ts))
                weight += 1.0 / mo + wp.dot(to, _inverse_inertia_world(qo, inertia_o, to))
                impulse = hit[3] / weight
                xs = xs + impulse / ms * n
                ss = ss + _inverse_inertia_world(qs, inertia_s, impulse * ts)
                xo = xo - impulse / mo * n
                so = so - _inverse_inertia_world(qo, inertia_o, impulse * to)
                total += impulse
                moment = moment + impulse * point
                normal = normal + (sign * impulse) * n
                found += 1
        if side == 0:
            place_a = xs
            place_b = xo
            spin_a = ss
            spin_b = so
        else:
            place_b = xs
            place_a = xo
            spin_b = ss
            spin_a = so
    upper = int(-1)  # noqa: UP018, RUF046 (a mutable int for Warp)
    spread = wp.mat33()
    if found > 0:
        n = wp.normalize(normal)
        point = moment / total
        spread = _spread(second, touched, touch_sum / touched)
        g = wp.length(gravity_m_per_s2)
        if g > 0.0:
            if wp.dot(n, -gravity_m_per_s2) > 0.5 * g:
                upper = a
            elif wp.dot(n, -gravity_m_per_s2) < -0.5 * g:
                upper = b
        mu = _mean(static_friction[a], static_friction[b])
        if mu > 0.0:
            ra = point - place_a
            rb = point - place_b
            start_a = (
                xa - displacement_m[a] + wp.quat_rotate_inv(turn[a], ra - wp.cross(spin_a, ra))
            )
            start_b = (
                xb - displacement_m[b] + wp.quat_rotate_inv(turn[b], rb - wp.cross(spin_b, rb))
            )
            slip = start_b - start_a  # a's point moved by point - start_a, b's by point - start_b
            tangential = slip - wp.dot(slip, n) * n
            sliding = wp.length(tangential)
            if sliding > 0.0:
                t = tangential / sliding
                qa = orientation[a]
                qb = orientation[b]
                ma = masses_kg[a]
                mb = masses_kg[b]
                height = float(0.0)  # noqa: UP018 (a mutable float for Warp)
                if upper == a:
                    height = wp.abs(wp.dot(point - place_a, n))
                elif upper == b:
                    height = wp.abs(wp.dot(point - place_b, n))
                wide = upper >= 0 and _wide(spread, t, height, mu)
                weight = 1.0 / ma + 1.0 / mb
                if not wide:
                    la = wp.cross(ra, t)
                    lb = wp.cross(rb, t)
                    weight += wp.dot(la, _inverse_inertia_world(qa, inertia_kg_m2[a], la))
                    weight += wp.dot(lb, _inverse_inertia_world(qb, inertia_kg_m2[b], lb))
                held = sliding / weight
                if held <= mu * total:
                    push = -held * t  # on a; b gets its opposite
                    place_a = place_a + push / ma
                    place_b = place_b - push / mb
                    if not wide:
                        spin_a = spin_a + _inverse_inertia_world(
                            qa, inertia_kg_m2[a], wp.cross(ra, push)
                        )
                        spin_b = spin_b - _inverse_inertia_world(
                            qb, inertia_kg_m2[b], wp.cross(rb, push)
                        )
                    elif upper == a:
                        spin_b = spin_b + _inverse_inertia_world(
                            qb, inertia_kg_m2[b], wp.cross(place_b - place_a, push)
                        )
                    else:
                        spin_a = spin_a + _inverse_inertia_world(
                            qa, inertia_kg_m2[a], wp.cross(place_a - place_b, -push)
                        )
    scale = 1.0 / float(wp.max(patches.per_body[a], patches.per_body[b]))
    wp.atomic_add(shift_m, a, scale * (place_a - xa))
    wp.atomic_add(spin_rad, a, scale * spin_a)
    wp.atomic_add(shift_m, b, scale * (place_b - xb))
    wp.atomic_add(spin_rad, b, scale * spin_b)
    patches.impulse[p] = scale * total
    patches.moment[p] = scale * moment
    patches.normal[p] = normal
    patches.spread[p] = spread
    patches.upper[p] = upper
    wp.atomic_add(contact_count, 0, found)


@wp.func
def _weight(
    a: int,
    b: int,
    point: wp.vec3,
    direction: wp.vec3,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
) -> float:
    """The pair's generalised inverse mass along `direction` at `point` (1/kg)."""
    ta = wp.cross(point - position_m[a], direction)
    tb = wp.cross(point - position_m[b], direction)
    wa = 1.0 / masses_kg[a] + wp.dot(
        ta, _inverse_inertia_world(orientation[a], inertia_kg_m2[a], ta)
    )
    wb = 1.0 / masses_kg[b] + wp.dot(
        tb, _inverse_inertia_world(orientation[b], inertia_kg_m2[b], tb)
    )
    return wa + wb


@wp.func
def _push(
    a: int,
    b: int,
    point: wp.vec3,
    impulse: wp.vec3,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    linear: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
):
    """Add `impulse` at `point` to body a and its opposite to body b: impulse / m to
    `linear`, I⁻¹ (r x impulse) to `angular`. A position correction (kg·m) gives a
    displacement and a small rotation vector; a velocity one (kg·m/s), velocities."""
    wp.atomic_add(linear, a, impulse / masses_kg[a])
    torque = wp.cross(point - position_m[a], impulse)
    wp.atomic_add(angular, a, _inverse_inertia_world(orientation[a], inertia_kg_m2[a], torque))
    wp.atomic_sub(linear, b, impulse / masses_kg[b])
    torque = wp.cross(point - position_m[b], impulse)
    wp.atomic_sub(angular, b, _inverse_inertia_world(orientation[b], inertia_kg_m2[b], torque))


@wp.kernel
def apply_corrections(
    shift_m: wp.array(dtype=wp.vec3),
    spin_rad: wp.array(dtype=wp.vec3),
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    rotated: wp.array(dtype=wp.vec3),
):
    """Move every body by its corrections, into its position and its increments (the
    rotation by the exact exponential of the rotation vector), then clear them."""
    i = wp.tid()
    shift = shift_m[i]
    position_m[i] = position_m[i] + shift
    displacement_m[i] = displacement_m[i] + shift
    s = spin_rad[i]
    correction = _rotation_quat(s, 1.0)
    orientation[i] = _unit(correction * orientation[i])
    turn[i] = _unit(correction * turn[i])
    rotated[i] = rotated[i] + s
    shift_m[i] = wp.vec3(0.0, 0.0, 0.0)
    spin_rad[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _point_velocity(
    body: int,
    point: wp.vec3,
    position_m: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    angular: wp.array(dtype=wp.vec3),
) -> wp.vec3:
    return velocity[body] + wp.cross(angular[body], point - position_m[body])


@wp.kernel
def solve_velocities(
    patches: Patches,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    velocity_before: wp.array(dtype=wp.vec3),
    angular_before: wp.array(dtype=wp.vec3),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    dynamic_friction: wp.array(dtype=float),
    restitution: wp.array(dtype=float),
    slow_m_per_s: float,
    dt_s: float,
    kick_m_per_s: wp.array(dtype=wp.vec3),
    twist_rad_per_s: wp.array(dtype=wp.vec3),
):
    """The velocity pass of the pairs' patches, in parallel, at each one's centre of
    pressure: restitution, dynamic friction and the friction against turning about the
    normal as `solve_static_velocities`, with the larger e of the two bodies and the
    geometric mean of their μd, scaled by 1 / (the larger pair count of the two) and
    given to both alike, so both momenta are kept. The friction's bound is not scaled:
    it is the patch's own normal impulse. On a wide patch with a body on top, the upper
    body slows without turning and the lower one takes the torque (x_lower - x_upper) x
    P, as in `solve_pairs`."""
    p = wp.tid()
    if p >= wp.min(patches.count[0], patches.a.shape[0]):
        return
    total = patches.impulse[p]
    normal_sum = patches.normal[p]
    if total <= 0.0 or wp.length(normal_sum) <= 0.0:
        return
    point = patches.moment[p] / total
    n = wp.normalize(normal_sum)
    a = patches.a[p]
    b = patches.b[p]
    scale = 1.0 / float(wp.max(patches.per_body[a], patches.per_body[b]))
    v = _point_velocity(a, point, position_m, velocity_m_per_s, angular_velocity_rad_per_s)
    v = v - _point_velocity(b, point, position_m, velocity_m_per_s, angular_velocity_rad_per_s)
    before = _point_velocity(a, point, position_m, velocity_before, angular_before)
    before = before - _point_velocity(b, point, position_m, velocity_before, angular_before)
    approach = wp.dot(before, n)
    e = wp.max(restitution[a], restitution[b])
    if wp.abs(approach) <= slow_m_per_s:
        e = 0.0
    vn = wp.dot(v, n)
    weight = _weight(a, b, point, n, position_m, orientation, masses_kg, inertia_kg_m2)
    normal_impulse = scale * (wp.max(-e * approach, approach) - vn) / weight
    _push(a, b, point, normal_impulse * n, position_m, orientation, masses_kg, inertia_kg_m2,
          kick_m_per_s, twist_rad_per_s)  # fmt: skip
    mu = _mean(dynamic_friction[a], dynamic_friction[b])
    if mu <= 0.0:
        return
    pressing = wp.max(total / dt_s + normal_impulse, 0.0)
    tangential = v - vn * n
    sliding = wp.length(tangential)
    if sliding > 0.0:
        t = tangential / sliding
        upper = patches.upper[p]
        height = float(0.0)  # noqa: UP018 (a mutable float for Warp)
        if upper >= 0:
            height = wp.abs(wp.dot(point - position_m[upper], n))
        wide = upper >= 0 and _wide(patches.spread[p], t, height, mu)
        if wide:
            ma = masses_kg[a]
            mb = masses_kg[b]
            rub = -wp.min(mu * pressing, scale * sliding / (1.0 / ma + 1.0 / mb)) * t  # on a
            wp.atomic_add(kick_m_per_s, a, rub / ma)
            wp.atomic_sub(kick_m_per_s, b, rub / mb)
        else:
            weight = _weight(a, b, point, t, position_m, orientation, masses_kg, inertia_kg_m2)
            rub = -wp.min(mu * pressing, scale * sliding / weight) * t
            _push(a, b, point, rub, position_m, orientation, masses_kg, inertia_kg_m2,
                  kick_m_per_s, twist_rad_per_s)  # fmt: skip
    twirl = wp.dot(angular_velocity_rad_per_s[a] - angular_velocity_rad_per_s[b], n)
    radius = _torsion_radius(patches.spread[p], n)
    if twirl != 0.0 and radius > 0.0:
        turn_a = _inverse_inertia_world(orientation[a], inertia_kg_m2[a], n)
        turn_b = _inverse_inertia_world(orientation[b], inertia_kg_m2[b], n)
        resist = wp.dot(n, turn_a) + wp.dot(n, turn_b)
        torque = -wp.sign(twirl) * wp.min(mu * pressing * radius, scale * wp.abs(twirl) / resist)
        wp.atomic_add(twist_rad_per_s, a, torque * turn_a)
        wp.atomic_sub(twist_rad_per_s, b, torque * turn_b)


@wp.kernel
def apply_velocity_corrections(
    kick_m_per_s: wp.array(dtype=wp.vec3),
    twist_rad_per_s: wp.array(dtype=wp.vec3),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
):
    """Add every body's velocity corrections, then clear them."""
    i = wp.tid()
    velocity_m_per_s[i] = velocity_m_per_s[i] + kick_m_per_s[i]
    angular_velocity_rad_per_s[i] = angular_velocity_rad_per_s[i] + twist_rad_per_s[i]
    kick_m_per_s[i] = wp.vec3(0.0, 0.0, 0.0)
    twist_rad_per_s[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def count_nonfinite(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    status: wp.array(dtype=float),
):
    """status[0] counts the bodies whose state holds NaN or infinity. Orientations are
    scaled to unit length by `_unit`, which keeps a NaN NaN."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    v = velocity_m_per_s[i]
    w = angular_velocity_rad_per_s[i]
    total = x[0] + x[1] + x[2] + q[0] + q[1] + q[2] + q[3] + v[0] + v[1] + v[2] + w[0] + w[1] + w[2]
    if not wp.isfinite(total):
        wp.atomic_add(status, 0, 1.0)


@wp.kernel
def contact_status(
    contact_count: wp.array(dtype=wp.int32),
    static_contacts: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    status: wp.array(dtype=float),
):
    """status[1] the last substep's contacts (between bodies and with the static
    solids), status[2] the pairs of bodies dropped for want of room in the last step."""
    status[1] = float(contact_count[0] + static_contacts[0])
    status[2] = float(overflows[0])

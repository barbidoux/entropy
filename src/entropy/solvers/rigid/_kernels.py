"""Warp kernels of the rigid bodies (SPEC 5.4).

pyright cannot type Warp's kernel maths (its stubs do not cover vectors and quaternions
in kernels), so those checks are off in this file, as in the liquid's kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false, reportReturnType=false

import warp as wp

from entropy.fields.solids import nearest

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
"""A capsule's sample spheres lie along its axis at most a radius apart, so nothing
thinner than about its radius slips between them, up to 25 spheres: a capsule longer
than 24 radii has wider gaps (#101's review)."""
TOUCH_M = wp.constant(0.005)
"""A sample within 5 mm of a static surface counts towards its patch's width (`_wide`),
sunk in or not: lifted by its bottom face's centre, a box resting flat has one contact
only, yet the whole face holds it."""
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
    return wp.clamp(int(wp.ceil(2.0 * size[1] / size[0])) + 1, 1, CAPSULE_MAX_SAMPLES)


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


@wp.func
def _mean(a: float, b: float) -> float:
    """A pair's friction: the geometric mean of the two coefficients (ADR 0032)."""
    return wp.sqrt(a * b)


@wp.func
def _spread(second: wp.mat33, total: float, offset: wp.vec3) -> wp.mat33:
    """A patch's spread: the covariance of its points (m²), from their second moment
    `second` about a reference point, summed with weights adding up to `total`, and their
    weighted mean's `offset` from that point."""
    return second / total - wp.outer(offset, offset)


@wp.func
def _wide(spread: wp.mat33, direction: wp.vec3, height_m: float, mu: float) -> bool:
    """Whether a patch is wide enough along `direction` that friction there cannot tip the
    body: its contacts reach √3 sigma along it (the half length of an even spread), at least μ
    times the height of the body's centre above the patch, so the normal forces can shift
    forward and hold the friction's torque (ADR 0032)."""
    reach2 = 3.0 * wp.dot(direction, spread * direction)
    return reach2 >= (mu * height_m) * (mu * height_m)


@wp.func
def _torsion_radius(spread: wp.mat33, n: wp.vec3) -> float:
    """The root-mean-square distance of a patch's contacts from its centre of pressure,
    across the normal (m): the lever of its friction against turning about the normal."""
    across = spread[0, 0] + spread[1, 1] + spread[2, 2] - wp.dot(n, spread * n)
    return wp.sqrt(wp.max(across, 0.0))


# Against the static solids -------------------------------------------------------------


@wp.kernel
def solve_static(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    rotated: wp.array(dtype=wp.vec3),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
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
    """Body i against the static solids, by its own thread: one Gauss-Seidel sweep of
    its sample spheres, each correction applied before the next sample is tested, with
    compliance 0. Within the sweep the body turns to first order (a sample moves by
    spin x r) and by the exact rotation once at the end: composing the samples' small
    rotations one by one would add a second-order turn about the normal, and a box at
    rest on the floor would slowly start to spin. A box's samples go face centres first,
    corners last: a box flat on the floor is then lifted by its bottom face's centre, with
    no turn at all, where corners first would tilt it a little every substep and slide
    the boxes stacked on it off (frictionless) within seconds. A sample takes the surface nearest to
    where it was at the start of the substep and sinks in when its place now passes that
    surface's tangent plane, so a thin wall keeps its inner normal however fast a body
    comes (#101's review). The
    contacts gather into up to `STATIC_PATCHES` patches by direction (a fourth direction
    joins the closest), entries 3i to 3i + 2: their normal impulses (kg·m), the moments
    of their points and their normals, for the velocity pass.

    Then static friction (Müller et al. 2020, §3.5; ADR 0032), once per patch, one after
    the other: the correction that undoes the sliding of the patch's centre of pressure
    this substep, kept only when it is within μs times the patch's normal impulse. One
    exact correction at the centre of pressure, where one per contact would leave part of
    the sliding each substep and a box on a slope would creep. On a patch wide enough
    along the sliding (`_wide`), the correction moves the body without turning it: the
    normal forces would shift and hold the torque, which one point cannot show, and a
    box on a slope would creep by rolling about its centre of pressure. A patch's width
    is the spread of its samples within `TOUCH_M` of the surface, kept in `patch_spread`
    for the velocity pass."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    x0 = x - displacement_m[i]
    q0 = wp.quat_inverse(turn[i]) * q
    mass = masses_kg[i]
    inertia = inertia_kg_m2[i]
    shape = shapes[i]
    size = sizes_m[i]
    shift = wp.vec3(0.0, 0.0, 0.0)
    spin = wp.vec3(0.0, 0.0, 0.0)
    origin = x
    for s in range(STATIC_PATCHES):
        patch_impulse[STATIC_PATCHES * i + s] = 0.0
        patch_moment[STATIC_PATCHES * i + s] = wp.vec3(0.0, 0.0, 0.0)
        patch_normal[STATIC_PATCHES * i + s] = wp.vec3(0.0, 0.0, 0.0)
        patch_spread[STATIC_PATCHES * i + s] = wp.mat33()
        patch_touch[STATIC_PATCHES * i + s] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    used = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    found = int(0)  # noqa: UP018, RUF046 (a mutable int for Warp)
    n0 = wp.vec3(0.0, 0.0, 0.0)
    n1 = wp.vec3(0.0, 0.0, 0.0)
    n2 = wp.vec3(0.0, 0.0, 0.0)
    count = _samples(shape, size)
    for back in range(count):
        k = count - 1 - back  # a box's face centres first, its corners last
        sample = _sample(shape, size, k, signs)
        local = wp.vec3(sample[0], sample[1], sample[2])
        start = x0 + wp.quat_rotate(q0, local)
        hit = nearest(start, box_centres, box_halves, tank_lows, tank_highs, tank_walls,
                      plane_points, plane_normals)  # fmt: skip
        n = wp.vec3(hit[0], hit[1], hit[2])
        surface = start - hit[3] * n
        offset = wp.quat_rotate(q, local)
        centre = x + offset + wp.cross(spin, offset)
        depth = sample[3] - wp.dot(centre - surface, n)
        point = centre - sample[3] * n
        slot = int(-1)  # noqa: UP018, RUF046 (a mutable int for Warp)
        if used > 0 and wp.dot(n, n0) > PATCH_COS:
            slot = 0
        elif used > 1 and wp.dot(n, n1) > PATCH_COS:
            slot = 1
        elif used > 2 and wp.dot(n, n2) > PATCH_COS:
            slot = 2
        elif depth > 0.0:
            if used == 0:
                n0 = n
                slot = 0
                used = 1
            elif used == 1:
                n1 = n
                slot = 1
                used = 2
            elif used == 2:
                n2 = n
                slot = 2
                used = 3
            else:
                slot = 0
                best = wp.dot(n, n0)
                if wp.dot(n, n1) > best:
                    slot = 1
                    best = wp.dot(n, n1)
                if wp.dot(n, n2) > best:
                    slot = 2
        if slot >= 0 and depth > -TOUCH_M:
            entry = STATIC_PATCHES * i + slot
            patch_touch[entry] = patch_touch[entry] + wp.vec4(
                point[0] - origin[0], point[1] - origin[1], point[2] - origin[2], 1.0
            )
            patch_spread[entry] = patch_spread[entry] + wp.outer(point - origin, point - origin)
        if depth > 0.0:
            found += 1
            arm = point - x
            t = wp.cross(arm, n)
            weight = 1.0 / mass + wp.dot(t, _inverse_inertia_world(q, inertia, t))
            impulse = depth / weight
            step = impulse / mass * n
            x = x + step
            shift = shift + step
            spin = spin + _inverse_inertia_world(q, inertia, impulse * t)
            entry = STATIC_PATCHES * i + slot
            patch_impulse[entry] = patch_impulse[entry] + impulse
            patch_moment[entry] = patch_moment[entry] + impulse * point
            patch_normal[entry] = patch_normal[entry] + impulse * n
    mu = _mean(static_friction[i], solids_static_friction)
    undo = wp.quat_inverse(turn[i])
    for s in range(STATIC_PATCHES):
        entry = STATIC_PATCHES * i + s
        total = patch_impulse[entry]
        normal_sum = patch_normal[entry]
        if total > 0.0 and wp.length(normal_sum) > 0.0:
            point = patch_moment[entry] / total
            touch = patch_touch[entry]
            mean = wp.vec3(touch[0], touch[1], touch[2]) / touch[3]
            spread = _spread(patch_spread[entry], touch[3], mean)
            patch_spread[entry] = spread
            n = wp.normalize(normal_sum)
            arm = point - x
            # Where the body's material point now at `point` was when the substep began.
            start = x0 + wp.quat_rotate(undo, arm - wp.cross(spin, arm))
            slip = point - start
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
                    step = -held / mass * direction
                    x = x + step
                    shift = shift + step
                    if not wide:
                        spin = spin - _inverse_inertia_world(q, inertia, held * lever)
    if found > 0:
        correction = _rotation_quat(spin, 1.0)
        position_m[i] = x
        orientation[i] = _unit(correction * q)
        displacement_m[i] = displacement_m[i] + shift
        turn[i] = _unit(correction * turn[i])
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
    """The velocity pass of body i's static patches (Müller et al. 2020, §3.6; ADR 0032),
    by its own thread, one after the other, at each patch's centre of pressure.

    Restitution: the separating speed there becomes e times the approaching speed it had
    before the substep, or the separating speed it had if it was separating; no bounce
    below `slow_m_per_s` (2|g|h), so resting bodies do not jitter. The depth corrections
    then never turn into speed, and a body spawned in the floor does not pop out. The
    pair's e is the larger of the body's and the solids'.

    Dynamic friction: an impulse against the sliding there, of at most μd times the
    patch's normal impulse in the substep (its depth corrections over h plus the
    restitution's impulse, so a contact met at its surface still rubs); on a wide patch
    (`_wide`) it slows the body without turning it, as static friction does. Then the
    friction against turning about the normal: at most μd times the normal impulse times
    the patch's torsion radius. The pair's μd is the geometric mean of the body's and the
    solids'."""
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
            before = velocity_before[i] + wp.cross(angular_before[i], arm)
            now = v + wp.cross(w, arm)
            approach = wp.dot(before, n)
            bounce = e
            if wp.abs(approach) <= slow_m_per_s:
                bounce = 0.0
            lever = wp.cross(arm, n)
            weight = 1.0 / mass + wp.dot(lever, _inverse_inertia_world(q, inertia, lever))
            impulse = (wp.max(-bounce * approach, approach) - wp.dot(now, n)) / weight
            v = v + impulse / mass * n
            w = w + _inverse_inertia_world(q, inertia, impulse * lever)
            now = v + wp.cross(w, arm)
            tangential = now - wp.dot(now, n) * n
            sliding = wp.length(tangential)
            pressing = wp.max(total / dt_s + impulse, 0.0)
            if sliding > 0.0 and mu > 0.0:
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
            if twirl != 0.0 and mu > 0.0 and radius > 0.0:
                resist = wp.dot(n, _inverse_inertia_world(q, inertia, n))
                torque = wp.min(mu * pressing * radius, wp.abs(twirl) / resist)
                w = w - wp.sign(twirl) * torque * _inverse_inertia_world(q, inertia, n)
    if changed > 0:
        velocity_m_per_s[i] = v
        angular_velocity_rad_per_s[i] = w


# Between bodies ---------------------------------------------------------------------------


# The patches of a substep's pairs of bodies (ADR 0031): all the contacts of one pair.
# Friction and the velocity pass act once per patch, at its centre of pressure (ADR 0032).
@wp.struct
class Patches:
    count: wp.array(dtype=wp.int32)
    a: wp.array(dtype=wp.int32)
    b: wp.array(dtype=wp.int32)
    impulse: wp.array(dtype=float)
    moment: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)
    anchor: wp.array(dtype=wp.vec3)
    second: wp.array(dtype=wp.mat33)
    per_body: wp.array(dtype=wp.int32)


@wp.func
def _new_patch(a: int, b: int, point: wp.vec3, patches: Patches) -> int:
    """A new patch of bodies a and b, anchored at its first contact's `point` (its spread
    is summed about it), or -1 when there is no room left; its contacts then get no
    velocity pass (they are dropped too, since there are no more patches than
    contacts)."""
    slot = wp.atomic_add(patches.count, 0, 1)
    if slot >= patches.a.shape[0]:
        return -1
    patches.a[slot] = a
    patches.b[slot] = b
    patches.impulse[slot] = 0.0
    patches.moment[slot] = wp.vec3(0.0, 0.0, 0.0)
    patches.normal[slot] = wp.vec3(0.0, 0.0, 0.0)
    patches.anchor[slot] = point
    patches.second[slot] = wp.mat33()
    wp.atomic_add(patches.per_body, a, 1)
    wp.atomic_add(patches.per_body, b, 1)
    return slot


@wp.func
def _add_contact(
    a: int,
    b: int,
    point_m: wp.vec3,
    normal: wp.vec3,
    depth_m: float,
    patch: int,
    contact_count: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
    contact_patch: wp.array(dtype=wp.int32),
):
    slot = wp.atomic_add(contact_count, 0, 1)
    if slot >= contact_a.shape[0]:
        wp.atomic_add(overflows, 0, 1)
        return
    contact_a[slot] = a
    contact_b[slot] = b
    contact_point_m[slot] = point_m
    contact_normal[slot] = normal
    contact_depth_m[slot] = depth_m
    contact_patch[slot] = patch
    wp.atomic_add(body_contacts, a, 1)
    wp.atomic_add(body_contacts, b, 1)


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
def _against_body(
    a: int,
    b: int,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    signs: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
    contact_patch: wp.array(dtype=wp.int32),
    patches: Patches,
    patch: int,
    tested: int,
    axis: wp.vec3,
) -> int:
    """Body a's first `tested` sample spheres against body b's surface: a contact for
    each that sinks in, its normal pushing a out of b, in the pair's patch (made at its
    first contact, while `patch` is still -1). Two boxes give `axis` (non-zero), their
    shared normal: a sample inside b sinks along it by b's reach along it past the
    sample's. Returns the pair's patch."""
    current = int(patch)
    xa = position_m[a]
    qa = orientation[a]
    xb = position_m[b]
    qb = orientation[b]
    shape_a = shapes[a]
    shared = wp.length(axis) > 0.0
    reach = float(0.0)  # noqa: UP018 (a mutable float for Warp)
    if shared:
        reach = wp.dot(xb, axis) + _box_radius(wp.quat_to_matrix(qb), sizes_m[b], axis)
    for k in range(tested):
        sample = _sample(shape_a, sizes_m[a], k, signs)
        centre = xa + wp.quat_rotate(qa, wp.vec3(sample[0], sample[1], sample[2]))
        hit = _body_distance(shapes[b], sizes_m[b], wp.quat_rotate_inv(qb, centre - xb))
        depth = sample[3] - hit[3]
        n = wp.quat_rotate(qb, wp.vec3(hit[0], hit[1], hit[2]))
        if shared:
            n = axis
            depth = wp.min(depth, sample[3] + reach - wp.dot(centre, axis))
        if depth > 0.0:
            if current < 0:
                current = _new_patch(a, b, centre - sample[3] * n, patches)
            _add_contact(a, b, centre - sample[3] * n, n, depth, current, contact_count,
                         overflows, body_contacts, contact_a, contact_b, contact_point_m,
                         contact_normal, contact_depth_m, contact_patch)  # fmt: skip
    return current


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
    contact_count: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
    contact_patch: wp.array(dtype=wp.int32),
    patches: Patches,
):
    """Body i's contacts with the other bodies: for each neighbour j > i within reach
    (the hash grid), i's samples against j and j's against i (as `_tested` says; two
    spheres once), in one patch for the pair. Each pair is met once."""
    i = wp.tid()
    x = position_m[i]
    query = wp.hash_grid_query(grid, x, bounds_m[i] + max_bound_m)
    j = int(0)  # noqa: UP018, RUF046 (Warp needs a mutable int)
    while wp.hash_grid_query_next(query, j):
        if j > i and wp.length(position_m[j] - x) < bounds_m[i] + bounds_m[j]:
            pair = int(-1)  # noqa: UP018, RUF046 (a mutable int for Warp)
            there = _tested(shapes[i], sizes_m[i], shapes[j])
            axis = wp.vec3(0.0, 0.0, 0.0)
            if shapes[i] == 0 and shapes[j] == 0:
                axis = _box_axis(x, orientation[i], sizes_m[i], position_m[j], orientation[j],
                                 sizes_m[j])  # fmt: skip
            pair = _against_body(i, j, position_m, orientation, shapes, sizes_m, signs,
                                 contact_count, overflows, body_contacts, contact_a, contact_b,
                                 contact_point_m, contact_normal, contact_depth_m, contact_patch,
                                 patches, pair, there, axis)  # fmt: skip
            back = _tested(shapes[j], sizes_m[j], shapes[i])
            if shapes[i] == 1 and shapes[j] == 1:
                back = 0
            pair = _against_body(j, i, position_m, orientation, shapes, sizes_m, signs,
                                 contact_count, overflows, body_contacts, contact_a, contact_b,
                                 contact_point_m, contact_normal, contact_depth_m, contact_patch,
                                 patches, pair, back, -axis)  # fmt: skip


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
def solve_contacts(
    contact_count: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
    contact_patch: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    patches: Patches,
    shift_m: wp.array(dtype=wp.vec3),
    spin: wp.array(dtype=wp.vec3),
):
    """One XPBD iteration of every contact between bodies (compliance 0), in parallel
    (Jacobi). Each contact's correction is scaled by 1 / (the larger contact count of its
    two bodies) and given to both alike, so momentum is kept (ADR 0031). The corrections
    add up in `shift_m` (a position) and `spin` (a small rotation vector), and into the
    contact's patch: its normal impulse, centre of pressure and normal."""
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_a.shape[0]):
        return
    a = contact_a[c]
    b = contact_b[c]
    n = contact_normal[c]
    point = contact_point_m[c]
    count = wp.max(body_contacts[a], body_contacts[b])
    weight = _weight(a, b, point, n, position_m, orientation, masses_kg, inertia_kg_m2)
    impulse = contact_depth_m[c] / weight / float(count)
    _push(
        a, b, point, impulse * n, position_m, orientation, masses_kg, inertia_kg_m2, shift_m, spin
    )
    p = contact_patch[c]
    if p >= 0:
        sign = 1.0
        if a != patches.a[p]:
            sign = -1.0
        wp.atomic_add(patches.impulse, p, impulse)
        wp.atomic_add(patches.moment, p, impulse * point)
        wp.atomic_add(patches.normal, p, (sign * impulse) * n)
        offset = point - patches.anchor[p]
        wp.atomic_add(patches.second, p, impulse * wp.outer(offset, offset))


@wp.func
def _point_motion(
    body: int,
    point_m: wp.vec3,
    position_m: wp.array(dtype=wp.vec3),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
) -> wp.vec3:
    """How far a body's material point now at `point_m` moved this substep."""
    r = point_m - position_m[body]
    return displacement_m[body] + r - wp.quat_rotate_inv(turn[body], r)


@wp.kernel
def solve_friction(
    patches: Patches,
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    static_friction: wp.array(dtype=float),
    shift_m: wp.array(dtype=wp.vec3),
    spin: wp.array(dtype=wp.vec3),
):
    """Static friction between bodies (Müller et al. 2020, §3.5; ADR 0032), after their
    depths, once per pair's patch, in parallel: the correction that undoes the sliding
    of the patch's centre of pressure this substep, kept only when it is within μs (the
    geometric mean of the two) times the patch's normal impulse. Scaled by 1 / (the
    larger patch count of the two bodies) and given to both alike, so momentum is kept."""
    p = wp.tid()
    if p >= wp.min(patches.count[0], patches.a.shape[0]):
        return
    total = patches.impulse[p]
    normal_sum = patches.normal[p]
    if total <= 0.0 or wp.length(normal_sum) <= 0.0:
        return
    a = patches.a[p]
    b = patches.b[p]
    mu = _mean(static_friction[a], static_friction[b])
    if mu <= 0.0:
        return
    point = patches.moment[p] / total
    n = wp.normalize(normal_sum)
    slip = _point_motion(a, point, position_m, displacement_m, turn)
    slip = slip - _point_motion(b, point, position_m, displacement_m, turn)
    tangential = slip - wp.dot(slip, n) * n
    sliding = wp.length(tangential)
    if sliding <= 0.0:
        return
    t = tangential / sliding
    count = wp.max(patches.per_body[a], patches.per_body[b])
    weight = _weight(a, b, point, t, position_m, orientation, masses_kg, inertia_kg_m2)
    held = sliding / weight / float(count)
    if held <= mu * total:
        _push(a, b, point, -held * t, position_m, orientation, masses_kg, inertia_kg_m2,
              shift_m, spin)  # fmt: skip


@wp.kernel
def apply_corrections(
    shift_m: wp.array(dtype=wp.vec3),
    spin: wp.array(dtype=wp.vec3),
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
    s = spin[i]
    correction = _rotation_quat(s, 1.0)
    orientation[i] = _unit(correction * orientation[i])
    turn[i] = _unit(correction * turn[i])
    rotated[i] = rotated[i] + s
    shift_m[i] = wp.vec3(0.0, 0.0, 0.0)
    spin[i] = wp.vec3(0.0, 0.0, 0.0)


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
    kick: wp.array(dtype=wp.vec3),
    twist: wp.array(dtype=wp.vec3),
):
    """The velocity pass of the pairs' patches, in parallel, at each one's centre of
    pressure: restitution, dynamic friction and the friction against turning about the
    normal as `solve_static_velocities`, with the larger e of the two bodies and the
    geometric mean of their μd, scaled by 1 / (the larger patch count of the two bodies)
    and given to both alike. The friction's bound is not scaled: it is the patch's own
    normal impulse. Between bodies friction always turns them (no wide rule), so both
    momenta are kept."""
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
    count = wp.max(patches.per_body[a], patches.per_body[b])
    v = _point_velocity(a, point, position_m, velocity_m_per_s, angular_velocity_rad_per_s)
    v = v - _point_velocity(b, point, position_m, velocity_m_per_s, angular_velocity_rad_per_s)
    before = _point_velocity(a, point, position_m, velocity_before, angular_before)
    before = before - _point_velocity(b, point, position_m, velocity_before, angular_before)
    scale = 1.0 / float(count)
    approach = wp.dot(before, n)
    e = wp.max(restitution[a], restitution[b])
    if wp.abs(approach) <= slow_m_per_s:
        e = 0.0
    vn = wp.dot(v, n)
    weight = _weight(a, b, point, n, position_m, orientation, masses_kg, inertia_kg_m2)
    normal_impulse = scale * (wp.max(-e * approach, approach) - vn) / weight
    impulse = normal_impulse * n
    mu = _mean(dynamic_friction[a], dynamic_friction[b])
    tangential = v - vn * n
    sliding = wp.length(tangential)
    if sliding > 0.0 and mu > 0.0:
        t = tangential / sliding
        weight = _weight(a, b, point, t, position_m, orientation, masses_kg, inertia_kg_m2)
        bound = mu * wp.max(total / dt_s + normal_impulse, 0.0)
        impulse = impulse - wp.min(bound, scale * sliding / weight) * t
    _push(a, b, point, impulse, position_m, orientation, masses_kg, inertia_kg_m2, kick, twist)
    wa = angular_velocity_rad_per_s[a]
    wb = angular_velocity_rad_per_s[b]
    twirl = wp.dot(wa - wb, n)
    spread = _spread(patches.second[p], total, point - patches.anchor[p])
    radius = _torsion_radius(spread, n)
    if twirl != 0.0 and mu > 0.0 and radius > 0.0:
        turn_a = _inverse_inertia_world(orientation[a], inertia_kg_m2[a], n)
        turn_b = _inverse_inertia_world(orientation[b], inertia_kg_m2[b], n)
        resist = wp.dot(n, turn_a) + wp.dot(n, turn_b)
        bound = mu * wp.max(total / dt_s + normal_impulse, 0.0) * radius
        torque = -wp.sign(twirl) * wp.min(bound, scale * wp.abs(twirl) / resist)
        wp.atomic_add(twist, a, torque * turn_a)
        wp.atomic_sub(twist, b, torque * turn_b)


@wp.kernel
def apply_velocity_corrections(
    kick: wp.array(dtype=wp.vec3),
    twist: wp.array(dtype=wp.vec3),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
):
    """Add every body's velocity corrections, then clear them."""
    i = wp.tid()
    velocity_m_per_s[i] = velocity_m_per_s[i] + kick[i]
    angular_velocity_rad_per_s[i] = angular_velocity_rad_per_s[i] + twist[i]
    kick[i] = wp.vec3(0.0, 0.0, 0.0)
    twist[i] = wp.vec3(0.0, 0.0, 0.0)


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
    capacity: int,
    status: wp.array(dtype=float),
):
    """status[1] the last substep's contacts (between bodies, at most the capacity, and
    with the static solids), status[2] the contacts between bodies dropped for want of
    room in the last step."""
    status[1] = float(wp.min(contact_count[0], capacity) + static_contacts[0])
    status[2] = float(overflows[0])

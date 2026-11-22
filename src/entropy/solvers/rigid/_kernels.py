"""Warp kernels of the rigid bodies (SPEC 5.4).

pyright cannot type Warp's kernel maths (its stubs do not cover vectors and quaternions
in kernels), so those checks are off in this file, as in the liquid's kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false, reportReturnType=false

import warp as wp

from entropy.fields._solid_kernels import nearest

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
def _angular_velocity(rotation: wp.quat, dt_s: float) -> wp.vec3:
    """The angular velocity that turns by `rotation` in `dt_s` (the exact logarithm).

    No sign flip for a negative w: the rotation is built from the velocity itself, so a
    turn of more than π in a substep (a spin above π/h, about 1,500 rad/s at 1/480 s) is
    a real one and keeps its sense, up to 2π/h. Constraints add small corrections only."""
    v = wp.vec3(rotation[0], rotation[1], rotation[2])
    s = wp.length(v)
    if s < 1.0e-12:
        return 2.0 * v / dt_s
    return v / s * (2.0 * wp.atan2(s, rotation[3]) / dt_s)


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
):
    """The first half of an XPBD substep (Müller et al. 2020): move every body with
    gravity and the gyroscopic torque, and keep the motion as increments, a displacement
    and a turn. Constraints add their corrections to both; `recover` turns them into the
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


@wp.kernel
def recover(
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    dt_s: float,
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
):
    """The second half of a substep: the velocities that made this substep's motion."""
    i = wp.tid()
    velocity_m_per_s[i] = displacement_m[i] / dt_s
    angular_velocity_rad_per_s[i] = _angular_velocity(turn[i], dt_s)


# Contacts (ADR 0031) ----------------------------------------------------------------

BOX_SAMPLES = wp.constant(26)
"""A box's sample points: its 8 corners, 12 edge midpoints and 6 face centres. Two
equal boxes face to face put every corner and edge midpoint on the other's edges, at
distance 0: only the face centres sink in."""
CAPSULE_SAMPLES = wp.constant(5)
"""A capsule's sample spheres, along its axis at -h, -h/2, 0, h/2 and h."""


@wp.func
def _samples(shape: int) -> int:
    if shape == 0:
        return BOX_SAMPLES
    if shape == 1:
        return 1
    return CAPSULE_SAMPLES


@wp.func
def _sample(shape: int, size: wp.vec3, k: int, signs: wp.array(dtype=wp.vec3)) -> wp.vec4:
    """(x, y, z, radius) of a body's sample k, in its frame: a sphere about the point
    that the surface wraps (radius 0 for a box's corners and edges)."""
    if shape == 0:
        c = wp.cw_mul(signs[k], size)
        return wp.vec4(c[0], c[1], c[2], 0.0)
    if shape == 1:
        return wp.vec4(0.0, 0.0, 0.0, size[0])
    t = -1.0 + 0.5 * float(k)
    return wp.vec4(0.0, 0.0, t * size[1], size[0])


@wp.func
def _body_distance(shape: int, size: wp.vec3, p: wp.vec3) -> wp.vec4:
    """(outward normal x, y, z, signed distance) of a body's surface at a point of its
    own frame: exact for the box (Quilez), the sphere and the capsule."""
    if shape == 0:
        q = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2])) - size
        outside = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        inside = wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)
        d = wp.length(outside) + inside
        sign = wp.vec3(wp.sign(p[0]), wp.sign(p[1]), wp.sign(p[2]))
        for a in range(3):
            if sign[a] == 0.0:
                sign[a] = 1.0
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
def _add_contact(
    a: int,
    b: int,
    point_m: wp.vec3,
    normal: wp.vec3,
    depth_m: float,
    contact_count: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
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
    wp.atomic_add(body_contacts, a, 1)
    if b >= 0:
        wp.atomic_add(body_contacts, b, 1)


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
):
    """Body a's sample spheres against body b's surface: a contact for each that sinks
    in, its normal pushing a out of b."""
    xa = position_m[a]
    qa = orientation[a]
    xb = position_m[b]
    qb = orientation[b]
    shape_a = shapes[a]
    for k in range(_samples(shape_a)):
        sample = _sample(shape_a, sizes_m[a], k, signs)
        centre = xa + wp.quat_rotate(qa, wp.vec3(sample[0], sample[1], sample[2]))
        hit = _body_distance(shapes[b], sizes_m[b], wp.quat_rotate_inv(qb, centre - xb))
        depth = sample[3] - hit[3]
        if depth > 0.0:
            n = wp.quat_rotate(qb, wp.vec3(hit[0], hit[1], hit[2]))
            _add_contact(a, b, centre - sample[3] * n, n, depth, contact_count, overflows,
                         body_contacts, contact_a, contact_b, contact_point_m, contact_normal,
                         contact_depth_m)  # fmt: skip


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
    box_centres: wp.array(dtype=wp.vec3),
    box_halves: wp.array(dtype=wp.vec3),
    tank_lows: wp.array(dtype=wp.vec3),
    tank_highs: wp.array(dtype=wp.vec3),
    tank_walls: wp.array(dtype=float),
    plane_points: wp.array(dtype=wp.vec3),
    plane_normals: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=wp.int32),
    overflows: wp.array(dtype=wp.int32),
    body_contacts: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
):
    """Body i's contacts: its sample spheres against the static solids, then, for each
    neighbour j > i within reach (the hash grid), i's samples against j and j's against
    i. Each pair is met once."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    shape = shapes[i]
    for k in range(_samples(shape)):
        sample = _sample(shape, sizes_m[i], k, signs)
        centre = x + wp.quat_rotate(q, wp.vec3(sample[0], sample[1], sample[2]))
        hit = nearest(centre, box_centres, box_halves, tank_lows, tank_highs, tank_walls,
                      plane_points, plane_normals)  # fmt: skip
        depth = sample[3] - hit[3]
        if depth > 0.0:
            n = wp.vec3(hit[0], hit[1], hit[2])
            _add_contact(i, -1, centre - sample[3] * n, n, depth, contact_count, overflows,
                         body_contacts, contact_a, contact_b, contact_point_m, contact_normal,
                         contact_depth_m)  # fmt: skip
    query = wp.hash_grid_query(grid, x, bounds_m[i] + max_bound_m)
    j = int(0)  # noqa: UP018, RUF046 (Warp needs a mutable int)
    while wp.hash_grid_query_next(query, j):
        if j > i and wp.length(position_m[j] - x) < bounds_m[i] + bounds_m[j]:
            _against_body(i, j, position_m, orientation, shapes, sizes_m, signs, contact_count,
                          overflows, body_contacts, contact_a, contact_b, contact_point_m,
                          contact_normal, contact_depth_m)  # fmt: skip
            _against_body(j, i, position_m, orientation, shapes, sizes_m, signs, contact_count,
                          overflows, body_contacts, contact_a, contact_b, contact_point_m,
                          contact_normal, contact_depth_m)  # fmt: skip


@wp.func
def _inverse_inertia_world(q: wp.quat, inertia: wp.vec3, v: wp.vec3) -> wp.vec3:
    """I⁻¹ v in the world's frame, from the principal moments in the body's."""
    return wp.quat_rotate(q, wp.cw_div(wp.quat_rotate_inv(q, v), inertia))


@wp.kernel
def solve_contacts(
    contact_count: wp.array(dtype=wp.int32),
    contact_a: wp.array(dtype=wp.int32),
    contact_b: wp.array(dtype=wp.int32),
    contact_point_m: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_depth_m: wp.array(dtype=float),
    body_contacts: wp.array(dtype=wp.int32),
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    masses_kg: wp.array(dtype=float),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    shift_m: wp.array(dtype=wp.vec3),
    spin: wp.array(dtype=wp.vec3),
):
    """One XPBD iteration of every contact (compliance 0), in parallel: the impulse that
    closes its depth, scaled by 1 / (the larger contact count of its two bodies) and
    given to both alike, so that momentum is kept (ADR 0031). The bodies' corrections
    add up in `shift_m` (a position) and `spin` (a small rotation vector)."""
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_a.shape[0]):
        return
    a = contact_a[c]
    b = contact_b[c]
    n = contact_normal[c]
    point = contact_point_m[c]
    qa = orientation[a]
    ra = point - position_m[a]
    ta = wp.cross(ra, n)
    weight = 1.0 / masses_kg[a] + wp.dot(ta, _inverse_inertia_world(qa, inertia_kg_m2[a], ta))
    count = body_contacts[a]
    if b >= 0:
        qb = orientation[b]
        rb = point - position_m[b]
        tb = wp.cross(rb, n)
        weight += 1.0 / masses_kg[b] + wp.dot(tb, _inverse_inertia_world(qb, inertia_kg_m2[b], tb))
        count = wp.max(count, body_contacts[b])
    impulse = (contact_depth_m[c] / (weight * float(count))) * n
    wp.atomic_add(shift_m, a, impulse / masses_kg[a])
    wp.atomic_add(spin, a, _inverse_inertia_world(qa, inertia_kg_m2[a], wp.cross(ra, impulse)))
    if b >= 0:
        rb = point - position_m[b]
        qb = orientation[b]
        wp.atomic_sub(shift_m, b, impulse / masses_kg[b])
        wp.atomic_sub(spin, b, _inverse_inertia_world(qb, inertia_kg_m2[b], wp.cross(rb, impulse)))


@wp.kernel
def apply_corrections(
    shift_m: wp.array(dtype=wp.vec3),
    spin: wp.array(dtype=wp.vec3),
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    displacement_m: wp.array(dtype=wp.vec3),
    turn: wp.array(dtype=wp.quat),
    body_contacts: wp.array(dtype=wp.int32),
):
    """Move every body by its corrections, into its position and its increments, then
    clear them for the next substep."""
    i = wp.tid()
    shift = shift_m[i]
    position_m[i] = position_m[i] + shift
    displacement_m[i] = displacement_m[i] + shift
    s = spin[i]
    correction = wp.quat(0.5 * s[0], 0.5 * s[1], 0.5 * s[2], 0.0)
    q = orientation[i]
    orientation[i] = _unit(q + correction * q)
    t = turn[i]
    turn[i] = _unit(t + correction * t)
    shift_m[i] = wp.vec3(0.0, 0.0, 0.0)
    spin[i] = wp.vec3(0.0, 0.0, 0.0)
    body_contacts[i] = 0


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
    overflows: wp.array(dtype=wp.int32),
    capacity: int,
    status: wp.array(dtype=float),
):
    """status[1] the last substep's contacts (at most the capacity), status[2] the
    contacts dropped for want of room since the start."""
    status[1] = float(wp.min(contact_count[0], capacity))
    status[2] = float(overflows[0])

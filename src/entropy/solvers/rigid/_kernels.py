"""Warp kernels of the rigid bodies (SPEC 5.4).

pyright cannot type Warp's kernel maths (its stubs do not cover vectors and quaternions
in kernels), so those checks are off in this file, as in the liquid's kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false, reportReturnType=false

import warp as wp


@wp.func
def _rotation_quat(omega: wp.vec3, dt: float) -> wp.quat:
    """The rotation by angle |omega| dt about omega's direction (the exact exponential)."""
    speed = wp.length(omega)
    if speed * dt < 1.0e-12:
        return wp.quat_identity()
    return wp.quat_from_axis_angle(omega / speed, speed * dt)


@wp.func
def _angular_velocity(rotation: wp.quat, dt: float) -> wp.vec3:
    """The angular velocity that turns by `rotation` in `dt` (the exact logarithm)."""
    q = rotation
    if q[3] < 0.0:
        q = wp.quat(-q[0], -q[1], -q[2], -q[3])
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    if s < 1.0e-12:
        return 2.0 * v / dt
    return v / s * (2.0 * wp.atan2(s, q[3]) / dt)


@wp.kernel
def substep(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    gravity_m_per_s2: wp.vec3,
    dt_s: float,
):
    """One XPBD substep without constraints (Müller et al. 2020): predict with gravity
    and the gyroscopic torque, then recover the velocities from the motion.

    The motion is kept as increments, a displacement and a small rotation, and the
    velocities are recovered from them, not from the difference of two nearby positions
    or orientations: in float32 that difference loses digits (at 20 m from the origin a
    position's spacing is 1.9e-6 m; a quaternion's components near 1 lose about 1e-7 of
    a rotation of 1e-3 rad). Constraints (contacts, #87) will add their corrections to
    the increments (ADR 0030)."""
    i = wp.tid()
    q0 = orientation[i]
    v = velocity_m_per_s[i] + gravity_m_per_s2 * dt_s
    # Gyroscopic torque, in the body's frame, where the inertia is diagonal. Only its
    # change of the angular velocity goes through the rotations: a round trip of the
    # velocity itself would scale it by |q|^4, which float32 leaves 1e-7 off 1 (a 1e-4
    # drift a second at 480 substeps).
    inertia = inertia_kg_m2[i]
    w = angular_velocity[i]
    w_body = wp.quat_rotate_inv(q0, w)
    torque = -wp.cross(w_body, wp.cw_mul(inertia, w_body))
    w = w + wp.quat_rotate(q0, dt_s * wp.cw_div(torque, inertia))
    displacement = v * dt_s
    rotation = _rotation_quat(w, dt_s)
    position_m[i] = position_m[i] + displacement
    orientation[i] = wp.normalize(rotation * q0)
    velocity_m_per_s[i] = displacement / dt_s
    angular_velocity[i] = _angular_velocity(rotation, dt_s)


@wp.kernel
def count_nonfinite(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity: wp.array(dtype=wp.vec3),
    status: wp.array(dtype=float),
):
    """status[0] counts the bodies whose state holds NaN or infinity."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    v = velocity_m_per_s[i]
    w = angular_velocity[i]
    total = x[0] + x[1] + x[2] + q[0] + q[1] + q[2] + q[3] + v[0] + v[1] + v[2] + w[0] + w[1] + w[2]
    if not wp.isfinite(total):
        wp.atomic_add(status, 0, 1.0)

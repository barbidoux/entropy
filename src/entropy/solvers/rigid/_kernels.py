"""Warp kernels of the rigid bodies (SPEC 5.4).

pyright cannot type Warp's kernel maths (its stubs do not cover vectors and quaternions
in kernels), so those checks are off in this file, as in the liquid's kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false, reportReturnType=false

import warp as wp

GYROSCOPIC_ITERATIONS = 4
"""Fixed-point iterations of the implicit midpoint rule for the gyroscopic torque."""


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
    both every substep (#98's review); the iteration converges while |w| dt is well
    under 1 (a few hundred rad/s at 1/480 s)."""
    w_next = w_body
    for _k in range(GYROSCOPIC_ITERATIONS):
        w_mid = 0.5 * (w_body + w_next)
        w_next = w_body - dt_s * wp.cw_div(wp.cross(w_mid, wp.cw_mul(inertia, w_mid)), inertia)
    return w_next


@wp.kernel
def substep(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    inertia_kg_m2: wp.array(dtype=wp.vec3),
    gravity_m_per_s2: wp.vec3,
    dt_s: float,
):
    """One XPBD substep without constraints (Müller et al. 2020): predict with gravity
    and the gyroscopic torque, then recover the velocities from the motion.

    The motion is kept as increments, a displacement and a small rotation, and the
    velocities are recovered from them, not from the difference of two nearby positions
    or orientations, which float32 does not keep exact (ADR 0030). Constraints
    (contacts, #87) will add their corrections to the increments."""
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
    orientation[i] = wp.normalize(rotation * q0)
    velocity_m_per_s[i] = displacement / dt_s
    angular_velocity_rad_per_s[i] = _angular_velocity(rotation, dt_s)


@wp.kernel
def count_nonfinite(
    position_m: wp.array(dtype=wp.vec3),
    orientation: wp.array(dtype=wp.quat),
    velocity_m_per_s: wp.array(dtype=wp.vec3),
    angular_velocity_rad_per_s: wp.array(dtype=wp.vec3),
    status: wp.array(dtype=float),
):
    """status[0] counts the bodies whose state holds NaN or infinity, or whose
    orientation is no longer a unit quaternion (normalising a NaN gives zero)."""
    i = wp.tid()
    x = position_m[i]
    q = orientation[i]
    v = velocity_m_per_s[i]
    w = angular_velocity_rad_per_s[i]
    total = x[0] + x[1] + x[2] + q[0] + q[1] + q[2] + q[3] + v[0] + v[1] + v[2] + w[0] + w[1] + w[2]
    norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if not wp.isfinite(total) or wp.abs(norm - 1.0) > 1.0e-3:
        wp.atomic_add(status, 0, 1.0)

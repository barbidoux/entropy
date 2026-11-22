"""Warp kernels of `entropy.render.bodies`.

pyright cannot type Warp's kernel maths, so those checks are off in this file, as in the
solvers' kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false

import warp as wp


@wp.kernel
def write_instances(
    positions_m: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    albedo_linear_rgb: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec4),
):
    """Four vec4s per body: position and shape, orientation, size, albedo."""
    i = wp.tid()
    p = positions_m[i]
    q = orientations[i]
    s = sizes_m[i]
    c = albedo_linear_rgb[i]
    out[4 * i] = wp.vec4(p[0], p[1], p[2], float(shapes[i]))
    out[4 * i + 1] = wp.vec4(q[0], q[1], q[2], q[3])
    out[4 * i + 2] = wp.vec4(s[0], s[1], s[2], 0.0)
    out[4 * i + 3] = wp.vec4(c[0], c[1], c[2], 1.0)

"""Warp kernels of `entropy.render.bodies`.

pyright cannot type Warp's kernel maths, so those checks are off in this file, as in the
solvers' kernels."""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false, reportIndexIssue=false
# pyright: reportInvalidTypeForm=false

import warp as wp

COMMAND_INTS = wp.constant(5)
"""int32s per indirect draw command, as moderngl strides them (20 bytes)."""


@wp.kernel
def reset_commands(
    vertex_counts: wp.array(dtype=wp.int32),
    capacity: int,
    commands: wp.array(dtype=wp.int32),
):
    """One indirect draw per mesh k (glDrawArraysIndirect's count, instance count,
    first vertex, base instance, then a pad: moderngl reads indirect commands 20 bytes
    apart): its vertices, no instances yet, from vertex 0, its instances starting at k
    times capacity."""
    k = wp.tid()
    commands[COMMAND_INTS * k] = vertex_counts[k]
    commands[COMMAND_INTS * k + 1] = 0
    commands[COMMAND_INTS * k + 2] = 0
    commands[COMMAND_INTS * k + 3] = k * capacity
    commands[COMMAND_INTS * k + 4] = 0


@wp.kernel
def write_instances(
    positions_m: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    shapes: wp.array(dtype=wp.int32),
    sizes_m: wp.array(dtype=wp.vec3),
    albedo_linear_rgb: wp.array(dtype=wp.vec3),
    meshes: int,
    capacity: int,
    commands: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.vec4),
):
    """Each body takes the next slot of its shape's range (shape times capacity onwards),
    counted in its mesh's draw command, and writes four vec4s there: position, its
    orientation, its size and its albedo. A shape with no mesh is not drawn."""
    i = wp.tid()
    shape = shapes[i]
    if shape < 0 or shape >= meshes:
        return
    slot = wp.atomic_add(commands, COMMAND_INTS * shape + 1, 1)
    row = 4 * (shape * capacity + slot)
    p = positions_m[i]
    q = orientations[i]
    s = sizes_m[i]
    c = albedo_linear_rgb[i]
    out[row] = wp.vec4(p[0], p[1], p[2], 1.0)
    out[row + 1] = wp.vec4(q[0], q[1], q[2], q[3])
    out[row + 2] = wp.vec4(s[0], s[1], s[2], 0.0)
    out[row + 3] = wp.vec4(c[0], c[1], c[2], 1.0)

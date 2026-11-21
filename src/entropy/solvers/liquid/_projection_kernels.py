"""Warp kernels of `entropy.solvers.liquid.projection` (ghost-fluid pressure projection).

pyright cannot type Warp's kernel maths: its stubs give 3D thread indices union
types. This file holds kernels only, so the rules below are relaxed here and
nowhere else; the host code in `projection.py` stays fully checked.
"""

# pyright: reportOperatorIssue=false, reportCallIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAssignmentType=false

import warp as wp

AIR = wp.constant(0)
FLUID = wp.constant(1)
SOLID = wp.constant(2)
THETA_MIN = wp.constant(0.01)
"""Smallest θ: a surface closer than 1 % of a cell to a fluid centre is put there."""


@wp.func
def _is_wall(along: int, n: int) -> bool:
    return along == 0 or along == n


@wp.kernel
def face_theta(
    labels: wp.array3d(dtype=wp.int32),
    phi_m: wp.array3d(dtype=float),
    di: int,
    dj: int,
    dk: int,
    theta: wp.array3d(dtype=float),
):
    """θ of each face between a fluid and an air cell: the fraction of the way from the
    fluid centre to the air centre where the level set crosses zero (Enright et al.
    2003). 1 elsewhere, and 1 when the labels and the level set disagree."""
    i, j, k = wp.tid()
    along = i * di + j * dj + k * dk
    n = labels.shape[0] * di + labels.shape[1] * dj + labels.shape[2] * dk
    theta[i, j, k] = 1.0
    if _is_wall(along, n):
        return
    a = labels[i - di, j - dj, k - dk]
    b = labels[i, j, k]
    phi_fluid = wp.float32(0.0)
    phi_air = wp.float32(0.0)
    if a == FLUID and b == AIR:
        phi_fluid = phi_m[i - di, j - dj, k - dk]
        phi_air = phi_m[i, j, k]
    elif a == AIR and b == FLUID:
        phi_fluid = phi_m[i, j, k]
        phi_air = phi_m[i - di, j - dj, k - dk]
    else:
        return
    if phi_fluid < 0.0 and phi_air >= 0.0:
        theta[i, j, k] = wp.clamp(phi_fluid / (phi_fluid - phi_air), THETA_MIN, 1.0)


@wp.kernel
def zero_solid_faces(
    labels: wp.array3d(dtype=wp.int32), di: int, dj: int, dk: int, u: wp.array3d(dtype=float)
):
    """Static solids and the grid's walls do not move: their faces carry no flow."""
    i, j, k = wp.tid()
    along = i * di + j * dj + k * dk
    n = labels.shape[0] * di + labels.shape[1] * dj + labels.shape[2] * dk
    if _is_wall(along, n):
        u[i, j, k] = 0.0
        return
    if labels[i - di, j - dj, k - dk] == SOLID or labels[i, j, k] == SOLID:
        u[i, j, k] = 0.0


@wp.kernel
def right_hand_side(
    labels: wp.array3d(dtype=wp.int32),
    divergence: wp.array3d(dtype=float),
    scale: float,
    rhs: wp.array3d(dtype=float),
):
    """rhs = -(rho/dt) div(u*) on fluid cells (A is about -laplacian, ADR 0016), 0 elsewhere."""
    i, j, k = wp.tid()
    if labels[i, j, k] == FLUID:
        rhs[i, j, k] = scale * divergence[i, j, k]
    else:
        rhs[i, j, k] = 0.0


@wp.kernel
def subtract_gradient(
    labels: wp.array3d(dtype=wp.int32),
    pressure: wp.array3d(dtype=float),
    theta: wp.array3d(dtype=float),
    di: int,
    dj: int,
    dk: int,
    coefficient: float,
    u: wp.array3d(dtype=float),
):
    """u -= dt/(rho dx) (p_upper - p_lower) on faces touching fluid. A missing air pressure
    is the ghost value that puts p = 0 at θ: p_air = p_fluid (θ - 1)/θ."""
    i, j, k = wp.tid()
    along = i * di + j * dj + k * dk
    n = labels.shape[0] * di + labels.shape[1] * dj + labels.shape[2] * dk
    if _is_wall(along, n):
        return
    a = labels[i - di, j - dj, k - dk]
    b = labels[i, j, k]
    if a == SOLID or b == SOLID:
        return
    t = wp.clamp(theta[i, j, k], THETA_MIN, 1.0)
    if a == FLUID and b == FLUID:
        u[i, j, k] = u[i, j, k] - coefficient * (
            pressure[i, j, k] - pressure[i - di, j - dj, k - dk]
        )
    elif a == FLUID and b == AIR:
        u[i, j, k] = u[i, j, k] + coefficient * pressure[i - di, j - dj, k - dk] / t
    elif a == AIR and b == FLUID:
        u[i, j, k] = u[i, j, k] - coefficient * pressure[i, j, k] / t

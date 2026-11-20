"""Weighted Jacobi iteration for the 7-point Poisson stencil.

Solves ∇²p = rhs on a uniform 3D grid of spacing h with the second-order
7-point stencil. Values outside the grid are zero (homogeneous Dirichlet).
That is the only boundary condition so far: solid walls (Neumann) and the
free surface arrive with the pressure solvers of M1 (SPEC 5.2), where this
step serves as the multigrid smoother.

Units follow the caller: with p in Pa and h in m, rhs is in Pa/m².
"""

import warp as wp


@wp.kernel
def _jacobi_step_kernel(
    p: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    h2_m2: float,
    omega: float,
):
    i, j, k = wp.tid()
    nx = p.shape[0]
    ny = p.shape[1]
    nz = p.shape[2]
    total = float(0.0)
    if i > 0:
        total += p[i - 1, j, k]
    if i < nx - 1:
        total += p[i + 1, j, k]
    if j > 0:
        total += p[i, j - 1, k]
    if j < ny - 1:
        total += p[i, j + 1, k]
    if k > 0:
        total += p[i, j, k - 1]
    if k < nz - 1:
        total += p[i, j, k + 1]
    jacobi = (total - h2_m2 * rhs[i, j, k]) / 6.0
    out[i, j, k] = (1.0 - omega) * p[i, j, k] + omega * jacobi


def jacobi_step(
    p: wp.array, rhs: wp.array, out: wp.array, *, h_m: float, omega: float = 1.0
) -> None:
    """One weighted Jacobi sweep from `p` into `out` (same shape, same device).

    `h_m` is the cell size in metres; `omega` the weight (1 is plain Jacobi,
    6/7 the usual smoother weight in 3D).
    """
    wp.launch(
        _jacobi_step_kernel,
        dim=p.shape,
        inputs=[p, rhs, out, h_m * h_m, omega],
        device=p.device,
    )


def jacobi_solve(
    p: wp.array, rhs: wp.array, *, h_m: float, iterations: int, omega: float = 1.0
) -> wp.array:
    """Run `iterations` sweeps starting from `p`; return the array holding the result.

    Ping-pongs between `p` and one scratch array, so `p` is overwritten.
    """
    scratch = wp.empty_like(p)
    current, other = p, scratch
    for _ in range(iterations):
        jacobi_step(current, rhs, other, h_m=h_m, omega=omega)
        current, other = other, current
    return current

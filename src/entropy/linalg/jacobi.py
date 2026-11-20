"""Weighted Jacobi iteration for the 7-point Poisson stencil.

Solves ∇²p = rhs on a uniform 3D grid of spacing h with the second-order
7-point stencil. Values outside the grid are zero: the zero sits one full
spacing h beyond the last unknown, as for unknowns on the nodes of a grid
whose boundary nodes are fixed (homogeneous Dirichlet). That is the only
boundary condition so far. On the cell-centred grid of SPEC 5.2, a boundary
on a cell face would make this placement first-order; walls (Neumann) and the
free surface arrive with the M1 pressure solvers, where this step serves as
the multigrid smoother.

Units follow the caller: with p in Pa and h in m, rhs is in Pa/m².
"""

import warp as wp

CHUNK = 8
"""Cells each thread updates along i, keeping p[i-1] and p[i] in registers.

About 1.7x (192³) to 2.4x (128³) faster than one thread per cell, and as fast as
a pure streaming pass with the same traffic: docs/benchmarks/2026-11-20-m0-laptop.md.
"""


@wp.kernel
def _jacobi_step_kernel(
    p: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    h2_m2: float,
    omega: float,
    chunk: int,
):
    # 3D launch: tid() returns three indices, which Warp's stubs do not express.
    c, j, k = wp.tid()  # pyright: ignore[reportGeneralTypeIssues, reportAssignmentType]
    nx = p.shape[0]
    ny = p.shape[1]
    nz = p.shape[2]
    i0 = c * chunk
    i1 = wp.min(i0 + chunk, nx)
    below = wp.float32(0.0)
    if i0 > 0:
        below = p[i0 - 1, j, k]
    centre = p[i0, j, k]
    for i in range(i0, i1):
        above = wp.float32(0.0)
        if i < nx - 1:
            above = p[i + 1, j, k]
        total = below + above
        if j > 0:
            total += p[i, j - 1, k]
        if j < ny - 1:
            total += p[i, j + 1, k]
        if k > 0:
            total += p[i, j, k - 1]
        if k < nz - 1:
            total += p[i, j, k + 1]
        jacobi = (total - h2_m2 * rhs[i, j, k]) / 6.0
        out[i, j, k] = (1.0 - omega) * centre + omega * jacobi
        below = centre
        centre = above


def _check(p: wp.array, rhs: wp.array, out: wp.array) -> None:
    if not p.shape == rhs.shape == out.shape:
        raise ValueError(f"shapes differ: p {p.shape}, rhs {rhs.shape}, out {out.shape}")
    if not p.device == rhs.device == out.device:
        raise ValueError(f"devices differ: p {p.device}, rhs {rhs.device}, out {out.device}")
    if out.ptr in (p.ptr, rhs.ptr):
        raise ValueError("out must not alias p or rhs: a Jacobi sweep reads the old values")


def jacobi_step(
    p: wp.array, rhs: wp.array, out: wp.array, *, h_m: float, omega: float = 1.0
) -> None:
    """One weighted Jacobi sweep from `p` into `out`.

    The three arrays have the same shape and device, and `out` is a separate
    array. `h_m` is the cell size in metres; `omega` the weight (1 is plain
    Jacobi, 6/7 the usual smoother weight in 3D; 0 < omega ≤ 1 converges).
    """
    _check(p, rhs, out)
    nx, ny, nz = p.shape
    wp.launch(
        _jacobi_step_kernel,
        dim=((nx + CHUNK - 1) // CHUNK, ny, nz),
        inputs=[p, rhs, out, h_m * h_m, omega, CHUNK],
        device=p.device,
    )


def jacobi_solve(
    p: wp.array,
    rhs: wp.array,
    *,
    h_m: float,
    iterations: int,
    omega: float = 1.0,
    scratch: wp.array | None = None,
) -> wp.array:
    """Run `iterations` sweeps starting from `p`; return the array holding the result.

    Ping-pongs between `p` and `scratch` (allocated if not given), so both are
    overwritten and the result is one of them.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be ≥ 0, got {iterations}")
    current, other = p, scratch if scratch is not None else wp.empty_like(p)
    for _ in range(iterations):
        jacobi_step(current, rhs, other, h_m=h_m, omega=omega)
        current, other = other, current
    return current

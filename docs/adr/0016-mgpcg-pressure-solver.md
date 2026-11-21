# ADR 0016: The MGPCG pressure solver

- Status: accepted
- Date: 2026-11-21

## Context

The liquid (M1) and the gas (M3) solve a pressure Poisson equation every substep (SPEC 5.2). SPEC 3 budgets the whole liquid at 9 ms per frame, and V-LIQ-05 asks for second order. Issue #46 asks for conjugate gradient preconditioned by multigrid (McAdams, Sifakis, Teran 2010), running without a CPU read-back per iteration.

## Decision

- **One interface for every boundary.**
  - Cells are labelled fluid, air or solid; the grid boundary is a wall (ADR 0015).
  - The operator is (A p)_c = (diag_c·p_c − Σ w·p_n) / Δx² over fluid neighbours. The face weights w are 0 or 1 today and will be fractions for Batty's solids. diag adds w/θ for each air face, so θ = ½ gives the second-order free surface of Enright et al. (2003).
  - When all weights are 0 or 1, the kernels take them from the labels instead of reading three arrays.
- **The V-cycle** is symmetric, so CG stays valid:
  - Jacobi smoothing, damped at ω = 6/7, with 2 sweeps before and 2 after.
  - Coarse cells are labelled air if any child is air, else fluid if any child is, else solid. Coarse grids use unit weights and θ = 1.
  - Prolongation is trilinear between cell centres, and restriction is R = Pᵀ/8 by scattering, so it is P's exact transpose.
  - Coarsening goes down to axes of 4 cells, with 16 Jacobi sweeps on the coarsest grid.
- **CG's scalars stay on the device**, in float64. Dot products are two-stage: partial sums, then a few atomics.
- **Fixed batches, not conditional graph nodes.** On CUDA the solve start and each batch of 4 iterations are captured once as CUDA graphs and replayed. The residual is read back once per batch, which makes about 4 synchronisations per solve. Iterations past convergence only lower the residual further. Conditional nodes would remove the read-backs; #36 measures them first.
- **The arrays stay the same objects.** The graphs hold pointers to the labels, weights, diagonals and work vectors. The liquid relabels its cells every substep by writing into `system.labels` and calling `Mgpcg.refresh()`, which recomputes every level in place. Callers pass the same `b` and `x` at every solve. A new array means a new capture, which costs several milliseconds.
- **Named limit: every body of fluid must touch air.** Otherwise A is singular. A sealed pocket does not converge: `converged` is false after `max_iterations`. An isolated fluid cell gets an arbitrary pressure through the diagonal's 1e-12 floor. The liquid (#49) must label such pockets or remove the constant mode before solving.
- **Measured** on the laptop (unofficial: 20 s warm-up). Each substep refreshes the operator in place, then solves from zero to 10⁻⁵ with a random right-hand side, in 12 iterations every time. Water at 2 m/s takes 2 substeps per 1/30 s frame:

  | Grid | Per substep | Per frame |
  | --- | --- | --- |
  | Test Tank (72 × 72 × 40) | 4.0 ms | 7.9 ms |
  | Full house (160 × 128 × 96) | 13.8 ms | 27.6 ms |
  | 128³ | 14.7 ms | 29.5 ms |

  - Every grid is at risk. On the Test Tank, the solve alone takes 88 % of the liquid's 9 ms budget. On the house grid it exceeds the whole 22 ms simulation budget, and SPEC 3's estimate of 5.0–8.7 ms per substep was about half the measured cost.
  - These solves start from zero. The liquid starts from the previous pressure, which should cut the iteration count; #50 measures it.

## Alternatives considered

- **Red-black Gauss–Seidel smoothing:** fewer iterations, but a symmetric V-cycle needs the colours in opposite orders before and after, and it doubles the kernels per sweep. Jacobi is simpler, and on a GPU it is one kernel per sweep.
- **Piecewise-constant prolongation:** cheaper, but weaker for cell-centred Poisson problems.
- **A float32 dot product** (`warp.utils.array_inner`): faster, but CG's scalars lose digits near 10⁻⁵ on 2 million cells.
- **Conditional graph nodes (`wp.capture_while`):** the target once #36 has measured them.

## How we would know it is wrong

- V-LIQ-02 or V-LIQ-04 fail because the solve stalls above its tolerance on real liquid shapes (thin sheets, many air pockets).
- At the M1 gate (#57), on the Test Tank with 1 M particles and a warm start from the previous pressure, the pressure solve takes more than 5 ms per frame, leaving under 4 ms of the liquid's 9 ms for transfers and advection.

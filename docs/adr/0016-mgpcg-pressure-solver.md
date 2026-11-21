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
- **Fixed batches, not conditional graph nodes.** On CUDA the solve start and each batch of 4 iterations are captured once as CUDA graphs and replayed. The residual is read back once per batch, and iterations past convergence change nothing. Conditional nodes would remove the read-backs; #36 measures them first.
- **Measured** (laptop, unofficial: 20 s warm-up), from zero to 10⁻⁵ with a random right-hand side, 12 iterations every time:
  - Test Tank grid (72 × 72 × 40): 3.4 ms;
  - full house grid (160 × 128 × 96): 12.9 ms;
  - 128³: 14.0 ms.

  The house grid is above SPEC 3's 5.0–8.7 ms estimate, so SPEC 3 marks it at risk. The Test Tank, which M1 needs, is not.

## Alternatives considered

- **Red-black Gauss–Seidel smoothing:** fewer iterations, but a symmetric V-cycle needs the colours in opposite orders before and after, and it doubles the kernels per sweep. Jacobi is simpler, and on a GPU it is one kernel per sweep.
- **Piecewise-constant prolongation:** cheaper, but weaker for cell-centred Poisson problems.
- **A float32 dot product** (`warp.utils.array_inner`): faster, but CG's scalars lose digits near 10⁻⁵ on 2 million cells.
- **Conditional graph nodes (`wp.capture_while`):** the target once #36 has measured them.

## How we would know it is wrong

- V-LIQ-02 or V-LIQ-04 fail because the solve stalls above its tolerance on real liquid shapes (thin sheets, many air pockets).
- At the M1 gate, with a warm start from the previous pressure, the liquid's pressure solve on the Test Tank still takes most of its 9 ms frame budget.

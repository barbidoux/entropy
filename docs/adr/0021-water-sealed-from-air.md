# ADR 0021: Water sealed from air

- Status: accepted
- Date: 2026-11-21

## Context

A body of water that touches no air makes the pressure system singular (ADR 0016):
- the pressure is fixed only up to a constant;
- a solution exists only if the right-hand side sums to zero over the body.

Inside closed walls the sum is zero in exact arithmetic (issue #69). In practice three
things break it:

- **The volume correction.** It gives the projection a target divergence (ADR 0018), and
  that target's mean need not be zero.
- **Float32 rounding of the right-hand side.** It is never exactly consistent.
- **The V-cycle.** Its Jacobi sweeps and its singular coarsest grid put constants into
  the preconditioned residual z. In float32, A p never sums to exactly zero either.
  Together these make CG wander along the constant: the pressure drifts, and the solve
  slows or fails.

Measured before this decision, with water filled to the lid:

| Grid | Device | What happened |
| --- | --- | --- |
| 12 × 10 × 16, lattice or jittered | CPU and CUDA | runs to the 200-iteration cap, pressures up to 10¹⁰ Pa, particles at 8 to 546 m/s |
| 72 × 72 × 40 (the Test Tank's grid), 600 frames | CUDA | 1 to 4 solves at the cap, 60 ms each |
| 32 × 32 × 40 still column, warm start | CPU | 16 to 200 iterations, pressure off by up to 4·10⁶ Pa |

In M1 the only solids are the grid's walls. This case therefore happens only when water
fills the whole grid, for example when the flood source or the hose runs long enough.

## Decision

MGPCG handles a grid with no air itself (`entropy.linalg.mgpcg`). The projection needs
nothing extra.

- **Detection.** The start of the solve flags any air cell on the device. The solve's
  first read-back (b·b and r·r) carries that flag. Only when there is no air does the
  solve take the sealed path, at the cost of one more read-back. With air, it costs one
  store per air cell. Measured on CUDA at 72 × 72 × 40 with air, `project()` takes
  2.69 ms against 2.69 ms before, and a 4-iteration batch 924 µs against 927 µs.
- **Sealed path, at the start.**
  - The fluid cells are counted.
  - x loses its mean over the fluid cells, so the answer's mean is 0.
  - r = b − A x loses its mean. b's part along the constant cannot be reached, so the
    answer is the least-squares one: the answer for b minus its mean.
  - z = M r and p = z are computed again.
- **Sealed path, every iteration.** z loses its mean after the V-cycle, before r·z, in
  separate CUDA graphs for the sealed case. Without this, from rest the solve took up to
  52 iterations on a 32 × 32 × 40 grid, or did not converge. Removing the mean of the
  first V-cycle's z as well changes the level by only about 10⁻⁸ of the pressure range,
  so that step is left out.
- **Precision.** Means are summed and subtracted in float64.

Measured after this decision:
- The 32 × 32 × 40 column converges in 4 iterations from the last substep and in 8 from
  rest.
- 600 frames at 72 × 72 × 40 on CUDA, lattice or jittered: every substep's solve
  converges in at most 8 iterations (median 4), and the water stays under 0.4 mm/s.
- A closed box full of water under gravity has the hydrostatic gradient within 0.1 %.

Named limits:
- **Water sealed by static solids while air remains elsewhere in the grid** is not
  detected.
- **Several bodies separated by solids, in a grid with no air,** are not handled either:
  the global mean is not each body's mean. Measured on 16 × 8 × 16 with a solid plane
  and a ±0.01/s target per body, the solve ran to its cap with pressures of 10⁹ to
  10¹² Pa.
  The Test Tank cannot produce either case. Setups with static solids (`solid_phi_m`)
  can, and M2's moving solids will need per-body sums (connected components).
- **The sealed pressure level (mean 0) is a convention.** Nothing reads it until solids
  are coupled (M2).

## Alternatives considered

- **Removing the means in the projection, around the solve.** This was the first
  version of this decision. It fixed the 12 × 10 × 16 grid, but the Test Tank's grid
  still hit the cap: the trouble is inside CG. Its slot sums also cost 161 µs per substep
  with or without air.
- **Pinning one fluid cell to p = 0 when there is no air.** The system becomes definite.
  But the velocity update would treat that cell as air, and the multigrid would converge
  slowly towards a single Dirichlet point.
- **Tools that stop filling at the rim.** They prevent the case but do not protect
  other sealed shapes. They may still be added for play.
- **Per-body components every substep.** This means a labelling pass over the grid. It
  is not needed while the grid's walls are the only solids.

## How we would know it is wrong

- A grid full of water shows motion, or one of its pressure solves hits the iteration
  cap.
- Static solids that trap water (M2) produce solves that do not converge while air
  remains.
- `project()` with air costs measurably more than it did before this decision.

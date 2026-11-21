# ADR 0021: Water sealed from air

- Status: accepted
- Date: 2026-11-21

## Context

A body of water that touches no air makes the pressure system singular (ADR 0016): the
pressure is fixed only up to a constant, and a solution exists only if the right-hand
side sums to zero over the body. Inside closed walls it does in exact arithmetic (issue
#69). Two things break that in practice:

- The volume correction gives the projection a target divergence (ADR 0018), whose
  mean need not be zero.
- Float32 rounding. Measured on a 12 × 10 × 16 grid full of water: jittered particles
  converge in 8 iterations, but a lattice of particles runs to the 200-iteration cap
  with pressures near 10⁹ Pa and velocities of 5 m/s.

In M1 the only solids are the grid's walls, so this happens only when water fills the
whole grid (the flood source or the hose run long enough).

## Decision

- When no cell of the grid is air, the projection removes the right-hand side's mean
  over the fluid cells before the solve, which makes the system consistent.
- After the solve, it sets the pressure's level so that its mean over the fluid cells
  is 0.
- Both use device sums spread over 1,024 slots and a one-thread kernel that applies
  them only when the air count is 0. No sync point is added, and grids with air are
  unchanged.

Measured: a closed box full of water under gravity converges, with the hydrostatic
gradient within 0.1 % and still water. A grid of lattice or jittered particles filled
to the lid stays under 5 mm/s for 1 s.

Named limits:
- A body sealed by static solids while air remains elsewhere in the grid is not
  detected; M2's moving solids will need per-body sums (connected components).
- A sealed body's pressure level (mean 0) is a convention. Nothing reads it until
  solids are coupled (M2).

## Alternatives considered

- **Tools that stop filling at the rim:** they prevent the case, but they would not
  protect against rounding in other sealed shapes. They may still be added for play.
- **Per-body components every substep:** a labelling pass of the grid. Not needed while
  the grid's walls are the only solids.

## How we would know it is wrong

- A grid full of water shows motion, or the pressure solve hits its iteration cap.
- Static solids that trap water (M2) produce non-converging solves while air remains.

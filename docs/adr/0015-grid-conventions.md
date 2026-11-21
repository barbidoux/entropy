# ADR 0015: Grid conventions: MAC staggering, clamped sampling, closed boundary

- Status: accepted
- Date: 2026-11-21

## Context

The liquid (M1) and later the gas (M3) share the grid of `entropy.fields.mac`
(issue #45). Three conventions decide how every solver built on it behaves at the
edges, and changing them later would touch every kernel:

1. where values live;
2. what a point outside the grid reads;
3. what the grid boundary is.

## Decision

- **Staggering (MAC):**
  - Pressure and other scalars sit at cell centres, origin + (i + ½, j + ½, k + ½)·Δx.
  - Each velocity component sits on the faces normal to its axis: u at origin + (i, j + ½, k + ½)·Δx, with (nx + 1, ny, nz) values, and likewise v and w.
  - This is the standard layout for incompressible solvers (Bridson 2015): the divergence and the pressure gradient are second-order central differences, with no checkerboard mode.
- **Sampling outside the grid clamps to the nearest samples.**
  - Each component clamps to its own sample range, with no extrapolation. The coordinate is clamped before it is converted to an integer, so infinities and huge values give the same result on CPU and CUDA.
  - A NaN position reads NaN, so the NaN guard (SPEC 5.9) sees the upstream bug instead of a plausible value.
- **The grid boundary is a closed wall.**
  - The gradient is 0 on boundary faces, so `divergence(gradient(p))` is the homogeneous-Neumann Laplacian.
  - Open boundaries, the free surface and solids are stated by the pressure projection (#49) through cell labels and face weights, not by the grid.

## Alternatives considered

- **Collocated grid (everything at cell centres):** simpler interpolation, but it needs Rhie–Chow-style corrections against checkerboard pressure. No graphics liquid solver of the APIC/FLIP family uses it.
- **Extrapolation, or zero, outside the grid:**
  - Linear extrapolation can return speeds the grid never held, which makes particles near the edge explode.
  - Zero velocity outside makes the edge look like a wall even where the map is open.
  - Clamping is the least surprising choice, and the domain is sized so the liquid stays inside it.
- **Dirichlet boundary (as in `entropy.linalg.jacobi`):** wrong for a tank, whose walls must stop the flow. The free surface, the only Dirichlet part, is not on the grid boundary.

## How we would know it is wrong

Particles that reach the grid edge gain energy, or stick, in V-LIQ-02 (sloshing) or V-LIQ-03 (dam break). Or the pressure projection (#49) needs boundary terms that the closed-wall gradient cannot express without special cases.

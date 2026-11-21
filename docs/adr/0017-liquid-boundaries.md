# ADR 0017: Liquid boundaries: domain walls, voxelized static solids, ghost-fluid free surface

- Status: accepted
- Date: 2026-11-21

## Context

The pressure projection (SPEC 5.2, issue #49) needs three boundaries: the walls that
hold the water, solids inside the water, and the free surface. Two facts shape them:

- The Test Tank's glass is 2 cm thick, against 6.25 cm cells. A wall thinner than a
  cell can miss every cell centre (`entropy.world.sdf`, named limit), so water sampled
  that way would leak through it.
- V-LIQ-01 asks for hydrostatic pressure within 2 % at every depth, down to the first
  cell under the surface (3 cm deep). That needs the surface located to a fraction of a
  millimetre, so a second-order free surface.

## Decision

- **The tank's walls are the grid's walls.** The liquid grid of the Test Tank spans its
  cavity, extended upward. Its sides and bottom are the grid boundary, which is a
  closed wall (ADR 0015): exact, whatever the glass thickness.

  Named limit: water cannot leave the tank over the rim. It stays inside the cavity's
  column. Spilling onto the floor needs a domain larger than the tank, with the glass
  as a cut-cell or variational solid (Batty et al. 2007), which comes with the moving
  solids of M2.
- **Static solids inside the domain are voxelized.** A cell whose centre is inside a
  solid (signed distance below 0) is labelled solid, and the faces touching it carry
  no flow. Named limit: solids thinner than a cell can be missed. The Test Tank has
  none inside its cavity.
- **The free surface is second order.**
  - The liquid gives the projection a level set `phi_m` at cell centres, negative in
    the water.
  - Each face between a fluid and an air cell gets θ = φ_fluid / (φ_fluid − φ_air),
    clamped to [0.01, 1], with p = 0 at that fraction of the way (Enright et al. 2003).
  - MGPCG takes one θ per face (ADR 0016); the velocity update uses the matching
    ghost pressure p_air = p_fluid (θ − 1)/θ.
  - Where the labels and the level set disagree, θ = 1 (first order).
- **Inputs, not reconstruction.** The projection receives labels and the level set. The
  liquid builds both from its particles every substep (#50), so V-LIQ-01 and V-LIQ-04
  test the projection itself, with a known surface.

## Alternatives considered

- **Voxelizing the glass into the grid:** it leaks at 6.25 cm cells; it works only
  with walls of at least two cells (12.5 cm of glass).
- **Batty's variational face weights now:** they handle thin and slanted solids, but
  the 2 cm glass still needs weights sampled finely along each face. They cost 12–18
  sweep equivalents per iteration (SPEC 3), and they belong with the moving solids of
  M2, which need them anyway.
- **A first-order free surface (θ = 1 or ½):** pressure is off by up to half a cell
  times ρg. Near the surface that is far beyond V-LIQ-01's 2 %.

## How we would know it is wrong

- With a level set reconstructed from particles (#50), a still tank (V-LIQ-01 run
  through the full step) drifts or shows surface ripples above 1 mm.
- Or Matt wants water to spill out of the tank in M1. Then the domain must outgrow the
  tank, and the glass must become a solid the grid can see.

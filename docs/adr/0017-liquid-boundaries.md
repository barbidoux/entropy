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

  Named limits:
  - Water cannot leave the tank over the rim. It stays inside the cavity's column.
    Spilling onto the floor needs a domain larger than the tank, and glass the grid
    can see: at least two cells thick, a finer grid, or faces closed wherever a thin
    solid crosses the segment between two cell centres. M2 decides, with the moving
    solids.
  - The grid's top is a closed wall too. A splash that reaches it stops there instead
    of rising further, so the grid extends well above the rim.
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
  - MGPCG floors θ at 10⁻³ as a safety net for any caller (ADR 0016); the projection's
    own floor of 0.01 is the one that applies to the liquid.
- **Inputs, not reconstruction.** The projection receives labels and the level set. The
  liquid builds both from its particles every substep (#50), so V-LIQ-01 and V-LIQ-04
  test the projection itself, with a known surface.
- **Sealed pockets are deferred** (#69). A body of water that touches no air makes the
  pressure system singular (ADR 0016). In M1 the only solids are the grid's walls, so
  that happens only if water fills the whole grid. Until #69, the projection does not
  detect it: the solve reports `converged = False`.

## Alternatives considered

- **Voxelizing the glass into the grid:** it leaks at 6.25 cm cells; it works only
  with walls of at least two cells (12.5 cm of glass).
- **Batty's variational weights now** (Batty et al. 2007): they weight each face by
  its open fraction, which handles slanted solids. They do not stop the leak: 2 cm of
  glass lying between two faces 6.25 cm apart cuts no face, so its weights stay 1
  (about 0.68 as volume fractions), and water leaks as with voxels. They also raise
  an MGPCG iteration from about 12 to about 18 sweep equivalents (+50 %, M0 laptop
  report, `docs/benchmarks/2026-11-20-m0-laptop.md`): SPEC 3 estimates 7.1–12.7 ms per
  substep against 5.0–8.7 ms on the house grid. They belong with the moving solids of
  M2, which need them anyway.
- **A first-order free surface:** with θ = 1 (surface at the air cell's centre) the
  pressure is off by up to one cell times ρg, with θ = ½ (surface on the face) by up
  to half a cell. Near the surface either is far beyond V-LIQ-01's 2 %.

## How we would know it is wrong

- With a level set reconstructed from particles (#50), a still tank (V-LIQ-01 run
  through the full step) drifts or shows surface ripples above 1 mm.
- Or Matt wants water to spill out of the tank in M1. Then the domain must outgrow the
  tank, and the glass must become a solid the grid can see.

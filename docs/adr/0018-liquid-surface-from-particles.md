# ADR 0018: The liquid's surface and volume come from its particles' mass

- Status: accepted
- Date: 2026-11-21

## Context

The pressure projection takes cell labels and a level set `phi_m` as inputs (ADR 0017).
The liquid must build both from its particles every substep (issue #50). Three facts,
measured during #50 and its review, shape how:

- ADR 0017's falsifier: a still tank run through the full step must not drift or show
  ripples above 1 mm. Particles are seeded with a random jitter, so the surface must
  not follow each particle.
- A first version used Zhu & Bridson's (2005) level set: φ = |x − x̄| − r, with x̄ the
  weighted mean of the particles within R. Calibrated on a seeded lattice, it was
  right at rest. Once the water had sloshed and the particles were mixed, it put the
  surface 2.5 cm (1.6 particle spacings) below the top particles on a 3.1 cm grid, so
  the pressure treated the top layer as air and the visible volume fell by 3–5 % in
  100 s.
- Particles also drift apart or together over long runs (clustering, a known trait of
  particle-in-cell liquids). The reviewer measured 8.00 particles per bulk cell
  rising to 8.76 after 6,000 frames.

## Decision

- **Density.** Each cell's density is the mean of its six face masses from the
  particle-to-grid transfer (quadratic B-splines, a partition of unity), divided by
  the same quantity for still water filling every cell outside the solids. That rest
  field is measured once, with the same transfer, so the density is 1 in still water
  against walls and solids too. Measured noise for jittered particles: 2.6 % per cell;
  bias under 0.1 %, and none on a lattice.
- **Surface.** The surface is where the density is ½. The particles' mass is smoothed
  by a symmetric kernel, so its half height lies where the water ends, however the
  particles are arranged. φ = (½ − density) · dx / (2/3) is close to the distance to
  the surface near it (the density rises by 2/3 per cell across a flat surface).
  Labels: solid where the static solids' distance is negative, fluid where φ < 0,
  else air. Labels and φ agree, so every fluid-air face gets its θ.
- **Volume correction**, a target divergence the projection gives each fluid cell
  instead of zero, with two terms:
  - Local density: rate · (density − 1) in cells with no air within two cells (nearer
    the surface the density reads low for lack of water), at a rate of 0.1 per
    substep. Crowded cells expand, sparse ones contract.
  - Volume control (Kim et al. 2007): the same in every fluid cell, proportional and
    integral in the relative error between the level set's volume and the volume of
    the particles inside it. Gains 1.2 /s and 0.36 /s² (critically damped), a dead
    band of 0.25 % (above the measure's 0.075 % bias on still water, so still water
    stays still) and an integral bounded at 0.02 s. Particles in flight do not count
    until they join the water.
- **Velocity extrapolation.** After the projection, two layers of air faces take the
  mean of their valid neighbours (Bridson 2015), then wall and solid faces are set to 0
  again. The quadratic B-spline reaches 1.5 cells, so particles near the surface read
  sound velocities.
- **Motion.** Particles move with the midpoint rule through the grid velocity. They are
  then kept 10⁻⁴ cell inside the grid, whose walls are the tank's (ADR 0017), and
  pushed out of static solids along the distance's gradient. A non-finite position is
  left as it is, for the NaN guard.

Measured with this decision (16 × 4 × 16 sloshing tank unless noted):
- still water, jittered, 800 frames: volume unchanged, particles at rest (0.4 mm/s);
- sloshing, 2,000 frames: the one-second mean volume stays within 0.6 %; 8,000 frames:
  within 0.7 %; a 2× finer grid over 6,000 frames: within 0.43 %.

Named limits:
- Water thinner than about a cell (a lone droplet, a thin sheet) has a density below ½
  everywhere, so the pressure ignores it: it falls freely and moves with the
  extrapolated velocity until it joins the water again.
- The volume correction is a numerical correction, not physics: it moves water a few
  millimetres per second at most to undo the method's own drift. Errors smaller than
  its dead band (0.25 %) are left.

## Alternatives considered

- **Zhu & Bridson's level set** (the first version): smooth and cheap, but its surface
  depends on how the particles are arranged, and sank below mixed particles.
- **Union of particle spheres** (φ = distance to the nearest particle − r): measured on
  the jittered still tank during #50, the surface was lumpy and sat 12 mm low.
- **Kernel weights for the density** (Zhu & Bridson's): 12 % noise per cell and a 1.6 %
  bias on a lattice, which would set the equilibrium volume.
- **Only the local density correction**: it stops bulk clustering but not the loss at
  the surface (−2.5 % in 2,000 frames of sloshing).
- **Implicit density projection** (Kugelstadt et al.): a second pressure solve per
  substep; more exact, twice the cost of the solve that is already at risk (SPEC 3).

## How we would know it is wrong

- V-LIQ-02, or the M1 soak's 10-minute run of the same measure, drifts beyond 1 %.
- Still water shimmers in play, or moves faster than a few millimetres per second.
- V-LIQ-03's front runs off Martin & Moyce's records because the surface lags or leads.

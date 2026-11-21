# ADR 0018: The liquid's surface comes from its particles (Zhu & Bridson level set)

- Status: accepted
- Date: 2026-11-21

## Context

The pressure projection takes cell labels and a level set `phi_m` as inputs (ADR 0017).
The liquid must build both from its particles every substep (issue #50). Two facts
shape how:

- ADR 0017's falsifier: a still tank run through the full step must not drift or show
  ripples above 1 mm. Particles are seeded with a random jitter, so the surface must
  not follow each particle.
- The second-order free surface needs φ accurate to a fraction of a millimetre near the
  surface, and labels that agree with φ, so that every fluid-air face gets its θ.

## Decision

- **Level set.** At each cell centre x, φ = |x − x̄| − r (Zhu & Bridson 2005):
  - x̄ is the mean of the particles within R of x, weighted by k(s) = (1 − s²)³ with
    s = |x − x_p| / R;
  - R = 2 particle spacings (one cell at 8 particles per cell);
  - r = 0.75 spacing. It is calibrated so that a flat layer of particles puts the
    surface half a spacing above its top particles, where the water ends. On the still
    tank this holds within 1 mm.
  - A cell with no particle within R gets φ = 3 cells (far in the air).
- **Labels from the level set.** A cell is solid where the static solids' distance is
  negative, else fluid where φ < 0, else air. Labels and φ then agree.
- **Velocity extrapolation.** After the projection, two layers of air faces take the
  mean of their valid neighbours (Bridson 2015). The quadratic B-spline reaches 1.5
  cells, so particles near the surface read sound velocities.
- **Motion.** Particles move with the midpoint rule through the grid velocity. They are
  then kept 10⁻⁴ cell inside the grid, whose walls are the tank's (ADR 0017), and
  pushed out of static solids along the distance's gradient.

Named limits:

- φ is a distance only near the surface. Far from it, it serves only to say "air" or
  "water".
- Water thinner than about one cell (a lone droplet, a thin sheet) can have φ > 0 at
  every cell centre. The pressure then ignores it: it falls freely and moves with the
  extrapolated velocity until it joins the water again.
- The level set does not conserve volume; the particles do. V-LIQ-02 measures the
  volume from the level set and bounds its drift.

## Alternatives considered

- **Union of particle spheres** (φ = distance to the nearest particle − r): measured on
  the jittered still tank, the surface was lumpy and its mean sat 12 mm low. The
  pressure then pushed the water around, so it never stayed still.
- **Cells that hold particles are fluid** (first-order labels): the surface is off by
  up to a cell, which ADR 0017 already rejected for the pressure.
- **Particle level set** (Enright et al. 2002): more accurate thin features, but it
  carries a second particle set and a level set advected on the grid. Not needed for
  the Test Tank; to reconsider if thin sheets matter for Matt.

## How we would know it is wrong

- V-LIQ-02 (sloshing) loses more than 1 % of the volume over 2000 steps.
- V-LIQ-03 (dam break): the front runs off Martin & Moyce's curve because the surface
  lags or leads.
- In play, Matt sees a still tank shimmer, or splashes vanish and reappear.

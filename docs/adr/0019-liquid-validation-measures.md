# ADR 0019: How V-LIQ-02 and V-LIQ-03 measure the water

- Status: accepted
- Date: 2026-11-21

## Context

V-LIQ-02 (volume conservation) and V-LIQ-03 (dam break against Martin & Moyce 1952)
compare the liquid's full step (issue #50) with a law and an experiment. SPEC 9.4 gave
the scenarios and tolerances, not how to measure. A first version of this decision was
rejected in review: it fitted a clock shift per run, which let solvers with gravity
off by 20 % pass, and it explained the shift by a gate that never existed. The physics
auditor then read the paper and the literature. What is known:

- **The release.** A waxed-paper diaphragm, held by beeswax on a heater strip, freed
  when a 36 V battery bank was switched through the strip (pp. 313–314). Filmed at
  about 300 frames per second.
- **The clock is a convention.** The motion's start could not be recorded. Every
  record was rescaled so that T = 1.19 when Z = 1.44, about the time since the current
  was applied (p. 314, Table 2's footnote). Table 6 bounds the first motion at T ≤ 0.23
  after the current (12 ms for the 2¼ in column). Martin & Moyce themselves shifted
  their clock to compare with theory (0.10 and 0.37, pp. 321 and 324).
- **Independent solvers need the same delay.** Released at once, they lead the
  records by about ΔT = 0.25: Landet, Mardal & Mortensen (2018), two-phase
  Navier–Stokes, 0.20–0.30; Hirt & Nichols (1981), SOLA-VOF, plotted the records
  about 0.29 earlier than Table 2 without saying so; ENTROPY, converged, 0.26.
- **The records scatter.** At Z = 3.00 the nine records of Table 2 run from T = 2.38 to
  2.66 (about ±5–7 % in Z); their mean is good to about ±2 %. Lobovsky et al. (2014)
  find dam-break experiments differ most for t* < 1, where t* = t·√(g/H) with H = 2a,
  that is T < 2.
- **The level set's volume wobbles.** With a few particles per cell it moves by a few
  tenths of a per cent from frame to frame. Sampling it at n³ points per cell snaps a
  flat surface to 1/n of a cell: with 4³ points, up to ±2 % of the test tank's depth;
  with 8³, up to ±1 % (its water at rest, 6.42 cells deep, reads −0.67 %). Since the
  drift is measured against the first second's mean, the snap of one surface height
  cancels; the first frame is checked against the particles' volume within 1 %.

Not known:
- the exact delay of the front records, since their origin is only roughly the heater
  current and Table 6 bounds the first visible motion, not the diaphragm clearing;
- how the diaphragm shaped the flow before T ≈ 1.2;
- whether the remaining 0–5 % lead comes from the free-slip floor, the scale (a 5.7 cm
  column against the game's 1 m one), surface tension or the measure. Larger columns
  run faster in other experiments (Lobovsky et al. 2014, Table 1).

## Decision

- **V-LIQ-02.**
  - A tank released with a tilted surface sloshes for 33 s of simulated time (2,000
    frames of 1/60 s at 1×).
  - The particle count does not change.
  - Volume: the share of 8³ points per cell where the level set, interpolated
    between cell centres, is negative. The first frame's volume is within 1 % of the
    particles'.
  - Drift: the largest change of the one-second mean from the first second's, at most
    1 %.
  - The M1 soak runs the same measure for 10 minutes, with violent motion.
- **V-LIQ-03.**
  - A column a = 1 m wide and 2 m high, 16 cells across (the game's cells), released at
    once.
  - Front: where the level set crosses zero along the floor's row of cells, averaged
    across the tank's width (a lone particle ahead of it is spray, SPEC 5.2).
  - Data: the means of the 2¼ in records in Table 2 of the paper
    (`content/validation/martin_moyce_1952.toml`).
  - The simulated clock is delayed by a fixed ΔT = 0.25, never fitted per run, because
    the paper's clock starts before the water moved.
  - Every tabulated mean with 1.19 ≤ T ≤ 3 (11 points) must be within 7 %.

Measured with this decision:
- V-LIQ-02 on CPU: the one-second mean stays within 0.6 %.
- V-LIQ-03 at 16 cells on CPU: −3.2 to +1.4 %; at 32 cells: −2.0 to +3.3 %. Against the
  1⅛ in column (not asserted): −7.2 to +1.4 %.
- What V-LIQ-03 catches: gravity × 0.8 reaches −13.3 % and gravity × 1.2 +9.3 %, so
  both fail by 2 points or more; gravity × 0.9 (−7.9 %) fails, gravity × 1.1 (+4.3 %)
  passes; velocity damped by 0.5 % per substep fails. It does not separate a
  first-order free surface, PIC or forward Euler from the right method: the full-step
  tests do (a rough surface settling for the first two, a solid-body rotation keeping
  its radius for the third).

## Alternatives considered

- **No delay:** a correct instant release leads the records by 13–18 % at every
  resolution, for a cause in the paper's clock, not the solver.
- **A delay fitted per run** (the rejected first version): it absorbs gravity errors of
  ±20 %.
- **PySPH's digitisation of Figure 3:** for the 2¼ in column, within about 0.04 in T
  of Table 2 but about 2 % low in Z at T = 2.0–2.55; for the 1⅛ in column one point is
  0.12 later than the table.
- **A 10 % tolerance:** gravity × 1.2 would pass.
- **Volume per frame within 1 %:** it fails on the measure's wobble, not on lost
  water.

## How we would know it is wrong

- A reading of the paper, or a later experiment, puts the front records' clock origin
  far from 0.25 before the motion.
- The converged delay of the solver moves away from 0.25 when the method changes.
- The M1 soak's 10-minute volume run drifts beyond 1 % while the 33 s run passes.

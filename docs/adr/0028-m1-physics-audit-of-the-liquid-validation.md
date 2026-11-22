# ADR 0028: M1 physics audit of the liquid's validation: Table 2's clock, Table 6, and the tests' scope

- Status: accepted
- Date: 2026-11-22

## Context

For the M1 gate (#58), a physics auditor with a fresh context re-derived V-LIQ-01 to
V-LIQ-05 from their references (SPEC 9.4). It read Martin & Moyce (1952) page by page and
Landet, Mardal & Mortensen (2018). All five tests are sound. The audit raised points
about what they say and what they cover:

- **Table 2's clock.** ADR 0019 says the paper's clock starts from the heater current,
  before the water moved, and that this is why the simulated clock is delayed by 0.25.
  The paper says otherwise. Table 2's records were rescaled so that T = 1.19 at Z = 1.44,
  a time it calls only "of the order of magnitude" of the time since the current
  (p. 314).
- **Table 6, on the heater clock.** Table 6 gives the height of the column's top at the
  wall behind it, H(τ), with τ = 0 when the current was applied. ENTROPY with no delay
  matches it within −4.1 % to +0.2 % at 16 cells and −3.9 % to +0.9 % at 32. The
  best-fit delay is only 0.05. With the 0.25 delay applied, H is off by +3.4 % to
  +10.8 %. Landet et al.'s Fig. 6b also matches Table 6 with no shift.
- **The delay itself still holds.** Solvers that release the column at once reach
  Z = 1.44 about 0.25 earlier than Table 2 says: Landet et al. 0.21–0.27 (Fig. 6a);
  ENTROPY 0.23 at 16 cells, 0.245 at 24 and 0.255 at 32. Any delay from 0.20 to 0.30
  passes. The cause is Table 2's normalisation, or a front that is slower in the
  experiment (the diaphragm, the floor's friction, surface tension at 5.7 cm); which
  one is not known.
- **Gravity.** ADR 0019 says V-LIQ-03 fails gravity × 0.9. That holds at 16 cells only
  (−7.95 %, by 0.95 point); at 32 cells × 0.9 passes (−6.6 %). Table 6 is more
  sensitive: × 0.9 reads +6.9 % and × 1.1 −8.7 %, at both resolutions.
- **V-LIQ-01 could not tell θ per face from θ = 1/2.** Water 1 m deep puts the surface
  on a face. It also tested only the projection with the surface given. Through the full
  step, with the game's seeding, the top layer reads −2.35 % to +0.3 %, because the
  particles' surface sits −1.4 to +0.8 mm off 1 m.
- **V-LIQ-05 never placed the surface between centres.** Both manufactured cases put the
  boundary halfway (θ = 1/2), so the "second-order free surface" of SPEC 5.2 was untested.
  The auditor's sphere, cutting the grid anywhere, converges at order 1.95; with one
  θ = 1/2 for all air faces, at order 0.55.
- **US-03 never compared 0.1× with 1×.** Each run passed V-LIQ-03 on its own, so two runs
  up to about 14 % apart could both pass. Measured, they differ by 2.6 % at most.
- **An unconverged pressure solve went unflagged.** The liquid ignored the solve's
  `converged`.
- **V-LIQ-02 has no 10-minute run.** ADR 0019 says the M1 soak repeats V-LIQ-02 for 10
  minutes; no such soak exists yet. A gentle 10-minute slosh drifted at most 0.71 %.

## Decision

- **Table 2's delay stays at ΔT = 0.25, with its true cause.** Table 2's clock is a
  normalisation, and solvers that release the column at once reach Z = 1.44 about 0.25
  earlier (Landet et al.; ENTROPY converged at 0.255). It is never fitted per run. The
  SPEC row, the test, the data file and this ADR say so; ADR 0019's "the clock starts
  from the heater current" is superseded here.
- **V-LIQ-03 adds Table 6.** The column's top at the wall, H = h/2a, against Table 6's
  means for n² = 2, a = 2¼ in, on the heater clock with no delay: every mean with
  0.56 ≤ τ ≤ 2.21 within 6 %.
  - It runs in the same dam break as the front.
  - The data are in `content/validation/martin_moyce_1952.toml`, read by the auditor
    and checked against the scan by the maintainer.
  - The window starts after τ = 0, where H = 1 by construction, and ends at τ = 2.21,
    the last mean within the dam break the front's test already runs (T = 3.2).
- **V-LIQ-01 covers a surface between centres and the full step.**
  - The projection test runs at 1 m (on a face) and 0.71 m (between centres). θ = 1/2
    fails the second by 42 %.
  - A full-step test holds still water 1 m deep for 1 s, seeded as the game seeds it:
    within 2 % below the top layer; in the top layer, within ρg × 1.5 mm (4.8 % of its
    306 Pa), for the particles' surface noise.
- **V-LIQ-05 adds the sphere.** −∇²p = f inside a sphere that cuts the grid anywhere,
  p = 0 on it, solved by the projection with its own θ per face: the error falls 3.4 to
  4.6 times per refinement (16, 32, 64 cells).
- **US-03 compares the two speeds.** Every 10th frame at 0.1× is a frame's instant at
  1×; the fronts agree there within 4 %.
- **Unconverged solves are counted.** The liquid's `diagnostics()["unconverged_solves"]`
  counts them, and the HUD shows the count when it is not zero.
- **V-LIQ-04's row is scoped.** It checks random states projected with no target
  divergence; the volume correction's target has its own test (ADR 0018).
- **V-LIQ-02's 10-minute run** is the M1 gate's soak (#58).

## Alternatives considered

- **Drop the delay and use Table 6 only.** Table 2's front is the observable most
  solvers are compared on, and the delay is independently supported.
- **Fit the delay from Table 6.** It reads 0.05, which cannot explain Table 2's 0.25;
  the two clocks differ, and fitting either per run would hide gravity errors (ADR 0019).
- **Widen V-LIQ-03 to T ≤ 3.33.** It passes (−2.4 % and +0.9 % at the two extra points)
  but adds little; Table 6 adds a second observable instead.
- **A clock-free check of the front's speed** (dZ/dT for Z = 5 to 9 in a tank 10a long).
  ENTROPY reads 1.756 against 1.754 from Table 2. It needs a tank twice as long; left
  for when the liquid's tests get slower budgets.

## How we would know it is wrong

- A solver change moves Table 6's fit away from no delay, or Table 2's converged delay
  away from 0.20–0.30.
- A later reading of Martin & Moyce, or a new experiment, ties Table 2's clock to the
  release.
- The HUD shows unconverged solves in normal play.

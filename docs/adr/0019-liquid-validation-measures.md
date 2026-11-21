# ADR 0019: How V-LIQ-02 and V-LIQ-03 measure the water

- Status: accepted
- Date: 2026-11-21

## Context

V-LIQ-02 (volume conservation) and V-LIQ-03 (dam break against Martin & Moyce 1952)
compare the liquid's full step (issue #50) with a law and an experiment. SPEC 9.4
gives the scenarios and tolerances, not how to measure. Measured on the full step:

- **Volume from the level set.** Zhu & Bridson's φ (ADR 0018) is a distance only near
  the surface. Inside the water it stays at −r, 0.375 of a cell. A linear ramp over φ
  counts those cells as 7/8 full; on the sloshing tank it missed 16 % of the volume.
  Counting the points where φ, interpolated
  between cell centres, is negative (4³ points per cell) gives the particles' volume
  to 5 digits in a still tank.
- **Sloshing.** On the 16 × 4 × 16 test tank, over 2,000 frames, that volume wobbles
  from step to step (standard deviation 0.4 %, peaks of 1.3 %). It has no trend: a
  linear fit moves by −0.15 % over the run. The wobble comes from a few particles per
  cell lumping and spreading; the particles carry a fixed volume.
- **Dam break.** The simulated column is released at once, the experiment's was not.
  With the clock starting at release, the simulated front leads the measured one by
  13–18 % (32 cells across the column). The lead is the same at 8, 16 and 32 cells
  and at 1/60 s and 1/240 s frames, so it is not a resolution error. It opens before
  T ≈ 1. Later, both fronts move at the same speed (dZ/dT ≈ 1.43 for T = 2.5–3).
  Lobovsky et al. (2014, Sec. 4.3.1) compare dam-break experiments, Martin & Moyce's
  among them. They find the differences largest for t* < 1 and write that this "may
  be probably attributed to various techniques of dam gate removal".

## Decision

- **V-LIQ-02 volume:** the share of 4³ points per cell where the interpolated level
  set is negative. The first frame's volume must match the particles' volume within
  1 %. The drift is the largest change of the volume averaged over each second (60
  frames), from the first second's mean. It must stay within 1 %.
- **V-LIQ-03 front:** where the level set crosses zero along the floor's row of cells,
  averaged across the tank's width. It is the water's edge the player sees; a lone
  particle ahead of it is spray (SPEC 5.2).
- **V-LIQ-03 clock:** shifted so that the simulated front reaches the first measured
  point (Z = 1.217 at T = 0.832) at that time. The shift must be a delay of at most
  0.5 in T (27 ms for the 2.25 in column). Every later measured point with T ≤ 3 must
  then be within 10 %.
- **Reference data:** `content/validation/martin_moyce_1952.toml`, Figure 3 of the paper
  as digitised by PySPH. The paper itself was not readable here, so the physics
  auditor checks the values at the M1 gate (#58).

Measured with this decision:

| Cells across a | Frame | Shift in T (ms for a = 2.25 in) | Errors at T = 1.22, 2.00, 2.55 |
|---|---|---|---|
| 8 (CPU test, a = 25 cm) | 1/60 s | 0.10 (5) | +7.8 %, +5.5 %, +3.9 % |
| 16 (game cells, a = 1 m) | 1/60 s | 0.18 (10) | +6.5 %, +4.1 %, +3.2 % |
| 32 (a = 50 cm) | 1/60 s | 0.25 (14) | +2.4 %, +1.3 %, +1.0 % |
| 32 (a = 50 cm) | 1/240 s | 0.24 (13) | +3.2 %, +2.5 %, +2.4 % |

## Alternatives considered

- **No shift:** a correct instantaneous release misses by 13–18 % at every resolution.
  The test would fail for a cause outside the solver.
- **Front speed only** (the slope for T ≥ 1, as in Lobovsky et al.'s Table 1): it
  passes too, but it ignores where the front is. The shift keeps position in the test
  and bounds the delay it allows.
- **Model the gate:** Martin & Moyce's release mechanism and its timing were not
  readable here.
- **A 20 % tolerance:** it would hide real errors of 10–20 % elsewhere in the run.
- **Volume per frame within 1 %:** it fails on the measurement's wobble, not on lost
  water. A finer grid lowers the wobble but costs CPU time in every test run.

## How we would know it is wrong

- The physics auditor finds that the digitised data, or the release delay it implies
  (about 13 ms), do not fit the paper.
- The shift drifts with resolution instead of settling, or nears its 0.5 bound.
- The volume's one-second mean changes steadily over a longer run (water lost or
  gained), even though each run stays within 1 %.

# ADR 0013: One shared advance per frame, solver substeps derived from it

- Status: accepted
- Date: 2026-11-20
- Refines ADR 0008 (frame target and time stepping), which stays in force.

## Context

The M0 physics audit re-derived ADR 0008's model against the solvers to come:

- All solvers must advance the world by the same simulated time each frame, or the
  couplings of SPEC 6 exchange momentum and heat across different times.
- The CFL substep cap (4 per frame, SPEC 5.9) and the 1/30 s clamp together limit real
  time to flows of about 4 cells per advance: 7.5 m/s for water in 6.25 cm cells at
  30 fps. A hose at 20 L/s through a 45 mm nozzle (12.6 m/s), water falling 2.5 m
  (7 m/s) or a 0.5–1 MW furniture fire plume (about 7 m/s, McCaffrey 1979) reach it.
- XPBD with a fixed count of 16 substeps per frame would make its substep vary 20-fold
  with ADR 0008's advance, and with it the effective stiffness and the joint forces that
  decide fracture.
- Advection and particle–grid transfers lose a little at each substep, so more substeps
  per simulated second (slow motion) means slightly more dissipation, not "no loss".

## Decision

- **Order of a frame.** The frame first fixes one advance for the whole world: frame time
  × time scale, clamped to 1/30 s (ADR 0008), then shrunk so that no solver needs more
  than its substep cap, using maximum speeds computed on the GPU. Only then does any
  solver step. The advance actually taken is reported, so a slowdown from the cap shows
  in the HUD like one from the clamp.
- **Liquid and gas substeps:** n = ⌈advance · u_max / Δx⌉ (gravity included in u_max, as
  in Bridson), at most 4, each of advance / n. The count stays on the GPU (conditional
  graph nodes) so the CPU never waits for it.
- **Rigid-body substeps:** at most 1/480 s each, count = ⌈advance / (1/480 s)⌉, so
  XPBD stiffness and joint forces do not depend on frame rate or time scale.
- **Slow motion** is honest: cheaper per frame, slightly more dissipative per simulated
  second. US-03 compares integral quantities (dam-break front, total volume) at 0.1× and
  1× within a tolerance measured in M1.
- **Sources** (hose, flood, gas leak) emit by accumulated mass or volume carried across
  substeps and frames, so a small advance in slow motion loses nothing to rounding.

## Alternatives considered

- **Each solver with its own advance:** breaks the conservation of every coupling.
- **Exceeding the substep cap instead of slowing down:** frames grow without bound in
  violent scenes; SPEC 5.9 already prefers slowing the world down.
- **A fixed substep for everything (for example 1/240 s):** constant cost per simulated
  second makes slow motion no cheaper, against SPEC 4.6.

## Consequences

- `step_simulation` returns the simulated time it actually advanced (M1, issue #34).
- SPEC 3 rows are per frame: cost per substep times the substeps a scene needs.

## How we would know it is wrong

M1 or M3 scenes routinely slow down at the cap in ordinary use (a hose, a room fire), or
the shared advance makes one solver's cost dominate the frame in scenes where others are
idle.

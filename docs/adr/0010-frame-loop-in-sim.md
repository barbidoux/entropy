# ADR 0010: The frame loop skeleton lives in `sim`, behind a stages protocol

- Status: accepted
- Date: 2026-11-20
- Refines ADR 0005, which gave the main loop to `app` and rejected putting it in `sim`.

## Context

ADR 0005 made `app` the composition root, holding "wiring only: no physics, no rendering
passes, no game rules", and gave it the window and the main loop. Building the loop in M0
showed that most of it is not wiring: the order of a frame (SPEC 5.9), the advance of
simulated time with its clamp and time scale (ADR 0008), frame pacing against a deadline,
and the per-stage statistics for the HUD. All of it needs unit tests with a fake clock,
which ADR 0005 says `app` should not need. SPEC 8.1 already lists "frame order, substeps,
time scale" under `sim`.

The objection in ADR 0005 was that a loop in `sim` would have to import `render` or `ui`,
inverting the arrows of SPEC 8.

## Decision

- `entropy.sim.loop.FrameLoop` owns the frame skeleton: input, logic, one simulation
  advance, rendering, presenting, pacing, statistics.
- It knows its stages only through the `FrameStages` protocol (`poll_input`, `update`,
  `step_simulation`, `render`, `present`). `sim` imports nothing from `render`, `ui`,
  `game` or `app`; the architecture test still enforces that.
- `app` keeps the window and the wiring: it builds the stages from the game, the
  renderer and the HUD, and runs the loop.
- SPEC 8.1 says so in the `sim` and `app` rows.

## Alternatives considered

- **The whole loop in `app`:** puts tested logic (time stepping, pacing) in the
  composition root, against ADR 0005's own rule.
- **Time stepping in `sim`, pacing loop in `app`:** splits one frame's timing across two
  places that must agree on the clock; more wiring for no gain.

## Consequences

- The loop is tested on CPU with fake stages, clock and sleep.
- Stage times are CPU wall times; GPU time per solver comes with the solvers (issue #29).

## How we would know it is wrong

The protocol keeps growing with render- or interface-specific calls, or `app` starts
re-implementing loop logic around `FrameLoop`.

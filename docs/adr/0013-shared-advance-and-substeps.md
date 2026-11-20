# ADR 0013: One shared advance per frame, solver substeps derived from it

- Status: accepted
- Date: 2026-11-20
- Refines ADR 0008 (frame target and time stepping), which stays in force.

## Context

The M0 physics audit re-derived ADR 0008's model against the solvers to come:

- All solvers must advance the world by the same simulated time each frame, or the
  couplings of SPEC 6 exchange momentum and heat across different times.
- The CFL substep cap (4 per frame, SPEC 5.9) and the 1/30 s clamp together limit real
  time to flows of about 4 cells per advance. For water in 6.25 cm cells at 30 fps that
  is 7.5 m/s including the gravity term √(g·Δx) ≈ 0.8 m/s, so flows of about 6.7 m/s.
  - A hose at 20 L/s through a 45 mm nozzle (12.6 m/s) needs 8 substeps at 30 fps.
  - Water falling 2.5 m (7 m/s) needs 5.
  - A 0.5–1 MW furniture fire plume (6.7–7.7 m/s, McCaffrey 1979) needs 3–4 substeps
    in 7.5 cm cells.
  - The hose's top nozzle speed, 30 m/s (SPEC 4.2), needs 17.
- XPBD with a fixed count of 16 substeps per frame would make its substep vary 20-fold
  with ADR 0008's advance, and with it the effective stiffness and the joint forces that
  decide fracture.
- Advection and particle–grid transfers lose a little at each substep, so more substeps
  per simulated second (slow motion) means slightly more dissipation, not "no loss".

## Decision

- **Who fixes the advance.**
  - The CPU fixes one advance for the whole world before it enqueues the frame: frame
    time × time scale, clamped to 1/30 s (ADR 0008). It then shrinks the advance so that
    no fluid solver needs more than its substep cap.
  - It judges that from the previous frame's maximum speeds. They are copied to pinned
    memory and read one frame later, so no synchronisation point is added (SPEC 3).
  - The advance is therefore known on the CPU. It is reported (a slowdown from the cap
    shows in the HUD like one from the clamp) and returned by `step_simulation`
    (issue #34). Sources and the rigid-body substep count use it.
- **Liquid and gas substeps.**
  - The GPU counts them from each solver's current maximum speed, at most 4, each of
    advance / n. The count stays on the GPU (conditional graph nodes), so the CPU never
    waits for it.
  - Liquid: n = ⌈advance · (u_max + √(g·Δx)) / Δx⌉. The gravity term bounds the speed
    gained during a substep; it is Bridson's √(5·Δx·g) at a CFL number of 1 instead of 5.
  - Gas: n = ⌈advance · u_max / Δx⌉. M3 decides whether semi-Lagrangian advection may
    take more than one cell.
- **When speed outruns the estimate.**
  - If the current speeds need more than the cap, the substeps stay at the cap and that
    frame moves fluid more than one cell per substep. This happens when speed grew more
    within one frame than the previous frame's reading allowed.
  - The GPU counts these frames; the HUD and the log show them, and the next advance
    shrinks. This is a named simplification (SPEC 5.9).
  - M1 measures how often it happens in US-01 to US-03 and sets a safety margin on the
    speed estimate if needed.
- **Rigid-body substeps.**
  - Count = ⌈advance / (1/480 s)⌉, each of advance / count. A substep is never longer
    than 1/480 s, and never shorter than 1/960 s when the advance exceeds 1/480 s.
  - So at normal speed XPBD stiffness and joint forces vary within a factor of 2 instead
    of 20 with the frame rate. In slow motion, substeps shrink with the advance.
- **Slow motion is honest.** It is cheaper per frame and slightly more dissipative per
  simulated second. In US-03 the dam break run at 0.1× and at 1× must both meet
  V-LIQ-03 (front within 10 % of Martin & Moyce for T ≤ 3) and V-LIQ-02 (volume drift
  ≤ 1 %). These tolerances come from the references, not from the runs.
- **Sources** (hose, flood, gas leak) emit by accumulated mass or volume carried across
  substeps and frames, so a small advance in slow motion loses nothing to rounding.

## Expected outcome

At 30 fps, with the cap of 4:
- US-01's hose at 20 L/s through a 45 mm nozzle runs the world at about 0.56×. Real time comes
  back at 54 fps or more.
- The top nozzle speed of 30 m/s runs it at about 0.24×.
- A 0.5–1 MW room fire stays within the cap in 7.5 cm cells. In 5 cm cells it runs at
  about 0.8–0.9×.
- Water falling 2.5 m runs it at about 0.96×.

US-01 still holds: it counts 200 L in 10 s of simulated time. M1 decides between:
- a higher liquid cap where the tank's pressure solve is cheap enough;
- a wider nozzle by default (65 mm gives 6 m/s at 20 L/s, within the cap);
- accepting the slowdown for strong jets.

## Alternatives considered

- **Each solver with its own advance:** breaks the conservation of every coupling.
- **Exceeding the substep cap instead of slowing down:** frames grow without bound in
  violent scenes; SPEC 5.9 already prefers slowing the world down.
- **A fixed substep for everything (for example 1/240 s):**
  - Liquid and gas would take 8 steps per 1/30 s frame where the CFL condition asks for 1
    or 2.
  - An advance that is not a multiple of the step would need the accumulator that
    ADR 0008 rejected, or a shortened last step, which is a variable substep after all.
- **Reading this frame's maximum speed before fixing the advance:** exact, but it adds a
  synchronisation point every frame, which stalls the CPU until the GPU drains its queue.

## Consequences

- `step_simulation` returns the simulated time it actually advanced (M1, issue #34).
- SPEC 3 rows are per frame: cost per substep times the substeps a scene needs.
- SPEC 6 couplings are exchanged once per advance. Water and rigid bodies are the
  exception: they are solved together in every liquid substep's pressure solve. M2
  decides how rigid substeps interleave with liquid substeps.

## How we would know it is wrong

At its milestone gate, at 30 fps, US-01's hose (with the nozzle and cap M1 picks) or
US-06's room fire keeps less than 0.9 of the requested time for more than 10 % of frames
(HUD "time kept"). Or the shared advance makes one solver's cost
dominate the frame in scenes where the others are idle.

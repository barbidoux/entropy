# ADR 0008: Frame target and time stepping, one advance per frame, 60 fps cap

- Status: accepted
- Date: 2026-11-20

## Context

SPEC 13 left the default frame target open until the M0 benchmarks, with a recommended
default of "60 fps rendering, simulation allowed to fall back to 30 fps". SPEC 3 budgets
a 30 fps frame at 22 ms of simulation and 9 ms of rendering. SPEC 5.9 says the frame is
1/60 s, falling back to 1/30 s under load, that each frame advances the world by the
frame time times the time scale, and that the liquid and gas solvers split that advance
into CFL substeps, at most 4. SPEC 4.6 wants slow motion to keep the same physical
substep and cost less.

The M0 benchmarks (docs/benchmarks/2026-11-20-m0-laptop.md) measured a memory-bound
GPU: a stencil sweep, kernel launch overhead and bandwidth. From them, the estimated cost
of a 128³ gas step alone is 6.8–13.5 ms per substep. The laptop has one GPU: simulation
and rendering kernels run on it one after the other.

## Decision

- **One simulation advance per frame.** Each frame advances the world once by the
  measured frame time multiplied by the time scale, clamped to 1/30 s. Solvers split that
  advance into CFL substeps (at most 4, SPEC 5.9). There is no fixed-step accumulator.
- **60 fps cap.** The frame loop waits so that a frame lasts at least 1/60 s, whatever
  the display's refresh rate; vertical sync stays on to avoid tearing. A frame is
  therefore normally 1/60 s and stretches to 1/30 s under load.
- **Below 30 fps** the clamp makes the world advance less than real time: the effective
  time scale drops, and the HUD says so.
- **Slow motion** is the time scale: at 0.1×, each frame advances the world by a tenth of
  its duration, so the world still moves every frame, needs fewer CFL substeps, and costs
  less (SPEC 4.6). No interpolation between states is needed.
- **Simulation and rendering run together**, once per frame, in the order of SPEC 5.9.
  There is no separate render rate in v1.0.

## Alternatives considered

- **Rendering at 60 fps while the simulation runs at 30 fps:** on one GPU, a 22 ms
  simulation step and two 9 ms renders do not fit in 33 ms, and it needs interpolation
  between two simulation states, which doubles the render-relevant memory.
- **Fixed 1/60 s steps with an accumulator (several steps per frame):** at 30 fps every
  frame would run two full steps, doubling the per-frame cost of every solver against the
  SPEC 3 budgets, and slow motion would update the world only every few frames.
- **No frame cap, vertical sync only:** on a display above 60 Hz the game would render
  faster than the budgets assume and heat the GPU for nothing.

## Consequences

- SPEC 3 budgets are per frame: a solver's cost is its cost per substep times the number
  of CFL substeps the frame needs, which depends on the frame time and the flow speed.
- The frame time drives the simulation step, so it varies slightly from frame to frame;
  CPU golden tests drive the simulation with scripted fixed time steps instead.
- SPEC 5.9 is reworded to state this model explicitly.

## How we would know it is wrong

Visible jitter in motion caused by frame-to-frame changes of the step, or a solver whose
stability or accuracy depends on a constant step in a way CFL substepping does not cover.

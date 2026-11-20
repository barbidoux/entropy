# ADR 0008: Default frame target, 60 fps with a 30 fps floor

- Status: accepted
- Date: 2026-11-20

## Context

SPEC 13 left the default frame target open until the M0 benchmarks, with a recommended
default of "60 fps rendering, simulation allowed to fall back to 30 fps". SPEC 3 budgets
a 30 fps frame at 22 ms of simulation and 9 ms of rendering. SPEC 5.9 already says the
frame is 1/60 s, falling back to 1/30 s under load.

M0 measured (docs/benchmarks/2026-11-20-m0-laptop.md) that the solvers are bound by
memory bandwidth (500 GB/s sustained) and that a 128³ gas step alone is about 6 ms. The
laptop has one GPU: simulation and rendering kernels run on it one after the other.

## Decision

- Simulation and rendering run together, once per frame, in the order of SPEC 5.9.
- The frame targets 60 fps when the scene allows it, and falls back to 30 fps under load.
  Each frame advances the world by the frame time times the time scale, in fixed
  1/60 s steps (at most 4 per frame); below 30 fps the time scale drops and the HUD
  says so (SPEC 5.9).
- Vertical sync is on by default (`content/settings.toml`), so a light scene does not
  render faster than the display for nothing.
- There is no separate render rate in v1.0.

## Alternatives considered

- **Rendering at 60 fps while the simulation runs at 30 fps:** on one GPU, a 22 ms
  simulation step and two 9 ms renders do not fit in 33 ms. It would also need
  interpolation between two simulation states, which doubles the render-relevant
  memory and shows motion that was never simulated.
- **A fixed 30 fps:** wastes the headroom of light scenes, such as the M0 empty scene or
  a small water tank, where 60 fps is easy.

## Consequences

- One frame loop, one time step, no interpolation code.
- Heavy scenes play at 30 fps, light ones at 60 fps, and the HUD shows which.

## How we would know it is wrong

Typical scenes measured in M1–M3 sit between 30 and 60 fps with visible stutter from the
fixed 1/60 s step (alternating 1 and 2 steps per frame), or play-testing shows that a
smoother camera at 60 fps matters more than simulation detail.

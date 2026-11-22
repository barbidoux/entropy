# ADR 0029: How the M1 gate measures "1 million particles at 30 fps or more"

- Status: accepted
- Date: 2026-11-22

## Context

SPEC 11's M1 gate asks for "1 million particles at 30 fps or more", and #57 for a
`-m perf` scenario in the Test Tank, a dam break followed by a hose, sustained at 30 fps
after the 5-minute warm-up. Neither says which frames count, at what size, or what else
must hold. The first version of the benchmark could pass with the world frozen by the
NaN guard or a hose that poured nothing, since frames then cost only their rendering
(#80's review).

## Decision

The gate is `tests/perf/test_m1_gate.py`:
- **The game itself**: `GameStages` with the Test Tank, the liquid, every render pass and
  the HUD, the game's frame loop (60 fps cap, vsync), at the game's own window size,
  1600 × 900 (`content/settings.toml`), in a hidden window.
- **Two parts of 600 frames**, the first 60 of each not counted (warm caches, captured
  graphs): a dam break of 1,048,576 particles released as a step (2.375 m against
  1.625 m), then the hose at its default 20 L/s and 10 m/s, held from above the rim
  through the game's input.
- **The rate**: each frame's real time. The gate wants the 10th percentile of the frame
  rate at 30 fps or more in each part: 30 fps or more in 90 % of the frames.
- **What must also hold**: the NaN guard never froze the world; the hose's flow over
  the part's simulated time is accounted for within 2 % (in the liquid, in flight,
  lost, over the brim), and at least 90 % of it reached the tank.
- **What is recorded**, not asserted: the world's speed (simulated over real time), the
  substeps' spread, the overruns and unconverged solves added, and every GPU stage's
  median and 90th percentile.

Named limits:
- **1600 × 900, not 1920 × 1080.** SPEC 3's render target is 1920 × 1080; the game's
  window is 1600 × 900. If rendering scaled with pixels, the liquid pass's worst
  measured view (1.69 ms) would take about 2.4 ms at 1920 × 1080, within its 2.5 ms;
  not measured.
- **A hidden window.** The compositor's own GPU work is not counted. In the visible
  frame-pacing benchmark, 17.9 ms of GPU work made frames of 20.2 ms.
- **The 10th percentile over 540 frames** lets 54 frames per part fall under 30 fps.
  Isolated long frames do not fail it; a sustained slowdown does.
- **30 fps is not real time.** Under the hose the liquid runs at its substep cap and the
  world advances slower than real time (about 0.95×): the frame rate holds, the world's
  speed is recorded beside it.
- **The liquid alone.** M1 has no other solver; later gates share the GPU.
- **The GPU's thermal state.** The hose part is GPU-bound near 30 ms a frame, so the
  result moves with the clocks. Official runs, each after the 300 s warm-up: p10
  35.0 fps with the SM at 2235–2310 MHz; 29.8 fps after about 80 minutes of
  continuous load, the SM at a median 2010 MHz under thermal slowdown in 79 % of the
  samples (`docs/benchmarks/2026-11-22-m1-laptop.md`).
- **Stage times overlap.** The simulation (CUDA) and the passes (OpenGL) share the GPU
  and its time slices: in the hose part a stage's timer reads up to about twice its
  cost measured alone. They show where time goes, not costs to add up.

## Alternatives considered

- **The median frame rate.** It hides a slow tenth of the frames.
- **The slowest frame.** The frame-pacing benchmark shows isolated long frames under
  load (ADR 0026); one would fail the gate for a stall the player hardly sees.
- **A visible window at 1920 × 1080.** It would count the compositor, but the
  benchmark would depend on the display attached; Matt's play session covers the
  visible game.

## How we would know it is wrong

- Matt sees the water tank stutter in play while the gate passes.
- The visible game, at the same scenario, runs well under the hidden window's rate.

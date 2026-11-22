# ADR 0026: Frame pacing, the slowest time scale and solver loops on the GPU

- Status: accepted
- Date: 2026-11-22

## Context

The M0 physics audit left three questions (#36).

- **Frame pacing.** With vertical sync on a 60 Hz display, a frame that misses a refresh
  could wait for the next one. A loop that then stayed a refresh behind would run at
  30 fps for good, and would need hysteresis.
- **The slowest time scale.** Positions are float32, whose spacing is 9.5e-7 m from 8 m
  of the origin and 1.5e-5 m from 128 m to 256 m. At a small enough time scale, slow water
  far from the origin would move less than that spacing per step and stop.
- **Solver loops.** ADR 0016 replays the pressure solve as fixed batches of 4 iterations
  and reads the residual back after each batch. Conditional graph nodes
  (`wp.capture_while`) would keep the loop on the GPU, but their cost had not been
  measured.

## Decision

### No hysteresis

Measured with `tests/perf/test_m1_frame_pacing.py`. The frame loop (60 fps cap,
ADR 0008) runs in a window with vsync. The laptop drove its primary display: an external
3840 × 2160 monitor at 60 Hz, through Windows' compositor. The work per frame is either a
CPU busy-wait or a GPU fragment shader sized to take it (each pass measured with a GL timer
query, within 1 %), with one longer frame.

| Work per frame | One longer frame | Median before | Median after | Longest run over 25 ms after |
| --- | --- | --- | --- | --- |
| 8 ms, CPU or GPU | 25 ms | 16.6 ms | 16.6 ms | 0 |
| 15 ms, CPU or GPU | 40 ms | 16.7 ms | 16.7 ms | 0 |
| 17.5 ms, CPU | 25 ms | 17.6 ms | 17.6 ms | 0 |
| 17.5 ms, GPU | 25 ms | 18.5 ms | 18.3 ms | 2 |

- The loop never stays a refresh behind. A frame over the period runs at its own length,
  not at 33 ms.
- With GPU work the long frame shows up to three frames later in the loop's times. At 8
  and 15 ms of GPU work it never shows: the frames queued ahead absorb it.
- With 17.5 ms of GPU work a frame takes 18.5 ms: the compositor's own GPU work adds
  about 1 ms. The two runs over 25 ms are single frames, apart.
- **Named limit: judder.** These are the loop's times, not when frames reach the screen.
  At 17.5 ms per frame the compositor repeats about one refresh in 20. Hysteresis would
  trade that judder for a steady 30 fps; the game does not make that trade.
- **Not measured:** fullscreen, the laptop's own panel, and other refresh rates.

### The slowest time scale is 0.1×

The constant is `MIN_TIME_SCALE`.
- At 0.1×, a 60 fps frame advances 1/600 s. Water at 1 cm/s then moves 1.67e-5 m: more
  than the float32 spacing anywhere within 256 m of the origin on every axis, which
  covers SPEC 5.8's 256 m shallow-water grid around the origin.
- At 0.04× that step rounds away; the threshold is 0.046×.
- The time control offers nothing slower, and `LoopSettings` refuses a scale between
  0 (pause) and 0.1.

Named limits:
- **It assumes the 60 fps cap** (`min_frame_s`). An uncapped loop, such as some
  benchmarks run, advances less per frame.
- **It holds per step of a solver, and so for the liquid as it substeps today.** The
  liquid takes one substep at 0.1× unless water moves faster than about 37 m/s.
  Solvers with small fixed substeps (MPM, M4) need their own answer, for example
  positions relative to their grid's origin.
- **Slow water far from the origin is quantised, not stopped.** Beyond 128 m at 0.1×,
  speeds snap to multiples of 0.92 cm/s per step, and water slower than 0.46 cm/s stays
  put. At 1×, the quantum is 0.092 cm/s.

### The pressure solve keeps its batches for now

Measured with `tests/perf/test_m1_conditional_graphs.py` (unofficial, GPU shared): 40
iterations of a two-sweep body on a 96³ grid, plus a count-down kernel.

| Variant | GPU time |
| --- | --- |
| Batches of 4 with 10 read-backs | 1.97 ms |
| One conditional node | 1.75 ms |
| One unrolled graph | 1.56 ms |

- The conditional node costs 4–5 µs per iteration.
- It saves the 9 read-backs after the first, each about 45–60 µs ((batched − unrolled)
  / 9), less the node's own cost. With the game's GL context on the same GPU a read-back
  may cost more: in one run with another process on the GPU, batches took 6.1 ms.
- The CPU enqueues it in 70–80 µs.
- In the M1 gate's dam break of 1 M particles (an unofficial run of #57's benchmark;
  its results land with #57), the liquid's solve takes 8 iterations (median) and 12 at
  most. That is 2–3 batches, so 1–2 read-backs to save, about 50–120 µs a solve, less
  the node's 35–60 µs for those iterations: 0–85 µs net.

Conditional nodes become the loop of choice when a solver runs many iterations (gas, M3),
or when read-backs show in a frame's GPU profile. Until then the CPU still counts the
liquid's substeps, from the previous frame's speed (ADR 0013, SPEC 5.9).

## Alternatives considered

- **Hysteresis anyway** (hold 30 fps for a second after a slow frame): it would replace
  occasional judder with a steady 30 fps, which halves the frame rate for any single
  slow frame.
- **Positions relative to the grid's origin:** no extra memory per particle, only one
  origin per grid. This is the answer for far maps and fixed-substep solvers; M1's maps
  do not need it.
- **float64 positions:** twice the memory and bandwidth of every particle's position.
- **Conditional nodes in the pressure solve now:** a small gain, for a change to the
  solve's control flow, which ADR 0021's sealed path already made more involved.

## How we would know it is wrong

- The frame-pacing benchmark finds a sustained run of 33 ms frames: in fullscreen, on
  the laptop's panel, or at another refresh rate.
- Maps reach beyond 256 m from the origin, or a solver steps finer than 1/600 s at
  0.1×.
- The pressure solve's read-backs show as gaps in the GPU profile at the M1
  performance gate (#57).

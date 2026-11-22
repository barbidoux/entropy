# ADR 0026: Frame pacing, the slowest time scale and solver loops on the GPU

- Status: accepted
- Date: 2026-11-22

## Context

The M0 physics audit left three questions (#36).

- **Frame pacing.** With vertical sync on a 60 Hz panel, a frame that misses a refresh
  could wait for the next one. The loop might then stay a refresh behind and run at
  30 fps for good. That would call for hysteresis.
- **The slowest time scale.** Positions are float32, whose spacing is 1e-6 m at 8 m from
  the origin and 1.5e-5 m at 128 m. At a small enough time scale, slow water far from
  the origin would move by less than that spacing per step and stop.
- **Solver loops.** ADR 0016 replays the pressure solve as fixed batches of 4 iterations
  and reads the residual back after each. Conditional graph nodes (`wp.capture_while`)
  would keep the loop on the GPU, but their cost had not been measured.

## Decision

- **No hysteresis.** Measured on the laptop's 60 Hz panel (windowed, Windows'
  compositor) with `tests/perf/test_m1_frame_pacing.py`:
  - after a single 25 or 40 ms hitch, no frame takes over 25 ms;
  - with 17.5 ms of work, frames run at 17.5 ms, not 33 ms;
  - the loop's 60 fps deadline (ADR 0008) and vsync do not lock each other a refresh
    behind.
- **The slowest time scale is 0.1×** (`MIN_TIME_SCALE`). At 0.1×, a 60 fps frame advances
  1/600 s, and water at 1 cm/s moves 1.7e-5 m: still above the float32 spacing at 128 m,
  the largest map extent planned. At 0.04× that step rounds away. The time control offers
  nothing slower, and `LoopSettings` refuses a scale between 0 (pause) and 0.1.
- **The pressure solve keeps its batches for now.** Measured with
  `tests/perf/test_m1_conditional_graphs.py` (unofficial, GPU shared): 40 iterations of a
  two-sweep body on 96³, plus a count-down kernel.

  | Variant | GPU time |
  | --- | --- |
  | Batches of 4 with read-backs | 1.97 ms |
  | Conditional node | 1.75 ms |
  | Unrolled graph | 1.56 ms |

  - The conditional node costs 4–5 µs per iteration. The 10 read-backs it removes cost
    about 0.25 ms.
  - The CPU enqueues the conditional graph in 68 µs.
  - The liquid's solve converges in 4–8 iterations, so it has only 1–2 read-backs to
    save: about 50 µs a solve.

  Conditional nodes become the loop of choice when a solver runs many iterations
  (gas, M3) or when read-backs show in a frame's profile.

## Alternatives considered

- **Hysteresis anyway** (hold 30 fps for a second after a slow frame): it would slow the
  game for nothing on this setup.
- **A time scale floor in the solvers** (relative positions, or float64 positions): this
  costs memory and bandwidth on every particle, for a slow motion slower than the game
  offers.
- **Conditional nodes in the pressure solve now:** a small gain for code that changes the
  solve's control flow, which ADR 0021's sealed path made more involved.

## How we would know it is wrong

- The frame pacing benchmark finds frames stuck at 33 ms in fullscreen or on another
  panel.
- Maps grow past 128 m from the origin, or slower motion is asked for.
- The pressure solve's read-backs show as gaps in the GPU profile at the M1 performance
  gate (#57).

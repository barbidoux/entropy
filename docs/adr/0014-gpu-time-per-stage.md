# ADR 0014: GPU time per stage from CUDA events and OpenGL timer queries

- Status: accepted
- Date: 2026-11-20

## Context

SPEC 4.7 asks the HUD for milliseconds per solver, and the SPEC 3 budgets (22 ms for
simulation, 9 ms for rendering) are GPU budgets. The HUD's stage times were CPU wall
times. Warp work is enqueued asynchronously, so its cost showed up where the CPU first
waited, not in the stage that enqueued it (issue #29).

Issue #29 asked for CUDA events around solver steps and render passes, read a frame
later so the CPU never waits. Render passes are OpenGL draws, and CUDA events cannot
time OpenGL work.

## Decision

- **One timer, two backends.** `entropy.core.profiler.StageTimer` brackets named stages,
  groups them by frame and returns a frame's times once its backend says they can be
  read.
- **CUDA events for Warp work** (`CudaEvents`):
  - A result is read only when both events have completed (`Event.is_complete`), with
    `synchronize=False`, so the CPU never waits.
  - Stages may nest.
  - `wp.ScopedStream` goes around a stage, never inside it. Stages bracket CUDA graph
    launches and never sit inside a capture.
- **OpenGL timer queries for render passes** (`entropy.render.gl_timer.GlQueries`):
  - moderngl 5.13 can neither ask whether a query's result is available nor give the
    query's GL name, and reading a result waits for the GPU. Results are therefore read
    a fixed 3 frames after their frame ended.
  - This is a named simplification. If the GPU is still behind, the read stalls; it
    never returns a wrong time.
  - OpenGL cannot nest these queries, so a nested OpenGL stage raises.
- **One rolling window per stage.** CUDA and OpenGL results arrive separately and at
  different lags.

## Alternatives considered

- **Raw `GL_QUERY_RESULT_AVAILABLE` through ctypes:** the cleanest non-blocking read.
  But moderngl does not expose the query's GL name, so it would mean managing queries
  outside moderngl with raw GL calls. In the review of #41, reads 3 frames late took at
  most 0.08 ms over 600 frames with and without vsync, so the simpler path was kept.
- **CUDA events around render passes, as the issue said:** they would time only the
  interop kernel, not the drawing.
- **Nsight or other external profilers:** useful for deep dives, but they are not in the
  game and Matt cannot read them.

## How we would know it is wrong

The HUD's CPU time for the rendering stage jumps when GPU times are on, because reading
the OpenGL queries stalls: a frame-time spike right after a heavy frame, with the GPU
still behind 3 frames later. Or a stage's GPU time is persistently 0 or far from what
Nsight Systems shows for the same frame.

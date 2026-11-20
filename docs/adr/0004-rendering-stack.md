# ADR 0004: Rendering stack, OpenGL 4.6 through moderngl and GLFW, zero-copy CUDA interop

- Status: accepted (with the interop pull request for #10, which brings the code and the measurement it cites)
- Date: 2026-11-20

## Context

SPEC 7 asks for a real-time renderer that reads simulation data straight from GPU
memory, on one Windows laptop with an RTX 4090 Laptop GPU, written by a maintainer who
works in Python (ADR 0002) and tests every pull request on CPU (SPEC 9). SPEC 7.1 named
OpenGL 4.6, moderngl and GLFW as the intended stack and asked M0 to check zero-copy
interop on Windows (risk R8).

## Decision

- **Window and input:** GLFW (pyGLFW bindings), one native window, mouse captured for
  look with raw motion when available. `entropy.render.context.GameWindow`.
- **Graphics API:** OpenGL 4.6 core profile on the laptop, wrapped by moderngl. Shaders
  are GLSL 4.50 so the same code runs headless on Mesa at 4.5 in CI (ADR 0009).
- **Simulation to rendering:** zero-copy CUDA–OpenGL interop. Warp kernels write into
  OpenGL buffers registered with CUDA (`wp.RegisteredGLBuffer`, with Warp's own silent
  copy fallback disabled), mapped as Warp arrays for the write and unmapped before
  drawing. `entropy.render.interop.SharedBuffer`.
- **Fallback:** if registration fails, or on CPU, kernels write a Warp array that is
  copied to the OpenGL buffer through host memory. The fallback is never silent: the
  reason is logged, kept on the buffer and shown by the HUD.
- **Interface:** Dear ImGui through imgui-bundle for the HUD and, later, the panels of
  SPEC 4.7.

## Evidence from M0

- OpenGL 4.6.0 core context on the laptop (NVIDIA 596.71), natively on Windows.
- Zero-copy interop works on the laptop: a Warp kernel writes a registered buffer and
  OpenGL reads back exactly the values written (`-m gpu` test).
- Cost for 1 million particles (12 MB of positions) per frame, first trial: about 0.2 ms
  with zero copy against about 8.9 ms through host memory. The host copy alone would eat
  the whole 9 ms rendering budget of SPEC 3; the official figure is recorded with the M0
  gate benchmarks.
- Mesa's software renderer offers OpenGL 4.5, not 4.6, under Xvfb (ADR 0009).

## Alternatives considered

- **Vulkan (through a Python binding or C++):** explicit synchronisation and better
  multi-queue control, but several times the code for the same passes, and no software
  implementation that is as easy to run in CI. SPEC 7.1 already excluded it for v1.0.
- **Rendering inside Warp (its own OpenGL viewer, or ray tracing in kernels):** fine for
  debugging, not for the passes of SPEC 7.2 (screen-space fluids, volumes, shadows).
- **Pyglet or SDL instead of GLFW:** comparable; GLFW is small, used by imgui-bundle's
  own backends, and its context creation is the best known on Windows.
- **Copying through host memory only:** simplest, but measured at about 9 ms per frame
  for 1 million particles.

## Consequences

- One code path for rendering on the laptop and in CI, at the cost of avoiding OpenGL
  4.6-only features.
- Every simulation field that is drawn lives in an OpenGL buffer or texture that Warp
  maps for writing; solvers stay unaware of OpenGL (SPEC 8.1, render reads only).
- GLFW state is process-wide: ENTROPY opens one window at a time.

## How we would know it is wrong

Interop fails on a driver update and the fallback becomes the normal path, or the
rendering passes of M1–M3 cannot meet the 9 ms budget for reasons OpenGL cannot fix
(for example synchronisation stalls between CUDA and OpenGL that Vulkan would avoid).

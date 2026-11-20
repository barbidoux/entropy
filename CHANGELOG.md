# Changelog

All notable changes to ENTROPY are recorded here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- CUDA–OpenGL interop (#10): `entropy.render.interop.SharedBuffer` registers an OpenGL
  buffer with CUDA and maps it as a Warp array, with no copy; on CPU or if registration
  fails it copies through host memory (pinned staging) and logs why (Warp's own silent
  copy fallback is disabled); a failed map is an error, never a write through a null
  pointer. Benchmark for 1 million particles. ADR 0004 accepted.
  fails it copies through host memory and says why (Warp's own silent copy fallback is
  disabled). Benchmark for 1 million particles. ADR 0004 accepted.
- Game loop (#11): `entropy.sim.loop.FrameLoop` runs input, logic (real time, capped at
  0.1 s after stalls), one simulation advance per frame (frame time × time scale,
  clamped to 1/30 s, slowdown reported) and rendering, with a 60 fps cap paced against a
  deadline (ADR 0008); `entropy.core.profiler` keeps rolling frame and stage statistics,
  the effective world speed and GPU memory for the HUD. ADR 0010: the loop skeleton
  lives in `sim` behind a stages protocol.
- Free camera (#12): God mode flight in the Z-up frame (`entropy.game.free_camera`),
  keys and mouse mapped to actions per SPEC 4.5 (`entropy.game.input`), view and
  projection matrices (`entropy.render.camera`).
- M0 laptop benchmarks (#5, #6, #7, #8): memory bandwidth, Jacobi sweep at 128³, 192³
  and the house grid with a streaming reference, kernel launch overhead with and without
  CUDA graphs. First official results in `docs/benchmarks/results/` and the report
  `docs/benchmarks/2026-11-20-m0-laptop.md`.
- Game window (#9): GLFW window with an OpenGL 4.6 core context wrapped by moderngl
  (`entropy.render.context`); settings as immutable dataclasses loaded from
  `content/settings.toml` (`entropy.core.config`). CI runs the `render` tests headless
  with Mesa software OpenGL 4.5 under Xvfb (#15); ADR 0009 records why 4.5 in CI and
  GLSL 4.50 shaders. ADR 0004 records the rendering stack.
- `entropy.linalg.jacobi` (#6): weighted Jacobi sweep for the 7-point Poisson stencil
  with zero values outside the grid, tested against NumPy and the exact discrete
  solution of a manufactured problem on CPU and CUDA. Each thread marches 8 cells along
  i to reuse neighbours from registers. Inputs are checked (shape, device, aliasing).
  Measured by the M0 stencil benchmark; the M1 multigrid smoother.
- Benchmark harness (#4): `bench/harness.py` and the `-m perf` session fixture. Checks
  mains power, energy saver and the Windows power mode around every benchmark, warms
  the GPU up for 5 minutes, times work with CUDA events, and writes JSON results with
  GPU telemetry. Only runs meeting the SPEC 3 conditions are official and go to
  `docs/benchmarks/results/`. ADR 0007 records the method.
- Test harness (#3): `device` fixture over CPU and, with `-m gpu`, CUDA; a test
  enforcing the module dependency rules of SPEC 8.1, with `app` as the composition
  root; coverage measured in CI with a 90 % floor, Warp kernels excluded; ruff enforces
  `import warp as wp`.
- ADR 0005: transitive layering and the `app` composition root. SPEC 8.1 gains the
  `app` and `audio` rows.
- ADR 0006: `docs/SPEC.md` is the reference copy of the specification, as Matt decided
  when M0 started; `CLAUDE.md` and the SPEC header say so.
- Specification v0.1 (`docs/SPEC.md`), mirrored from the living document.
- Walking skeleton: `entropy` package, Warp toolchain tests (CPU float atomics,
  bit-reproducible CPU sums, CUDA availability on the laptop).
- Tooling: uv, pytest with `gpu`, `perf`, `soak`, `slow` and `render` markers, ruff,
  pyright, pre-commit, GitHub Actions CPU CI.
- ADR 0001 (record decisions), 0002 (Python and Warp), 0003 (SI units, Z-up frame).
- Maintainer guide (`CLAUDE.md`).

### Changed

- SPEC 3 recalibrated from the M0 measurements: gas box 96³ to 128³ and at risk (192³
  out of reach), liquid pressure solve at risk on the full house grid, measured hardware
  limits (450 GB/s planning bandwidth, provisional sustained GPU state, kernel launches
  from CUDA graphs only, host and GPU per-kernel costs). SPEC 5.3, 5.9 and 11 aligned.
- ADR 0008: one simulation advance per frame of the frame time × time scale, clamped to
  1/30 s, frame rate capped at 60 fps (SPEC 13, SPEC 5.9); SPEC 4.6 and US-03 reworded:
  slow motion matches normal speed within a tolerance, not bit for bit.
- Benchmarks re-heat the GPU for 10 s before each run and read a non-zero buffer.

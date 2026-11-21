# Changelog

All notable changes to ENTROPY are recorded here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Materials and static solids (#47):
  - `content/materials/water.toml`, `glass.toml` and `concrete.toml`, each density with
    its source;
  - `entropy.world.materials`, which refuses unknown keys, unsourced constants and
    impossible values;
  - `entropy.world.sdf`: unions of boxes as exact signed distance fields with outward
    normals, an open box for the Test Tank, and sampling at grid cell centres.
- Staggered MAC grid (#45, `entropy.fields.mac`):
  - `GridSpec` and face-centred `MacVelocity`;
  - trilinear sampling of velocity and cell scalars: exact on linear fields, clamped
    outside the grid alike on CPU and CUDA, and NaN for a NaN point;
  - divergence and gradient, with the grid boundary a closed wall (homogeneous Neumann).

  ADR 0015 records these grid conventions.
- GPU time per stage in the HUD (#29): `entropy.core.profiler.StageTimer` brackets
  stages with CUDA events (Warp work) or OpenGL timer queries (render passes,
  `entropy.render.gl_timer`). It reads results only once the GPU has finished them
  (CUDA) or three frames later (OpenGL, a named simplification; nested OpenGL stages
  raise). The HUD shows GPU ms per stage under the CPU stage times, both labelled and
  aligned. M0 times the interop test pattern and the scene, points and HUD passes;
  solvers add their own stages from M1. ADR 0014.
- Solver isolation (#21): the architecture test sees each solver in `solvers/` as its
  own unit. A solver may not import another solver. The interface
  (`solvers/interface.py`) names no solver, and the package root imports only the
  interface. `coupling` may import only the interface or the package root that
  re-exports it. SPEC 8.1 and 10.2 name the interface module.
- M0 gate report (#18, `docs/benchmarks/2026-11-20-m0-gate.md`):
  - every gate criterion with its evidence;
  - a second official benchmark session at `68636eb`, with the first official interop
    measurement (0.23 ms zero-copy against 6.7 ms through host memory for 1 M
    particles);
  - a 10-minute soak at 60 fps with no memory growth.
  The budgets hold. SPEC 3's sustained-state line is widened (82–93 °C, 110–122 W, SM
  1.05–2.3 GHz, memory 7001–9001 MHz). The optimistic ends of the gas and liquid
  estimates use each grid's slower L2 rate (house 15.0 ps, 128³ 15.2 ps). Matt's play session remains for the
  gate, and the Dell thermal profile stays open.
- The game (#16): `uv run entropy` (or `python -m entropy`) opens the window and runs
  the M0 scene with free flight (click to look, Esc to release), the F3 HUD and a
  200,000-point interop test pattern written by Warp each frame (not physics; it follows
  a formula in simulated time). `uv run entropy-shortcut` puts an ENTROPY shortcut on the
  Windows desktop. A start-up error is logged to `out/entropy.log` and shown in a
  message box in French. Keys are set in `content/settings.toml` by the labels printed
  on the keyboard, shipped as Z Q S D (#32, ADR 0012). A soak test (`-m soak`) runs the
  game with random input and checks process, Warp pool and GPU memory growth.
- Performance HUD (#14): Dear ImGui overlay toggled with F3 (`entropy.ui.hud`) showing
  fps, mean and worst frame time, time per stage (CPU wall times), VRAM (device and
  ENTROPY's Warp pool), the interop path, the time scale or pause, and a highlighted line
  when frames lose simulated time to the 1/30 s clamp (SPEC 5.9). pyGLFW is pointed at
  imgui-bundle's GLFW library without importing imgui-bundle (ADR 0011).
- Empty lit scene (#13): sky with a sun disk, ground with 1 m and 10 m grid lines, five
  reference cubes of known size, Lambert sun light plus a sky term, ACES tone mapping and
  sRGB output (`entropy.render.scene`), drawn at the window's real size; simplifications
  named in the module docstring. `CameraView` hands the camera to the renderer. The start
  camera shows all five cubes, each with a lit and a shaded face.
- CUDA–OpenGL interop (#10): `entropy.render.interop.SharedBuffer` registers an OpenGL
  buffer with CUDA and maps it as a Warp array, with no copy; on CPU or if registration
  fails it copies through host memory (pinned staging) and logs why (Warp's own silent
  copy fallback is disabled); a failed map is an error, never a write through a null
  pointer. Benchmark for 1 million particles. ADR 0004 accepted.
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

- The Dell thermal profile is *Optimized*, as Matt confirmed after the M0 sessions
  (2026-11-21). SPEC 3's measured limits are taken to apply to it and are no longer
  provisional. CLAUDE.md and the benchmark guide say so.
- SPEC 3 recalibrated from the M0 measurements: gas box 96³ to 128³ and at risk (192³
  out of reach), liquid pressure solve at risk on the full house grid, measured hardware
  limits (450 GB/s planning bandwidth, provisional sustained GPU state, kernel launches
  from CUDA graphs only, host and GPU per-kernel costs). SPEC 5.3, 5.9 and 11 aligned.
- ADR 0008: one simulation advance per frame of the frame time × time scale, clamped to
  1/30 s, frame rate capped at 60 fps (SPEC 13, SPEC 5.9); SPEC 4.6 and US-03 reworded:
  slow motion matches normal speed within a tolerance, not bit for bit.
- Benchmarks re-heat the GPU for 10 s before each run and read a non-zero buffer.
- M0 physics audit applied:
  - ADR 0013: the CPU fixes one shared advance per frame from the previous frame's
    maximum speeds, before any solver steps. Liquid and gas CFL substeps are capped at 4
    with explicit formulas, including liquid gravity. XPBD substeps are at most 1/480 s.
    Sources emit by accumulated volume. The ADR states the expected slowdown for fast
    jets and a measurable falsifier.
  - SPEC 3: costs are per substep times substeps. Water at 2 m/s needs 2 substeps. A
    0.5–1 MW fire plume at 6.7–7.7 m/s needs 3–4 in 7.5 cm cells (McCaffrey 1979). The
    liquid solve is 4.6–12.7 ms per substep, including variational weights. Shallow water
    is costed per frame (Kurganov & Petrova CFL), and estimates are labelled.
  - SPEC 4.6 and US-03: slow motion is slightly more dissipative and must meet V-LIQ-02
    and V-LIQ-03 at both speeds.
  - SPEC 6: couplings are exchanged once per advance.
  - The benchmark report states the MGPCG assumptions (12–18 sweep equivalents per
    iteration).
  - M1 issues #34–#36.
- README: how to play M0.

# Changelog

All notable changes to ENTROPY are recorded here, newest first.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Rigid contacts (#87, ADR 0031): bodies against the static solids and against each
  other.
  - A hash grid finds the pairs. Each body's sample spheres (a box's corners, edge
    midpoints and face centres, a sphere's centre, points along a capsule at most a
    radius apart) are tested against the other body's exact signed distance and against
    every static plane their path meets, the static boxes' inner faces left out. The
    static boxes' edges are tested against the body. Two boxes share their axis of least
    overlap as the normal.
  - Each body's contacts with the static solids are solved by its own thread, and each
    pair of bodies by its own, the pairs in parallel, scaled to keep momentum. A sweep
    takes a box's face centres first and a capsule's middle first.
  - A velocity pass at each contact patch stops the corrections turning into speed for a
    lone pair.
  - V-RB-04: linear momentum within 1e-4 in collisions without gravity; angular
    momentum within 2e-3, provisional until the M2 physics audit.
  - A body alone rests on the floor sunk by under 0.3 mm. A stack of 10 boxes of 20 cm
    sinks 7.7 mm. A box with 45 % of it over a table's edge stays on.
  - 10 and 20 cm boxes stay in the tank up to 30 m/s; smaller ones up to 20 m/s. A box
    slides across the seam of two slabs and past a table's edge untouched.
  - A long capsule stays on a thin beam.
  - A body touching only the static solids never gains energy from them: a box spawned
    deep inside a table is not flung, and a plank thrown at the floor at 30 m/s does
    not rebound higher than it fell. A box falling beside the tank's outer glass is
    untouched.
  - The gyroscopic step is split for fast spins: a thin plank keeps |L| and its energy
    within 1e-3 up to 3,000 rad/s, where it froze the solver with a NaN from about
    1,500 rad/s.
  - A NaN dump replays, contacts included (`RigidSolver.from_dump`).
  - Pairs past the room are counted per step.
  - The limits are named in ADR 0031, with their numbers. Without friction, bodies
    resting off-centre slide off; several bodies at once still pop and bounce.
  - The static solids' distance maths moves to `entropy.fields.solids`, shared by
    `world` and the solvers (SPEC 8.1).
  - Unofficial cost: 20.7 ms per 1/30 s frame at 5,000 bodies, against SPEC 3's 3 ms.
    That is #90's.
- Rigid bodies drawn with the solids (#91): `entropy.render.bodies` sorts the bodies by
  shape on the GPU into a buffer shared with OpenGL (zero copy on CUDA), writes one
  indirect draw per mesh, and draws boxes, spheres and capsules lit like the static
  boxes, before the water and the glass. 5,000 bodies at 1920 x 1080: 0.38 ms mixed,
  0.02 ms all boxes (unofficial; `-m perf` benchmark). `WorldFrame` draws them, timed as
  "bodies (OpenGL)", when given a `body_capacity`; the game wires them with the spawner
  (#93).
- Rigid bodies, first step (#86, ADR 0030): `entropy.solvers.rigid` with boxes, spheres
  and capsules, their mass and inertia from shape and density, XPBD substeps of at most
  1/480 s under gravity with the gyroscopic torque by the implicit midpoint rule (|L| and
  energy kept within 1e-3 over 10 s at 20 and 30 rad/s), the NaN guard (a dump with the
  frame's starting state, time step and gravity, replayable) and checkpoints. V-RB-01: a
  1 s fall within 0.21 % of ½gt² (limit 0.5 %). Velocities are recovered from the
  motion's increments, which keeps them exact in float32. No contacts yet.
- The M1 gate (#57, #58, ADR 0029):
  - `-m perf` gate benchmark: the game with 1 M particles, a dam break then the hose;
    the frame rate held in 90 % of the frames, the world's speed and every stage's GPU
    time; it fails on a frozen world or a hose that does not pour. Official on the
    laptop: dam break p10 58.7 fps, hose p10 35.0 fps (unofficially 35.4–36.1 with the
    final test, whose official run is owed, #85); the hose's 4-substep frame is above
    the simulation budget, and #81 brings it down before M3;
  - `-m soak` M1 soak: 10 minutes of random tool use, time controls and checkpoints in
    the Test Tank, the NaN guard checked every frame, no leak or crash (memory sampled
    in quiet windows, clear of F9's 3 s hold), and V-LIQ-02's volume criteria on the
    settled water every 30 s, at 1x (ADR 0028);
  - official benchmark session and reports (`docs/benchmarks/2026-11-22-m1-laptop.md`,
    `2026-11-22-m1-gate.md`); SPEC 3's liquid row measured (closes #65) and the gas
    row's 128³ solve updated.
- The M1 physics audit of the liquid's validation (#58, ADR 0028):
  - V-LIQ-01 at a surface between cell centres (0.71 m), and through the full step:
    still water 1 m deep for 1 s, seeded as the game seeds it;
  - V-LIQ-03 adds the column's top at the wall against Martin & Moyce's Table 6 on
    their heater clock, with no delay (within 6 %; measured -4.1 to +0.2 %): it catches
    gravity off by 10 %, which the front alone does not at 32 cells;
  - V-LIQ-05 adds a sphere cutting the grid anywhere, its surface placed by θ per face
    (order 1.95; one θ = 1/2 would give 0.55);
  - US-03 compares the fronts at 0.1x and 1x at the same simulated instants (within
    4 %; measured 2.6 %);
  - the liquid counts pressure solves that did not converge
    (`diagnostics()["unconverged_solves"]`), and the HUD shows them when there are some;
    overruns and unconverged solves are logged at the 1st, 10th, 100th... (#34).
- The liquid sorts its particles by half-cell at the start of every frame (#65, ADR
  0025): a stable radix sort gathered into the second particle buffer, which then
  becomes the live one, so the NaN guard's dump keeps the frame's starting state at no
  extra copy. Measured at 1 M particles by the independent review (unofficial): particles
  to grid 3.55 to 0.98 ms per substep, grid to particles 0.90 to 0.16 ms, for about
  0.2 ms of sorting per frame net of the copy it replaces. Sorting by whole cell piled
  the atomics on the same nodes (2.38 ms). A `-m perf` benchmark times the transfers in
  random, bucket, lattice and half-cell order, and the sort. Particles to grid in
  shared memory (Gao et al. 2018) was tried and not adopted (2.06 ms, three times the
  code).
- Frame pacing, the slowest time scale and solver loops on the GPU (#36, ADR 0026):
  - 0.1x is the slowest time scale (`MIN_TIME_SCALE`): float32 positions still move
    water at 1 cm/s within 256 m of the origin at 60 fps; the loop refuses a slower
    one; far from the origin slow speeds are quantised (a named limit);
  - `-m perf` benchmark of frame pacing with vsync (CPU and GPU work, the display
    recorded): one long frame never leaves the loop a refresh behind, so no hysteresis
    is added; judder at just over a refresh is a named limit;
  - `-m perf` benchmark of a solver loop as a conditional graph node: about 4-5 µs of
    GPU per iteration, against batches with read-backs (unofficial); the pressure solve
    keeps its batches for now.
- Follow-ups of #74's review: the bucket says how much it dropped into a full tank, in
  either shape; a tool held into a pause still gets its release, even after a switch;
  only a tool that was pressed is held and released, and a switch starts the new tool
  only if one was in use and the button is still held (no phantom click when the key
  and the release come in one frame); a release left from a pause no longer stops the
  tool a switch starts, and a button let go and pressed again within one frame keeps
  the tool going; a tool's key (`order`) must be 1 to 9; tests for
  hosing from inside the tank and for the eraser through the toolbox on CPU; the flood
  gameplay test checks the water is not squeezed (#76 has the solver's lag).
- The bucket, the flood source and the eraser (#54):
  - the bucket drops exactly its volume (0.01–2 m³) where the aim meets the rim over
    the tank; right click turns it into a dam, Martin & Moyce's column 2a high across
    the tank against the far wall (US-02 passes in the Test Tank);
  - the flood source raises the water over the whole floor at its rate (0.01–5 m³/s,
    the wheel scaling it by 1.25 per notch), by accumulated volume; its water comes in
    at the level the water stands at (a named simplification: from the floor, 2 m³ in
    4 s stood at half its depth);
  - the eraser removes the water within its radius (0.1–5 m) of where the aim meets
    it; the liquid removes particles on the GPU (prefix sum), keeping their order, and
    the eraser reads back only its level-set samples and two counts.
- Time controls and checkpoints (#55, ADR 0024): P pauses, [ and ] cycle 1x, 0.5x,
  0.25x and 0.1x, the period steps one frame while paused (the HUD names these keys as
  the keyboard prints them: ^ $ : on AZERTY); F5 saves and F9 restores, again to go
  further back; a ring of 6 checkpoints every 10 s of simulated time plus one at the
  start, copied to pinned host memory without waiting, the buffers of the checkpoint
  that falls off reused. Checkpoints keep in step with the NaN guard: a freeze drops
  those saved since the last good step, F9 checks the last step before restoring, and
  F5 does nothing while frozen. Clicks while paused or frozen are dropped. US-03: a dam
  break at 0.1x and at 1x both meet V-LIQ-03 and keep their volume within 1 %.
- The HUD counts the water (#56): particles, litres (the particles' own volume),
  cells of water, substeps, pressure iterations and overruns; the liquid's GPU time per
  phase (transfers, pressure, motion) through the step's context.
- Tools and the hose (#53, #35, ADR 0020):
  - tool plug-ins with their parameters, limits and defaults in `content/tools/`;
    keys 1–9 choose, the mouse wheel sets the power, the left button (mouse
    captured) uses; the palette shows the settings in SI units in the HUD;
  - the hose: flow 1–50 L/s, nozzle 5–30 m/s (default 10 m/s), released by
    accumulated volume, so slow motion loses nothing to rounding;
  - outside the liquid's grid the jet flies ballistically and is drawn; it enters
    the tank over the rim, and water that hits the glass or the ground is lost;
  - the tools act first in each frame, with the shared advance, inside the NaN guard
    (`Simulation.step(player=...)`): a frozen world takes no water;
  - tools are found through a registry from their content files (`order` is the key);
  - the grid holds at most its volume of water: past it, tools lose the water and the
    HUD says "Cuve pleine";
  - the hose's first frame of water is one frame long and its path is judged from the
    camera, so it cannot reach through the glass at point-blank range;
  - US-01: 10 s of hose at 20 L/s puts 200 L ± 2 % in the Test Tank, at 1x and 0.1x.
- Benchmarks sample GPU telemetry during each measurement (#25, ADR 0022): one
  `nvidia-smi -lms 100` process in the background, summarised in each result (min,
  median, max and count of each number; counts of P-states and clock event reasons),
  and stopped even when a benchmark raises (not when Python is killed outright).
  Compared passes are timed interleaved, round by round (`time_interleaved`, GPU-bound
  passes only); the stencil benchmark uses it, with 50 rounds, so its results no
  longer compare with M0's (the benchmark README says why).
- Water on screen (#52, SPEC 7.2):
  - the frame is drawn in linear light into a half-float target and tone-mapped last
    (`entropy.render.frame`); the glass now blends in linear light;
  - the liquid pass (`entropy.render.liquid`): particles as spheres, their depth
    smoothed by a narrow-range filter (Truong & Yuksel 2018), normals, thickness;
    Fresnel reflection of the sky, refraction by Snell's law over the thickness, and
    Beer-Lambert absorption with pure water's coefficients (Pope & Fry 1997, in
    `water.toml` with its refractive index);
  - `entropy.render.world_frame` draws solids, water, glass and tone mapping in
    order, each pass timed in the HUD; the floor shows a metre grid;
  - measured on the laptop: 1.56 ms for the liquid pass at 1 million particles,
    1600 x 900 (budget 2.5 ms); a still pool matches its reference image
    (SSIM >= 0.98).
- The game starts in the Test Tank (#51):
  - `entropy.sim.setup.liquid_setup` builds the liquid from the world. Its grid spans
    the tank's cavity plus 0.5 m of air above, in 6.25 cm cells; other solids inside
    become its static solid;
  - `entropy.render.world_view` draws the concrete slab with its material's colour,
    and the glass as panels that reflect the sky by Fresnel's law (n from
    `glass.toml`; no refraction, a named simplification); the ground lies under the
    lowest solid;
  - a non-finite liquid stops the world, which stays on screen, and the HUD says
    where the dump is (SPEC 5.9);
  - materials take optional constants with sources: refractive index, albedo, and
    absorption per metre.
- The liquid's full step (#50, `entropy.solvers.liquid.solver.LiquidSolver`):
  - each substep: particles to grid, gravity, density and level set from the
    particles' mass (the surface where the density is ½, ADR 0018), a volume
    correction (local density, and volume control after Kim et al. 2007), the
    projection, two layers of velocity extrapolated into the air, grid to particles,
    midpoint motion kept inside the grid and out of static solids;
  - substeps from the previous frame's speed and from added particles' speed
    (ADR 0013), at most 4; frames that needed more are counted as overruns;
  - NaN guard: a non-finite state stops the liquid until `restore`, and its dump holds
    the state the failing frame started from and its time step, to replay it;
  - the projection takes a target divergence; `Solver` gains `max_advance_s()`
    (SPEC 8.2).

  A still tank stays still: after 1 s, 99 % of the particles moved less than 1 mm and
  the surface is within 1 mm of the water's height. A jittered surface settles; a
  falling blob accelerates at g within 2 %.
- V-LIQ-02 and V-LIQ-03 pass (#50, ADR 0019):
  - a sloshing tank keeps its volume within 1 % over 33 s (measured: 0.6 %);
  - a 1 m column at the game's 6.25 cm cells follows the means of Martin & Moyce's
    Table 2 within 7 % for 1.19 ≤ T ≤ 3 (measured: −3.2 to +1.4 %), with the
    simulated clock delayed by a fixed 0.25, since the paper's clock starts before
    the water moved;
  - the paper's Table 2 sits in `content/validation/martin_moyce_1952.toml`.
- Pressure projection of the liquid (#49, `entropy.solvers.liquid.projection`):
  - faces against static solids and walls carry no flow;
  - one θ per fluid-air face from the liquid's level set (second-order free surface,
    Enright et al. 2003), which MGPCG now accepts per face;
  - a warm start from the last pressure.

  V-LIQ-01 (hydrostatic pressure within 2 %, with the surface on or between faces, along
  every axis and on either side) and V-LIQ-04 (divergence below 1e-5, on every device)
  pass. ADR 0017: the tank's walls are the grid's walls, static solids are voxelized;
  named limits: water cannot leave the tank, the grid's top is closed, and water sealed
  from air waits for #69. `system_from_labels` refuses `air_theta` with `face_theta`.
- Solver interface and APIC transfers (#48):
  - `entropy.solvers.interface.Solver` (SPEC 8.2), re-exported by the package root;
  - `entropy.solvers.liquid`: particles with position, velocity and the APIC affine
    matrix, and quadratic B-spline transfers to and from the MAC grid.

  Nodes outside the grid take the nearest sample (ADR 0015): no mass is lost at the
  walls, and a uniform flow stays uniform up to them. For particles at least one cell
  inside the grid, the transfers keep affine fields exactly and conserve mass, momentum
  and APIC angular momentum. NaN positions reach the grid as NaN.
- MGPCG pressure solver (#46, `entropy.linalg.mgpcg`):
  - cell labels (fluid, air, solid), face weights and a ghost-fluid θ, for the
    second-order free surface;
  - conjugate gradient preconditioned by a symmetric multigrid V-cycle;
  - float64 scalars kept on the device, with the solve start and batches of 4
    iterations replayed as CUDA graphs.

  V-LIQ-05 passes: the error falls 4× per refinement, with Dirichlet and with Neumann
  walls. `refresh()` relabels in place so graphs stay valid. Benchmark `-m perf`
  (relabel and solve from zero, per substep): 4.0 ms on the Test Tank grid, 13.8 ms on
  the full house grid. SPEC 3 is updated: both are at risk. ADR 0016.
- One shared advance per frame (#34, ADR 0013):
  - `entropy.sim.simulation.Simulation` shrinks the frame's advance to the tightest
    solver limit (`max_advance_s()`, from the previous frame's speeds) before any solver
    steps, then steps every solver by it, in order;
  - `step_simulation` returns the time actually advanced, and the frame loop counts
    only that, so a slowdown forced by a substep cap shows in the HUD;
  - SPEC 8.2 adds `max_advance_s()` to the solver contract.
- Map files, format v1, and the Test Tank (#51):
  - `entropy.world.maps`: `map.toml` (schema version, name, environment) and
    `scene.json` (entities), with load, save and a migration registry that refuses
    newer schemas;
  - `entropy.world.prefabs`: `glass_tank` and `floor_slab`;
  - `entropy.world.world.build_world`;
  - `content/maps/test_tank/`: a 4 × 4 × 2 m glass tank with 2 cm walls on a concrete
    slab.
- Materials and static solids (#47):
  - `content/materials/water.toml`, `glass.toml` and `concrete.toml`, each density with
    its source;
  - `entropy.world.materials`, which refuses unknown keys, unsourced constants and
    impossible values;
  - `entropy.world.sdf`: static solids as signed distance fields with outward
    normals. Primitives are boxes, open boxes (the Test Tank: outer box minus a cavity
    open at the top, exact inside the glass) and half-spaces; there is also sampling
    at grid cell centres, with its thin-wall limit named.
  - SPEC 5.1: concrete is 2,450 kg/m³ (sourced).
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

- V-LIQ-03's delay of 0.25 against Table 2 has an accurate rationale in the spec, the
  test and the data file: Table 2's clock is a normalisation, not the heater current
  (which is Table 6's clock), and solvers that release the column at once lead it by
  about 0.25, for reasons not known (ADR 0028, superseding ADR 0019 on this point).
- The liquid pass draws its spheres at half resolution (#57, ADR 0027): depth,
  thickness and smoothing in half-size targets; the composite at full resolution, its
  depth and normals interpolated between the four nearest half-size pixels on one
  surface, and water behind each output pixel's solid left out. At 1 M particles and
  1600 x 900 (unofficial): 0.58, 1.13 and 1.62 ms from the start camera, above the rim
  and up close, against 1.46, 3.58 and 4.64 ms before (budget 2.5 ms). The liquid pass
  benchmark covers the three views; the still pool's reference image is re-recorded.
- The HUD drops a GPU stage that has not been timed for a whole window of frames, such
  as a solver switched off (#43).
- CI runs actions/checkout 7.0.1 and astral-sh/setup-uv 10.3.3, both on node24, pinned
  exactly (#42).
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

### Fixed

- A sphere smaller than a pixel adds its own volume to the water's thickness (its
  chords weighted by d², ADR 0027): far water no longer reads too thick (+48 % to
  +484 % before, once spheres were a third to a half of a pixel across).
- Water sealed from air (#69, ADR 0021): when no cell is air, the pressure solve
  (MGPCG) drops the right-hand side's part along the constant, removes the constant from
  its preconditioned residual every iteration, and sets the pressure's mean to 0. A grid
  filled to the lid now stays still, every solve within 8 iterations at the Test Tank's
  size; before, solves ran to the 200-iteration cap with pressures up to 10¹⁰ Pa. Grids
  with air pay nothing measurable: the first read-back says whether any cell is air.

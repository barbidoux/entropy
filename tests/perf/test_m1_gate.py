"""M1 performance gate (#57, SPEC 3 and 11): 1 million particles at 30 fps or more.

The game itself runs (`GameStages`): the Test Tank, the liquid, the frame with every pass,
the HUD, at the game's 1600 x 900, with the game's frame loop (60 fps cap, vsync), in a
hidden window. The scenario has two parts.

1. **Dam break.** 1,048,576 particles, the tank's 32 m³ to its rim, released as a step:
   one half 2.375 m deep, the other 1.625 m. Measured over 600 frames.
2. **Hose.** The hose then pours into the moving water at its default 20 L/s and 10 m/s
   from above the rim. Measured over 600 frames more.

Each frame's real time gives the frame rate. Its tenth percentile is the rate held in
90 % of the frames, and the gate wants it at 30 fps or more (ADR 0029). Each stage's GPU
time comes from the game's own timers (CUDA events and OpenGL queries): the simulation
stage's span, the liquid's transfers, pressure and motion, and every render pass. The
first 60 frames of each part are not counted (warm caches, captured graphs). Each part
also records the world's speed (simulated over real time), the substeps' spread, and the
overruns and unconverged solves it added.

The world must not have frozen (the NaN guard), and the hose must have released its
flow over the part's simulated time, nearly all of it into the tank: otherwise a broken
world or a dry hose would pass on render time alone.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import glfw
import numpy as np

from bench.harness import BenchSession
from entropy.app.main import GameStages
from entropy.core.config import default_settings_path, load_settings
from entropy.core.profiler import FrameStats
from entropy.game.free_camera import CameraPose
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop
from entropy.sim.setup import LIQUID_CELL_M
from entropy.solvers.liquid.solver import fill_box

DX_M = LIQUID_CELL_M
SIZE = (1600, 900)
FRAMES = 600
SKIP = 60
GATE_FPS = 30.0


class _AllStats(FrameStats):
    """The game's statistics, keeping every frame's GPU stage times as well."""

    def __init__(self) -> None:
        super().__init__()
        self.gpu_samples: list[dict[str, float]] = []

    def record_gpu(self, stages_s: Mapping[str, float]) -> None:
        super().record_gpu(stages_s)
        self.gpu_samples.append(dict(stages_s))


def _part(loop: FrameLoop, stats: _AllStats, stages: GameStages) -> dict[str, Any]:
    assert stages.liquid is not None
    first_gpu = len(stats.gpu_samples)
    frame_ms, substeps, iterations = [], [], []
    simulated_s = counted_simulated_s = counted_real_s = 0.0
    before = stages.liquid.diagnostics()
    for index in range(FRAMES):
        if index == SKIP:
            first_gpu = len(stats.gpu_samples)  # the timers record one sample per stage
            before = stages.liquid.diagnostics()
        report = loop.run_frame()
        simulated_s += report.advance_s
        if index >= SKIP:
            frame_ms.append(report.frame_s * 1e3)
            counted_simulated_s += report.advance_s
            counted_real_s += report.frame_s
            d = stages.liquid.diagnostics()
            substeps.append(d["substeps"])
            iterations.append(d["pressure_iterations"])
    after = stages.liquid.diagnostics()
    frame = np.asarray(frame_ms)
    fps = 1e3 / frame
    gpu: dict[str, list[float]] = {}
    for sample in stats.gpu_samples[first_gpu:]:
        for name, seconds in sample.items():
            gpu.setdefault(name, []).append(seconds * 1e3)
    counts = np.bincount(np.asarray(substeps, dtype=int))
    return {
        "simulated_s": simulated_s,
        "world_speed": counted_simulated_s / counted_real_s,
        "substeps_frames": {str(n): int(c) for n, c in enumerate(counts) if c},
        "overruns": int(after["overruns"] - before["overruns"]),
        "unconverged_solves": int(after["unconverged_solves"] - before["unconverged_solves"]),
        "frames": len(frame),
        "fps_median": float(np.median(fps)),
        "fps_p10": float(np.percentile(fps, 10)),
        "frames_below_gate": int((fps < GATE_FPS).sum()),
        "frame_ms_median": float(np.median(frame)),
        "frame_ms_p90": float(np.percentile(frame, 90)),
        "frame_ms_max": float(frame.max()),
        "substeps_median": float(np.median(substeps)),
        "pressure_iterations_median": float(np.median(iterations)),
        "pressure_iterations_max": float(np.max(iterations)),
        "gpu_ms_median": {name: float(np.median(ms)) for name, ms in sorted(gpu.items())},
        "gpu_ms_p90": {name: float(np.percentile(ms, 90)) for name, ms in sorted(gpu.items())},
    }


def test_one_million_particles_at_30_fps_or_more(bench_session: BenchSession) -> None:
    base = load_settings(default_settings_path())
    window_settings = dataclasses.replace(
        base.window, width_px=SIZE[0], height_px=SIZE[1], title="M1 gate"
    )
    settings = dataclasses.replace(base, window=window_settings)
    stats = _AllStats()
    with GameWindow(settings.window, visible=False) as window:
        stages = GameStages(window, settings, stats)
        try:
            assert stages.liquid is not None
            deep = fill_box((-2.0, -2.0, 0.0), (0.0, 2.0, 2.375), DX_M, 8, seed=1)
            shallow = fill_box((0.0, -2.0, 0.0), (2.0, 2.0, 1.625), DX_M, 8, seed=2)
            stages.liquid.add_particles(np.concatenate([deep, shallow]))
            assert stages.liquid.particle_count() == 1_048_576
            loop = FrameLoop(
                stages, settings=stages.loop_settings, stats=stats, time_control=stages.time_control
            )
            loop.reset_clock()
            bench_session.start("m1_gate_1m")
            dam = _part(loop, stats, stages)
            # The hose (tool 1, the default), held from above the rim, aimed into the tank,
            # through the game's own input: a click captures the mouse, then the left
            # button is pressed and never released.
            stages.mode.pose = CameraPose((0.0, -2.6, 3.2), math.radians(90.0), math.radians(-50.0))
            stages.mode.on_mouse_button(pressed=True)
            stages.mode.input.on_mouse_button(glfw.MOUSE_BUTTON_LEFT, glfw.PRESS)
            hose = _part(loop, stats, stages)
            frozen = stages.simulation.frozen_by
            hose_tool = stages.toolbox.current
            flow_m3_per_s = hose_tool.values["flow_l_per_s"] * 1e-3
            released = flow_m3_per_s * hose["simulated_s"] / stages.particle_volume_m3
            poured_particles = stages.liquid.particle_count() - 1_048_576
            in_flight = len(stages.toolbox.jet)
            lost = stages.toolbox.jet.lost_parcels
            overflow = stages.toolbox.overflow_m3 / stages.particle_volume_m3
        finally:
            stages.release()
    path = bench_session.record(
        "m1_gate_1m",
        {
            "particles": 1_048_576,
            "size": list(SIZE),
            "gate_fps": GATE_FPS,
            "dam_break": dam,
            "hose": hose,
            "hose_particles_added": poured_particles,
            "hose_particles_released_expected": released,
            "hose_particles_in_flight": in_flight,
            "hose_particles_lost": lost,
            "hose_particles_overflow": overflow,
        },
    )
    print(
        f"\nM1 gate, 1 M particles at {SIZE[0]}x{SIZE[1]}: dam break {dam['fps_median']:.1f} fps "
        f"(p10 {dam['fps_p10']:.1f}), hose {hose['fps_median']:.1f} fps (p10 "
        f"{hose['fps_p10']:.1f}); world speed {dam['world_speed']:.3f} and "
        f"{hose['world_speed']:.3f}; hose poured {poured_particles} of {released:.0f} "
        f"particles ({in_flight} in flight, {lost} lost, {overflow:.0f} over the brim) -> {path}"
    )
    assert frozen is None, f"the NaN guard froze the world: {frozen}"
    assert hose_tool.spec.name == "hose"
    accounted = poured_particles + in_flight + lost + overflow
    assert abs(accounted - released) <= 0.02 * released + 2
    assert poured_particles >= 0.9 * released  # into the tank, not beside it
    assert dam["fps_p10"] >= GATE_FPS
    assert hose["fps_p10"] >= GATE_FPS

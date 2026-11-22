"""M1 performance gate (#57, SPEC 3 and 11): 1 million particles at 30 fps or more.

The game itself runs (`GameStages`): the Test Tank, the liquid, the frame with every pass,
the HUD, at the game's 1600 x 900, with the game's frame loop (60 fps cap, vsync), in a
hidden window. The scenario has two parts.

1. **Dam break.** 1,048,576 particles, the tank's 32 m³ to its rim, released as a step:
   one half 2.375 m deep, the other 1.625 m. Measured over 600 frames.
2. **Hose.** The hose then pours into the moving water at its default 20 L/s and 10 m/s
   from above the rim. Measured over 600 frames more.

Each frame's real time gives the frame rate. Its tenth percentile is the rate held 90 %
of the time, and the gate wants it at 30 fps or more. Each stage's GPU time comes from
the game's own timers (CUDA events and OpenGL queries): the liquid's transfers, pressure
and motion, and every render pass. The first 60 frames of each part are not counted
(warm caches, captured graphs).
"""

import dataclasses
import math
from typing import Any

import numpy as np

from bench.harness import BenchSession
from entropy.app.main import GameStages
from entropy.core.config import default_settings_path, load_settings
from entropy.core.profiler import FrameStats
from entropy.game.free_camera import CameraPose
from entropy.game.input import FrameInput
from entropy.render.context import GameWindow
from entropy.sim.loop import FrameLoop
from entropy.solvers.liquid.solver import fill_box

DX_M = 0.0625
SIZE = (1600, 900)
FRAMES = 600
SKIP = 60
GATE_FPS = 30.0


class _AllStats(FrameStats):
    """The game's statistics, keeping every frame's GPU stage times as well."""

    def __init__(self) -> None:
        super().__init__()
        self.gpu_samples: list[dict[str, float]] = []

    def record_gpu(self, stages_s) -> None:
        super().record_gpu(stages_s)
        self.gpu_samples.append(dict(stages_s))


def _part(loop: FrameLoop, stats: _AllStats, stages: GameStages) -> dict[str, Any]:
    first_gpu = len(stats.gpu_samples)
    frame_ms, substeps, iterations = [], [], []
    for index in range(FRAMES):
        report = loop.run_frame()
        if index >= SKIP:
            frame_ms.append(report.frame_s * 1e3)
            assert stages.liquid is not None
            d = stages.liquid.diagnostics()
            substeps.append(d["substeps"])
            iterations.append(d["pressure_iterations"])
    frame = np.asarray(frame_ms)
    fps = 1e3 / frame
    gpu: dict[str, list[float]] = {}
    for sample in stats.gpu_samples[first_gpu + SKIP :]:
        for name, seconds in sample.items():
            gpu.setdefault(name, []).append(seconds * 1e3)
    return {
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
            # The hose, held from above the rim, aimed down into the tank.
            stages.mode.pose = CameraPose((0.0, -2.6, 3.2), math.radians(90.0), math.radians(-50.0))
            stages.toolbox.handle(
                FrameInput(frozenset(), frozenset(), 0.0, 0.0, tool_pressed=True, tool_held=True)
            )
            stages.toolbox.handle = lambda frame: None  # keep the button held
            hose = _part(loop, stats, stages)
            poured_particles = stages.liquid.particle_count() - 1_048_576
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
        },
    )
    print(
        f"\nM1 gate, 1 M particles at {SIZE[0]}x{SIZE[1]}: dam break {dam['fps_median']:.1f} fps "
        f"(p10 {dam['fps_p10']:.1f}), hose {hose['fps_median']:.1f} fps (p10 "
        f"{hose['fps_p10']:.1f}); GPU ms {dam['gpu_ms_median']} -> {path}"
    )
    assert dam["fps_p10"] >= GATE_FPS
    assert hose["fps_p10"] >= GATE_FPS

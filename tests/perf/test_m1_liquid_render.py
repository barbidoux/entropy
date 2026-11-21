"""M1 measurement: the liquid pass at 1 million particles (issue #52, SPEC 7.2: 2.5 ms).

The Test Tank filled 2 m deep at 8 particles per 6.25 cm cell holds 1,048,576
particles. The whole frame is drawn as the game draws it (`WorldFrame`), at the
game's 1600 x 900, from the game's start camera; OpenGL timer queries time each
pass on the GPU. Ten frames warm up first; the next 60 are measured.
"""

import math

import numpy as np
import warp as wp

from bench.harness import BenchSession
from entropy.core.config import WindowSettings, default_settings_path, load_settings
from entropy.core.paths import content_dir
from entropy.core.profiler import StageTimer
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.gl_timer import GlQueries
from entropy.render.interop import SharedBuffer
from entropy.render.world_frame import WorldFrame
from entropy.solvers.liquid.solver import fill_box
from entropy.world.maps import load_map
from entropy.world.materials import load_material
from entropy.world.world import build_world

DEVICE = "cuda:0"
DX_M = 0.0625
BUDGET_MS = 2.5  # SPEC 7.2, liquid pass
SIZE = (1600, 900)


def test_liquid_pass_at_one_million_particles(bench_session: BenchSession) -> None:
    settings = load_settings(default_settings_path())
    camera = settings.camera
    yaw, pitch = math.radians(camera.start_yaw_deg), math.radians(camera.start_pitch_deg)
    forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    view = CameraView(
        camera.start_position_m,
        forward,
        fov_y_rad=math.radians(camera.fov_y_deg),
        near_m=camera.near_m,
        far_m=camera.far_m,
    )
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    points = fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), DX_M, 8, seed=1)
    assert len(points) == 1_048_576
    window_settings = WindowSettings(width_px=SIZE[0], height_px=SIZE[1], title="bench")
    with GameWindow(window_settings, visible=False) as window:
        ctx = window.ctx
        timer = StageTimer(GlQueries(ctx))
        target = ctx.simple_framebuffer(SIZE)
        with SharedBuffer(ctx, len(points), wp.vec3, DEVICE) as shared:
            with shared.writable() as array:
                wp.copy(array, wp.array(points, dtype=wp.vec3, device=DEVICE))
            frame = WorldFrame(
                ctx, world, shared, particle_volume_m3=DX_M**3 / 8, water=load_material("water")
            )
            times: dict[str, list[float]] = {}
            try:
                bench_session.start("liquid_pass_1m")
                for index in range(70):
                    frame.render(target, view, SIZE, count=len(points), timer=timer)
                    timer.end_frame()
                    for stages in timer.collect():
                        if index >= 10:
                            for name, seconds in stages.items():
                                times.setdefault(name, []).append(seconds * 1e3)
                ctx.finish()
            finally:
                frame.release()
                target.release()

    medians = {name: float(np.median(ms)) for name, ms in times.items()}
    liquid_ms = medians["liquid (OpenGL)"]
    path = bench_session.record(
        "liquid_pass_1m",
        {
            "particles": len(points),
            "size": list(SIZE),
            "median_ms": medians,
            "budget_ms": BUDGET_MS,
        },
    )
    print(
        f"\nliquid pass at 1 M particles, {SIZE[0]}x{SIZE[1]}: {liquid_ms:.2f} ms "
        f"(budget {BUDGET_MS} ms); all passes {medians} -> {path}"
    )
    assert liquid_ms <= BUDGET_MS

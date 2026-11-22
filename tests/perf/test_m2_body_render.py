"""M2 measurement: drawing 5,000 rigid bodies (#91, SPEC 3's rigid capacity, SPEC 7.2).

5,000 bodies, SPEC 3's capacity, stand on a 50 x 100 grid in front of the camera, all
in view: boxes, spheres and capsules in turn (`mix`), or all boxes (`boxes`). The frame
is drawn as the game draws it (`WorldFrame`, Test Tank, no water) at 1920 x 1080, SPEC
3's render target; OpenGL timer queries time the bodies' pass. Ten frames warm up; the
next 60 are measured. The bodies share the solids pass's 2.5 ms (SPEC 7.2) with the
scene.
"""

import math

import numpy as np
import pytest
import warp as wp

from bench.harness import BenchSession
from entropy.core.config import WindowSettings
from entropy.core.paths import content_dir
from entropy.core.profiler import StageTimer
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.gl_timer import GlQueries
from entropy.render.interop import SharedBuffer
from entropy.render.world_frame import WorldFrame
from entropy.solvers.rigid import Body, RigidSetup, RigidSolver, Shape
from entropy.world.maps import load_map
from entropy.world.materials import load_material
from entropy.world.world import build_world

DEVICE = "cuda:0"
BODIES = 5000
SIZE = (1920, 1080)
SOLIDS_BUDGET_MS = 2.5  # SPEC 7.2, the whole solids pass
SIZES = {
    Shape.BOX: (0.12, 0.08, 0.06),
    Shape.SPHERE: (0.08, 0.0, 0.0),
    Shape.CAPSULE: (0.05, 0.06, 0.0),
}


def _bodies(kind: str) -> list[Body]:
    rotate = (math.sin(0.3), 0.0, 0.0, math.cos(0.3))  # turned, so every face shows
    shapes = [Shape.BOX] if kind == "boxes" else [Shape.BOX, Shape.SPHERE, Shape.CAPSULE]
    bodies = []
    for k in range(BODIES):
        row, column = divmod(k, 100)
        shape = shapes[k % len(shapes)]
        position = (0.0, -4.95 + 0.1 * column, 0.2 + 0.1 * row)
        bodies.append(Body(shape, SIZES[shape], 500.0, position_m=position, orientation=rotate))
    return bodies


@pytest.mark.parametrize("kind", ["mix", "boxes"])
def test_drawing_5000_bodies(bench_session: BenchSession, kind: str) -> None:
    world = build_world(load_map(content_dir() / "maps" / "test_tank"))
    solver = RigidSolver.create(
        RigidSetup(gravity_m_per_s2=(0.0, 0.0, -9.81), max_bodies=BODIES), DEVICE
    )
    solver.add_bodies(_bodies(kind))
    colours = wp.array(np.full((BODIES, 3), 0.5, np.float32), dtype=wp.vec3, device=DEVICE)
    view = CameraView(
        (8.0, 0.0, 2.7), (-1.0, 0.0, 0.0), fov_y_rad=math.radians(70.0), near_m=0.05, far_m=500.0
    )
    with GameWindow(
        WindowSettings(width_px=SIZE[0], height_px=SIZE[1], title="bench"), visible=False
    ) as window:
        ctx = window.ctx
        timer = StageTimer(GlQueries(ctx))
        target = ctx.simple_framebuffer(SIZE)
        times: list[float] = []
        with SharedBuffer(ctx, 1, wp.vec3, DEVICE) as shared:
            frame = WorldFrame(
                ctx,
                world,
                shared,
                particle_volume_m3=0.0625**3 / 8,
                water=load_material("water"),
                body_capacity=BODIES,
            )
            try:
                assert frame.bodies is not None
                bench_session.start(f"bodies_5000_{kind}")
                for index in range(70):
                    frame.bodies.update(solver.fields(), BODIES, colours)
                    frame.render(target, view, SIZE, count=0, timer=timer)
                    timer.end_frame()
                    for stages in timer.collect():
                        if index >= 10 and "bodies (OpenGL)" in stages:
                            times.append(stages["bodies (OpenGL)"] * 1e3)
                ctx.finish()
            finally:
                frame.release()
                target.release()
    median_ms = float(np.median(times))
    path = bench_session.record(
        f"bodies_5000_{kind}",
        {"bodies": BODIES, "kind": kind, "size": list(SIZE), "median_ms": median_ms,
         "p90_ms": float(np.percentile(times, 90)), "solids_budget_ms": SOLIDS_BUDGET_MS},
    )  # fmt: skip
    print(f"\n5,000 bodies ({kind}) at {SIZE[0]}x{SIZE[1]}: {median_ms:.3f} ms -> {path}")
    assert median_ms <= SOLIDS_BUDGET_MS / 2  # leaves half the pass to the scene

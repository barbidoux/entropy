"""One frame of the world, pass by pass in SPEC 7.2's order (#51, #52).

1. Solids (`scene`): sky, ground and opaque boxes, in linear light, into an HDR target.
2. Liquid (`liquid`): the water over them, into a second HDR target with its depth.
3. Glass (`world_view`): transparent panels, hidden behind nearer water and solids.
4. Post (`frame`): tone mapping onto the display framebuffer.

The game draws the HUD after this. Each pass can be timed by a `StageTimer`
(SPEC 4.7: GPU time per pass).
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import moderngl

from entropy.core.profiler import StageTimer
from entropy.render.camera import CameraView
from entropy.render.frame import HdrTarget, ToneMapper
from entropy.render.interop import SharedBuffer
from entropy.render.liquid import LiquidRenderer
from entropy.render.scene import SceneRenderer
from entropy.render.world_view import GlassRenderer, glass_panels, ground_height_m, opaque_boxes
from entropy.world.materials import Material
from entropy.world.world import World

PARTICLE_RADIUS_SPACINGS = 0.75
"""Radius of a drawn particle, in particle spacings: the level set's radius (ADR 0018)."""


class WorldFrame:
    """Draws `world` and the water held in `particles` (see the module docstring)."""

    def __init__(
        self,
        ctx: moderngl.Context,
        world: World,
        particles: SharedBuffer,
        *,
        particle_volume_m3: float,
        water: Material,
    ) -> None:
        self.ctx = ctx
        self.scene = SceneRenderer(ctx, opaque_boxes(world), ground_z_m=ground_height_m(world))
        self.glass = GlassRenderer(ctx, glass_panels(world))
        spacing_m = particle_volume_m3 ** (1.0 / 3.0)
        self.liquid = LiquidRenderer(
            ctx,
            particles,
            radius_m=PARTICLE_RADIUS_SPACINGS * spacing_m,
            particle_volume_m3=particle_volume_m3,
            water=water,
        )
        self.tone_map = ToneMapper(ctx)
        self._targets: tuple[HdrTarget, HdrTarget] | None = None

    def _hdr(self, size: tuple[int, int]) -> tuple[HdrTarget, HdrTarget]:
        if self._targets is None or self._targets[0].size_px != size:
            self._release_targets()
            self._targets = (HdrTarget(self.ctx, size), HdrTarget(self.ctx, size))
        return self._targets

    def render(
        self,
        target: moderngl.Framebuffer,
        view: CameraView,
        size_px: tuple[int, int],
        *,
        count: int,
        timer: StageTimer | None = None,
    ) -> None:
        """Draw the frame onto `target` over `size_px`, with `count` particles of water."""
        width, height = size_px
        if width <= 0 or height <= 0 or not math.isfinite(width / height):
            return

        def stage(name: str):
            return timer.stage(name) if timer is not None else nullcontext()

        solids, water = self._hdr(size_px)
        with stage("scene (OpenGL)"):
            self.scene.render(solids.framebuffer, view, size_px)
        with stage("liquid (OpenGL)"):
            self.liquid.render(water, solids, view, count=count)
        with stage("glass (OpenGL)"):
            self.glass.render(water.framebuffer, view, size_px)
        with stage("tone map (OpenGL)"):
            self.tone_map.render(target, water, size_px)

    def _release_targets(self) -> None:
        if self._targets is not None:
            for hdr in self._targets:
                hdr.release()
        self._targets = None

    def release(self) -> None:
        self._release_targets()
        for renderer in (self.tone_map, self.liquid, self.glass, self.scene):
            renderer.release()

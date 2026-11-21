"""The world drawn (`entropy.render.world_view`, #51): opaque solids with their material's
colour, the tank's glass as transparent panels."""

from collections.abc import Iterator

import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.core.paths import content_dir
from entropy.render.camera import CameraView
from entropy.render.context import GameWindow
from entropy.render.frame import HdrTarget, ToneMapper, display_reference
from entropy.render.scene import (
    HORIZON,
    SKY_IRRADIANCE,
    SUN_DIRECTION,
    SUN_IRRADIANCE,
    SceneRenderer,
)
from entropy.render.world_view import (
    GlassRenderer,
    fresnel_f0,
    glass_panels,
    ground_height_m,
    opaque_boxes,
)
from entropy.world.maps import Environment, load_map
from entropy.world.materials import load_material
from entropy.world.sdf import Box, OpenBox
from entropy.world.world import World, build_world

pytestmark = pytest.mark.render

SIZE = (160, 120)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


@pytest.fixture(scope="module")
def test_tank() -> World:
    return build_world(load_map(content_dir() / "maps" / "test_tank"))


def _view(eye_m, target_m) -> CameraView:
    forward = np.asarray(target_m, float) - np.asarray(eye_m, float)
    x, y, z = (float(c) for c in forward / np.linalg.norm(forward))
    return CameraView(eye_m, (x, y, z), fov_y_rad=np.radians(40.0), near_m=0.05, far_m=500.0)


def _render(window: GameWindow, world: World, view: CameraView, *, glass: bool) -> np.ndarray:
    """RGB floats 0-255, row 0 at the top (linear light, then tone-mapped)."""
    hdr = HdrTarget(window.ctx, SIZE)
    target = window.ctx.simple_framebuffer(SIZE)
    scene = SceneRenderer(window.ctx, opaque_boxes(world), ground_z_m=ground_height_m(world))
    panes = GlassRenderer(window.ctx, glass_panels(world))
    mapper = ToneMapper(window.ctx)
    try:
        scene.render(hdr.framebuffer, view, SIZE)
        if glass:
            panes.render(hdr.framebuffer, view, SIZE)
        mapper.render(target, hdr, SIZE)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)
    finally:
        mapper.release()
        panes.release()
        scene.release()
        target.release()
        hdr.release()
    return image.reshape(SIZE[1], SIZE[0], 3)[::-1].astype(float)


def _centre(image: np.ndarray) -> np.ndarray:
    h, w = SIZE[1] // 2, SIZE[0] // 2
    return image[h - 2 : h + 3, w - 2 : w + 3].mean(axis=(0, 1))


def _lit_from_above(albedo: tuple[float, float, float]) -> np.ndarray:
    """The scene's lighting model for an upward face (see test_scene), in display units."""
    direct = max(SUN_DIRECTION[2], 0.0)
    radiance = np.asarray(albedo) * (
        np.asarray(SUN_IRRADIANCE) * direct + np.asarray(SKY_IRRADIANCE)
    )
    return 255.0 * display_reference(radiance)


def test_the_floor_has_its_material_s_colour(window: GameWindow, test_tank: World) -> None:
    """The concrete slab beside the tank, seen from above between its grid lines:
    concrete's albedo, lit."""
    view = _view((5.5, 0.5, 3.0), (5.503, 0.5, -0.02))  # down, 3 m beside the tank

    pixel = _centre(_render(window, test_tank, view, glass=True))

    albedo = load_material("concrete").albedo_linear_rgb
    assert albedo is not None
    np.testing.assert_allclose(pixel, _lit_from_above(albedo), atol=3.0)


BACKGROUND = (0.1, 0.1, 0.1)
"""A uniform radiance behind the glass, linear."""


def _glass_only(window: GameWindow, world: World, view: CameraView) -> np.ndarray:
    """The panels alone, over a uniform background: RGB floats 0-255 at the centre."""
    hdr = HdrTarget(window.ctx, SIZE)
    target = window.ctx.simple_framebuffer(SIZE)
    panes = GlassRenderer(window.ctx, glass_panels(world))
    mapper = ToneMapper(window.ctx)
    try:
        hdr.framebuffer.clear(*BACKGROUND, 1.0, depth=1.0)
        panes.render(hdr.framebuffer, view, SIZE)
        mapper.render(target, hdr, SIZE)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)
    finally:
        mapper.release()
        panes.release()
        target.release()
        hdr.release()
    return _centre(image.reshape(SIZE[1], SIZE[0], 3)[::-1].astype(float))


def test_glass_square_on_reflects_four_percent_per_surface(
    window: GameWindow, test_tank: World
) -> None:
    """Level through both x walls, square on: four surfaces, each reflecting F0 = 4 %
    (n = 1.5) of the horizon behind the camera and passing the rest, in linear light."""
    view = _view((5.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    passed = (1.0 - fresnel_f0(1.5)) ** 4

    pixel = _glass_only(window, test_tank, view)

    radiance = np.asarray(BACKGROUND) * passed + np.asarray(HORIZON) * (1 - passed)
    expected = 255.0 * display_reference(radiance)
    np.testing.assert_allclose(pixel, expected, atol=3.0)


def test_glass_reflects_more_at_a_grazing_angle(window: GameWindow, test_tank: World) -> None:
    """Fresnel: along the +x wall at about 84°, its outer surface alone reflects about
    60 %, so the horizon's reflection outweighs the grey behind."""
    square = _glass_only(window, test_tank, _view((5.0, 0.0, 1.0), (0.0, 0.0, 1.0)))
    grazing = _glass_only(window, test_tank, _view((2.6, -6.0, 1.0), (2.0, 0.0, 1.0)))

    background = 255.0 * display_reference(np.asarray(BACKGROUND))
    assert np.abs(grazing - background).sum() > 3.0 * np.abs(square - background).sum()


def test_fresnel_reflectance_square_on() -> None:
    assert fresnel_f0(1.5) == pytest.approx(0.04)
    assert fresnel_f0(1.0) == 0.0


def test_the_floor_shows_a_metre_grid(window: GameWindow, test_tank: World) -> None:
    """Lines on whole metres give the scale, and show refraction through water."""
    on_line = _centre(
        _render(window, test_tank, _view((5.0, 0.5, 3.0), (5.003, 0.5, -0.02)), glass=False)
    )
    between = _centre(
        _render(window, test_tank, _view((5.5, 0.5, 3.0), (5.503, 0.5, -0.02)), glass=False)
    )

    assert on_line.sum() < between.sum() - 30.0


def test_glass_panels_wrap_the_cavity() -> None:
    """Floor and four walls, wall_m thick, around the cavity, up to the rim."""
    tank = OpenBox(inner_min_m=(-2.0, -1.0, 0.0), inner_max_m=(2.0, 1.0, 1.5), wall_m=0.02)
    world = World("t", Environment(), ((tank, load_material("glass")),))

    panels = glass_panels(world)

    assert len(panels) == 5
    lows = np.array([np.subtract(p.centre_m, p.half_size_m) for p in panels])
    highs = np.array([np.add(p.centre_m, p.half_size_m) for p in panels])
    np.testing.assert_allclose(lows.min(axis=0), (-2.02, -1.02, -0.02), atol=1e-9)
    np.testing.assert_allclose(highs.max(axis=0), (2.02, 1.02, 1.5), atol=1e-9)
    assert all(p.refractive_index == 1.5 for p in panels)
    thin = [min(p.half_size_m) for p in panels]
    np.testing.assert_allclose(thin, 0.01)


def test_the_ground_lies_under_the_lowest_solid(test_tank: World) -> None:
    """The slab's bottom (30 cm under its top at -2 cm): no fight between two floors."""
    assert ground_height_m(test_tank) == pytest.approx(-0.32)
    empty = World("e", Environment(), ())
    assert ground_height_m(empty) == 0.0


def test_opaque_boxes_carry_their_material_s_albedo(test_tank: World) -> None:
    (slab,) = opaque_boxes(test_tank)

    assert slab.albedo_linear_rgb == load_material("concrete").albedo_linear_rgb
    assert slab.grid_m == 1.0
    np.testing.assert_allclose(slab.half_size_m, (10.0, 10.0, 0.15))


def test_a_solid_without_a_colour_is_drawn_neutral_grey() -> None:
    box = Box(centre_m=(0.0, 0.0, 0.5), half_size_m=(0.5, 0.5, 0.5))
    world = World("g", Environment(), ((box, load_material("water")),))

    (drawn,) = opaque_boxes(world)

    assert drawn.albedo_linear_rgb == (0.5, 0.5, 0.5)

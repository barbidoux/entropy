"""The frame's HDR target and tone-mapping pass (`entropy.render.frame`, SPEC 7.2 post)."""

from collections.abc import Iterator

import numpy as np
import pytest

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.render.frame import EXPOSURE, HdrTarget, ToneMapper, display_reference

pytestmark = pytest.mark.render

SIZE = (8, 4)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _tone_map(window: GameWindow, radiance: tuple[float, float, float]) -> np.ndarray:
    """Clear an HDR target to `radiance`, tone-map it into 8 bits, read one pixel."""
    hdr = HdrTarget(window.ctx, SIZE)
    target = window.ctx.simple_framebuffer(SIZE)
    mapper = ToneMapper(window.ctx)
    try:
        hdr.framebuffer.clear(*radiance, 1.0, depth=1.0)
        mapper.render(target, hdr, SIZE)
        image = np.frombuffer(target.read(components=3), dtype=np.uint8)
    finally:
        mapper.release()
        target.release()
        hdr.release()
    return image.reshape(SIZE[1], SIZE[0], 3)[1, 3].astype(float)


@pytest.mark.parametrize("value", [0.0, 0.05, 0.18, 1.0, 4.0])
def test_tone_mapping_is_exposure_aces_and_srgb(window: GameWindow, value: float) -> None:
    pixel = _tone_map(window, (value, value / 2, value / 4))

    expected = 255.0 * display_reference(np.array([value, value / 2, value / 4]))
    np.testing.assert_allclose(pixel, expected, atol=1.5)


def test_the_hdr_target_keeps_light_brighter_than_white(window: GameWindow) -> None:
    """Half floats: 2 and 8 stay apart until the tone curve, which compresses them."""
    dim, bright = _tone_map(window, (2.0, 2.0, 2.0)), _tone_map(window, (8.0, 8.0, 8.0))

    assert bright[0] > dim[0] + 10.0


def test_the_reference_matches_the_documented_curve() -> None:
    """ACES fit (Narkowicz 2015) after the exposure, then the sRGB transfer function."""
    x = 0.18 * EXPOSURE
    aces = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    srgb = 1.055 * aces ** (1 / 2.4) - 0.055

    assert display_reference(np.array([0.18]))[0] == pytest.approx(srgb)

"""The Dear ImGui overlay of the performance HUD (`entropy.ui.hud.PerformanceHud`)."""

from collections.abc import Iterator

import pytest

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.ui.hud import PerformanceHud

pytestmark = pytest.mark.render


@pytest.fixture
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _frames(hud: PerformanceHud, lines: list[str]) -> None:
    """Two frames: Dear ImGui hides an auto-sized window during its first frame."""
    for _ in range(2):
        hud.draw(lines)


def test_hud_draws_its_lines_when_visible(window: GameWindow) -> None:
    hud = PerformanceHud(window.handle)
    try:
        _frames(hud, ["60.0 fps", "frame 16.67 ms"])
        assert hud.last_vertex_count > 0
    finally:
        hud.release()


def test_f3_toggles_the_hud_off_and_on(window: GameWindow) -> None:
    hud = PerformanceHud(window.handle)
    try:
        hud.toggle()
        _frames(hud, ["60.0 fps"])
        hidden = hud.last_vertex_count
        hud.toggle()
        _frames(hud, ["60.0 fps"])

        assert hidden == 0
        assert hud.last_vertex_count > 0
    finally:
        hud.release()


def test_hud_starts_visible_and_release_is_idempotent(window: GameWindow) -> None:
    hud = PerformanceHud(window.handle)

    assert hud.visible
    hud.release()
    hud.release()

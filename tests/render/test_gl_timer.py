"""GPU time of OpenGL render passes (`entropy.render.gl_timer`), SPEC 4.7."""

from collections.abc import Iterator

import pytest

from entropy.core.config import WindowSettings
from entropy.core.profiler import StageTimer
from entropy.render.context import GameWindow
from entropy.render.gl_timer import GlQueries


class FakeQuery:
    def __init__(self, elapsed_ns: int) -> None:
        self.elapsed = elapsed_ns
        self.running = False

    def __enter__(self) -> "FakeQuery":
        self.running = True
        return self

    def __exit__(self, *args: object) -> None:
        self.running = False


class FakeContext:
    def __init__(self) -> None:
        self.made: list[FakeQuery] = []

    def query(self, *, time: bool) -> FakeQuery:
        assert time
        self.made.append(FakeQuery(elapsed_ns=1_500_000))
        return self.made[-1]


def test_a_query_is_read_only_after_its_lag() -> None:
    queries = GlQueries(FakeContext(), lag_frames=3)  # type: ignore[arg-type]
    token = queries.begin()
    queries.end(token)

    assert [queries.ready(token, frames_ago) for frames_ago in range(5)] == [
        False,
        False,
        False,
        True,
        True,
    ]
    assert queries.take_elapsed_s(token) == pytest.approx(0.0015)


def test_queries_bracket_the_pass_and_are_reused() -> None:
    ctx = FakeContext()
    queries = GlQueries(ctx, lag_frames=0)  # type: ignore[arg-type]
    first = queries.begin()
    assert ctx.made[0].running
    queries.end(first)
    assert not ctx.made[0].running
    queries.take_elapsed_s(first)

    queries.end(queries.begin())

    assert len(ctx.made) == 1


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


@pytest.mark.render
def test_an_opengl_pass_gets_a_gpu_time(window: GameWindow) -> None:
    timer = StageTimer(GlQueries(window.ctx, lag_frames=1))
    with timer.stage("clear"):
        window.ctx.clear(0.2, 0.3, 0.4)
    timer.end_frame()
    assert timer.collect() == []
    timer.end_frame()

    (frame,) = timer.collect()

    assert 0.0 <= frame["clear"] < 1.0

"""GPU time of OpenGL render passes (SPEC 4.7, issue #29).

`GlQueries` is a timing backend for `entropy.core.profiler.StageTimer`, built on
OpenGL timer queries (GL_TIME_ELAPSED) through moderngl.

Simplification, named: moderngl cannot ask whether a query's result is ready,
and reading it waits for the GPU. Results are therefore read `LAG_FRAMES`
frames after their frame ended. The driver keeps only a few frames in flight,
so by then the GPU has normally finished them and the read does not wait. If
it has not, the read waits for the GPU: a stall, never a wrong time.
"""

from __future__ import annotations

import moderngl

LAG_FRAMES = 3


class GlQueries:
    """OpenGL timer queries on `ctx`, reused once read."""

    def __init__(self, ctx: moderngl.Context, *, lag_frames: int = LAG_FRAMES) -> None:
        self._ctx = ctx
        self._lag_frames = lag_frames
        self._free: list[moderngl.Query] = []

    def begin(self) -> moderngl.Query:
        query = self._free.pop() if self._free else self._ctx.query(time=True)
        query.__enter__()
        return query

    def end(self, token: moderngl.Query) -> None:
        token.__exit__()  # moderngl's __exit__ takes *args and ignores them

    def ready(self, token: moderngl.Query, frames_ago: int) -> bool:
        return frames_ago >= self._lag_frames

    def take_elapsed_s(self, token: moderngl.Query) -> float:
        elapsed_ns = token.elapsed
        self._free.append(token)
        return elapsed_ns / 1e9

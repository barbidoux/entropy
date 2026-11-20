"""Warp kernels writing OpenGL buffers (`entropy.render.interop`), SPEC 7.1, risk R8."""

import logging
from collections.abc import Iterator

import numpy as np
import pytest
import warp as wp

from entropy.core.config import WindowSettings
from entropy.render import interop
from entropy.render.context import GameWindow
from entropy.render.interop import InteropMode, SharedBuffer

pytestmark = pytest.mark.render

COUNT = 1000


@wp.kernel
def _fill(points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    x = float(i)
    points[i] = wp.vec3(x, 2.0 * x, -x)


def _expected() -> np.ndarray:
    x = np.arange(COUNT, dtype=np.float32)
    return np.stack([x, 2 * x, -x], axis=1)


@pytest.fixture(scope="module")
def window(window_settings: WindowSettings) -> Iterator[GameWindow]:
    with GameWindow(window_settings, visible=False) as w:
        yield w


def _write_and_read(shared: SharedBuffer) -> np.ndarray:
    with shared.writable() as points:
        wp.launch(_fill, dim=COUNT, inputs=[points], device=points.device)
    return np.frombuffer(shared.buffer.read(), dtype=np.float32).reshape(COUNT, 3)


def test_host_copy_on_cpu_reaches_the_gl_buffer(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu")

    assert shared.mode is InteropMode.HOST_COPY
    np.testing.assert_array_equal(_write_and_read(shared), _expected())
    shared.release()


def test_cpu_device_explains_why_it_copies(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu")

    assert shared.fallback_reason is not None
    assert "cpu" in shared.fallback_reason
    shared.release()


def test_zero_copy_cannot_be_forced_on_cpu(window: GameWindow) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu", mode=InteropMode.ZERO_COPY)


def test_buffer_size_matches_count_and_type(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu")

    assert shared.buffer.size == COUNT * 12
    shared.release()


@pytest.mark.gpu
def test_zero_copy_on_cuda_reaches_the_gl_buffer(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0")

    assert shared.mode is InteropMode.ZERO_COPY, shared.fallback_reason
    np.testing.assert_array_equal(_write_and_read(shared), _expected())
    shared.release()


@pytest.mark.gpu
def test_host_copy_can_be_forced_on_cuda(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0", mode=InteropMode.HOST_COPY)

    assert shared.mode is InteropMode.HOST_COPY
    np.testing.assert_array_equal(_write_and_read(shared), _expected())
    shared.release()


@pytest.mark.gpu
def test_failed_registration_falls_back_loudly(
    window: GameWindow, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cudaGraphicsGLRegisterBuffer failed")

    monkeypatch.setattr(interop.wp, "RegisteredGLBuffer", refuse)
    with caplog.at_level(logging.WARNING, logger="entropy.render.interop"):
        shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0")

    assert shared.mode is InteropMode.HOST_COPY
    assert shared.fallback_reason is not None
    assert "cudaGraphicsGLRegisterBuffer failed" in shared.fallback_reason
    assert "host copy" in caplog.text
    np.testing.assert_array_equal(_write_and_read(shared), _expected())
    shared.release()

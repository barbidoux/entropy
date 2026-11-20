"""Warp kernels writing OpenGL buffers (`entropy.render.interop`), SPEC 7.1, risk R8."""

import logging
from collections.abc import Iterator

import glfw
import moderngl
import numpy as np
import pytest
import warp as wp
import warp._src.context as warp_context

from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.render.interop import InteropMode, SharedBuffer

pytestmark = pytest.mark.render

COUNT = 1000


@wp.kernel
def _fill(points: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    x = float(i)  # pyright: ignore[reportArgumentType] (1D tid is an int)
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


def _refuse_native_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CUDA refuse to register OpenGL buffers, at Warp's native boundary."""
    wp.init()  # the native runtime exists only once Warp is initialised
    monkeypatch.setattr(
        warp_context.runtime.core, "wp_cuda_graphics_register_gl_buffer", lambda *args: None
    )


def test_host_copy_on_cpu_reaches_the_gl_buffer(window: GameWindow) -> None:
    with SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu") as shared:
        assert shared.mode is InteropMode.HOST_COPY
        np.testing.assert_array_equal(_write_and_read(shared), _expected())


def test_the_cpu_host_copy_is_logged_with_its_reason(
    window: GameWindow, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        caplog.at_level(logging.WARNING, logger="entropy.render.interop"),
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu") as shared,
    ):
        assert shared.fallback_reason is not None
        assert "cpu" in shared.fallback_reason
        assert "host copy" in caplog.text


def test_zero_copy_cannot_be_forced_on_cpu(window: GameWindow) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu", mode=InteropMode.ZERO_COPY)


def test_buffer_size_matches_count_and_type(window: GameWindow) -> None:
    with SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu") as shared:
        assert shared.buffer.size == COUNT * 12


def test_a_requested_host_copy_says_so_and_release_is_idempotent(window: GameWindow) -> None:
    shared = SharedBuffer(window.ctx, COUNT, wp.vec3, device="cpu", mode=InteropMode.HOST_COPY)

    assert shared.fallback_reason == "host copy requested"
    np.testing.assert_array_equal(_write_and_read(shared), _expected())
    shared.release()
    shared.release()


@pytest.mark.gpu
def test_zero_copy_never_goes_through_host_memory(
    window: GameWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Correct values alone would not prove it: a host copy gives the same values."""
    with SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0") as shared:
        assert shared.mode is InteropMode.ZERO_COPY, shared.fallback_reason

        def no_host_copy(*args: object, **kwargs: object) -> None:
            raise AssertionError("the zero-copy path touched host memory")

        with monkeypatch.context() as patch:
            patch.setattr(moderngl.Buffer, "write", no_host_copy)
            patch.setattr(wp.array, "numpy", no_host_copy)
            with shared.writable() as points:
                wp.launch(_fill, dim=COUNT, inputs=[points], device=points.device)

        read = np.frombuffer(shared.buffer.read(), dtype=np.float32).reshape(COUNT, 3)
        np.testing.assert_array_equal(read, _expected())


@pytest.mark.gpu
def test_host_copy_can_be_forced_on_cuda(window: GameWindow) -> None:
    with SharedBuffer(
        window.ctx, COUNT, wp.vec3, device="cuda:0", mode=InteropMode.HOST_COPY
    ) as shared:
        assert shared.mode is InteropMode.HOST_COPY
        np.testing.assert_array_equal(_write_and_read(shared), _expected())


@pytest.mark.gpu
def test_failed_registration_falls_back_loudly(
    window: GameWindow, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _refuse_native_registration(monkeypatch)

    with (
        caplog.at_level(logging.WARNING, logger="entropy.render.interop"),
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0") as shared,
    ):
        assert shared.mode is InteropMode.HOST_COPY
        assert shared.fallback_reason is not None
        assert "registration failed" in shared.fallback_reason
        assert "host copy" in caplog.text
        np.testing.assert_array_equal(_write_and_read(shared), _expected())


@pytest.mark.gpu
def test_forced_zero_copy_reports_a_failed_registration(
    window: GameWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    _refuse_native_registration(monkeypatch)

    with pytest.raises(RuntimeError):
        SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0", mode=InteropMode.ZERO_COPY)


@pytest.mark.gpu
def test_a_failed_map_is_an_error_not_a_write_through_a_null_pointer(
    window: GameWindow,
) -> None:
    """Without a current GL context the map fails; a kernel must never see ptr 0."""
    with SharedBuffer(window.ctx, COUNT, wp.vec3, device="cuda:0") as shared:
        glfw.make_context_current(None)  # pyright: ignore[reportArgumentType]
        try:
            with pytest.raises(RuntimeError, match="could not map"), shared.writable():
                pass
        finally:
            glfw.make_context_current(window.handle)

    still_working = wp.full(4, 1.0, dtype=float, device="cuda:0")
    assert still_working.numpy().sum() == 4.0

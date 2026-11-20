"""Warp kernels writing OpenGL buffers (SPEC 7.1, risk R8, ADR 0004).

Zero-copy: the OpenGL buffer is registered with CUDA, mapped as a Warp array
for the duration of a write, then unmapped; the data never leaves the GPU.
Host copy: kernels write a separate Warp array, which is then copied to the
OpenGL buffer through host memory (a reused pinned staging array on CUDA).
The host copy is the only path on CPU and the fallback if registration fails
on CUDA; it is never chosen silently: the reason is logged, kept in
`fallback_reason` and shown by the HUD.

Use: release a SharedBuffer (or leave its `with` block) before closing the
window, since unregistering needs the OpenGL context.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from types import TracebackType
from typing import Any

import moderngl
import warp as wp

_log = logging.getLogger(__name__)

_WRITE_DISCARD = wp.RegisteredGLBuffer.WRITE_DISCARD
"""Kernels overwrite the whole buffer each frame; CUDA need not keep the old contents."""


class InteropMode(StrEnum):
    ZERO_COPY = "zero-copy"
    HOST_COPY = "host-copy"


class SharedBuffer:
    """An OpenGL buffer of `count` Warp values of type `dtype`, written by kernels."""

    def __init__(
        self,
        ctx: moderngl.Context,
        count: int,
        dtype: Any,
        device: str,
        *,
        mode: InteropMode | None = None,
    ) -> None:
        wp.init()
        self.count = count
        self.dtype = dtype
        self.device = wp.get_device(device)
        if mode is InteropMode.ZERO_COPY and not self.device.is_cuda:
            raise ValueError("zero-copy interop needs a CUDA device")
        self.fallback_reason: str | None = None
        self._registered: Any = None
        self._scratch: wp.array | None = None
        self._staging: wp.array | None = None
        self._released = False
        self.buffer = ctx.buffer(reserve=count * wp.types.type_size_in_bytes(dtype), dynamic=True)
        try:
            self._choose_path(mode)
        except BaseException:
            self.buffer.release()
            raise

    def _choose_path(self, mode: InteropMode | None) -> None:
        if mode is InteropMode.HOST_COPY:
            self.fallback_reason = "host copy requested"
            _log.info("Interop: host copy, as requested")
        elif not self.device.is_cuda:
            self.fallback_reason = f"device {self.device} is a cpu, not a CUDA GPU"
            _log.warning("Interop: host copy, because %s", self.fallback_reason)
        else:
            try:
                # fallback_to_copy=False: Warp would otherwise copy silently on failure.
                registered = wp.RegisteredGLBuffer(
                    self.buffer.glo, self.device, flags=_WRITE_DISCARD, fallback_to_copy=False
                )
            except RuntimeError as error:
                if mode is InteropMode.ZERO_COPY:
                    raise
                self.fallback_reason = f"CUDA-OpenGL registration failed: {error}"
                _log.warning("Interop: zero copy unavailable, using a host copy: %s", error)
            else:
                if getattr(registered, "resource", None) is None:
                    raise RuntimeError("CUDA-OpenGL registration returned no resource")
                self._registered = registered

        self.mode = InteropMode.ZERO_COPY if self._registered else InteropMode.HOST_COPY
        if self.mode is InteropMode.HOST_COPY:
            self._scratch = wp.empty(self.count, dtype=self.dtype, device=self.device)
            if self.device.is_cuda:
                self._staging = wp.empty(self.count, dtype=self.dtype, device="cpu", pinned=True)

    @contextmanager
    def writable(self) -> Iterator[wp.array]:
        """A Warp array to write this frame's values into; OpenGL sees them on exit.

        Contract: the array is valid only inside the block (its address may change
        from frame to frame, so it must not be kept or captured in a CUDA graph);
        kernels launch on the device's current stream; every element is written,
        since the zero-copy path does not keep last frame's values.
        """
        if self._registered is not None:
            array = self._registered.map(dtype=self.dtype, shape=(self.count,))
            if array.ptr == 0:
                # Warp ignores the native map error and returns a null array.
                raise RuntimeError(
                    "could not map the OpenGL buffer for CUDA: is its OpenGL context "
                    "current on this thread?"
                )
            try:
                yield array
            finally:
                self._registered.unmap()
        else:
            assert self._scratch is not None
            yield self._scratch
            if self._staging is not None:
                wp.copy(self._staging, self._scratch)
                wp.synchronize_device(self.device)
                self.buffer.write(self._staging.numpy())
            else:
                self.buffer.write(self._scratch.numpy())

    def release(self) -> None:
        """Unregister from CUDA, then free the OpenGL buffer. Safe to call twice."""
        if self._released:
            return
        self._released = True
        self._registered = None  # Warp unregisters when its object is collected
        self._scratch = None
        self._staging = None
        self.buffer.release()

    def __enter__(self) -> SharedBuffer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

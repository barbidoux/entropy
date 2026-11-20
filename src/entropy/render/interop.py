"""Warp kernels writing OpenGL buffers (SPEC 7.1, risk R8, ADR 0004).

Zero-copy: the OpenGL buffer is registered with CUDA, mapped as a Warp array
for the duration of a write, then unmapped; the data never leaves the GPU.
Host copy: kernels write a separate Warp array, which is then copied to the
OpenGL buffer through host memory. The host copy is the only path on CPU and
the fallback if registration fails on CUDA; it is never chosen silently: the
reason is logged, kept in `fallback_reason` and shown by the HUD.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
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
        self.buffer = ctx.buffer(reserve=count * wp.types.type_size_in_bytes(dtype), dynamic=True)
        self.fallback_reason: str | None = None
        self._registered: Any = None
        self._scratch: wp.array | None = None

        if mode is InteropMode.ZERO_COPY and not self.device.is_cuda:
            raise ValueError("zero-copy interop needs a CUDA device")
        if mode is InteropMode.HOST_COPY:
            self.fallback_reason = "host copy requested"
        elif not self.device.is_cuda:
            self.fallback_reason = f"device {device} is a cpu, not a CUDA GPU"
        else:
            try:
                # fallback_to_copy=False: Warp would otherwise copy silently on failure.
                self._registered = wp.RegisteredGLBuffer(
                    self.buffer.glo,
                    self.device,
                    flags=_WRITE_DISCARD,
                    fallback_to_copy=False,
                )
            except Exception as error:
                if mode is InteropMode.ZERO_COPY:
                    raise
                self.fallback_reason = f"CUDA-OpenGL registration failed: {error}"
                _log.warning("Zero-copy interop unavailable, using a host copy: %s", error)

        self.mode = InteropMode.ZERO_COPY if self._registered else InteropMode.HOST_COPY
        if self.mode is InteropMode.HOST_COPY:
            self._scratch = wp.empty(count, dtype=dtype, device=self.device)

    @contextmanager
    def writable(self) -> Iterator[wp.array]:
        """A Warp array to write this frame's values into; OpenGL sees them on exit."""
        if self._registered is not None:
            array = self._registered.map(dtype=self.dtype, shape=(self.count,))
            try:
                yield array
            finally:
                self._registered.unmap()
        else:
            assert self._scratch is not None
            yield self._scratch
            self.buffer.write(self._scratch.numpy().tobytes())

    def release(self) -> None:
        self._registered = None
        self._scratch = None
        self.buffer.release()

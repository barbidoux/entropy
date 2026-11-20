"""The game window and its OpenGL core context (SPEC 7.1, ADR 0004).

GLFW opens a native Windows window with an OpenGL core context of the version
in the settings (4.6 by default); moderngl wraps that context. ENTROPY uses a
single window, so this class also owns GLFW's process-wide initialisation and
terminates GLFW when the window closes.
"""

from __future__ import annotations

import logging
import warnings
from types import TracebackType
from typing import Any

import glfw
import moderngl

from entropy.core.config import WindowSettings

_log = logging.getLogger(__name__)


class GameWindow:
    """A GLFW window with a current moderngl context."""

    def __init__(self, settings: WindowSettings, *, visible: bool = True) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW could not start: no display or no OpenGL driver")
        major, minor = settings.gl_version
        glfw.default_window_hints()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, visible)
        with warnings.catch_warnings(record=True) as glfw_errors:
            warnings.simplefilter("always", glfw.GLFWError)
            handle = glfw.create_window(
                settings.width_px, settings.height_px, settings.title, None, None
            )
        if not handle:
            glfw.terminate()
            reasons = "; ".join(str(error.message) for error in glfw_errors) or "no details"
            raise RuntimeError(
                f"no OpenGL {major}.{minor} core context on this machine (GLFW: {reasons})"
            )
        for error in glfw_errors:
            _log.warning("GLFW: %s", error.message)
        self._handle = handle
        glfw.make_context_current(handle)
        glfw.swap_interval(1 if settings.vsync else 0)
        self.ctx = moderngl.create_context(require=major * 100 + minor * 10)
        info = self.ctx.info
        _log.info(
            "OpenGL %s on %s (%s)",
            info["GL_VERSION"],
            info["GL_RENDERER"],
            info["GL_VENDOR"],
        )

    @property
    def handle(self) -> Any:  # GLFW's ctypes window pointer
        """The GLFW window, for input callbacks."""
        if self._handle is None:
            raise RuntimeError("the window is closed")
        return self._handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def should_close(self) -> bool:
        return self._handle is None or bool(glfw.window_should_close(self._handle))

    def framebuffer_size_px(self) -> tuple[int, int]:
        width, height = glfw.get_framebuffer_size(self.handle)
        return int(width), int(height)

    def poll_events(self) -> None:
        glfw.poll_events()

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self.handle)

    def set_mouse_captured(self, captured: bool) -> None:
        """Hide and lock the cursor for mouse look, or give it back."""
        mode = glfw.CURSOR_DISABLED if captured else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self.handle, glfw.CURSOR, mode)
        if captured and glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(self.handle, glfw.RAW_MOUSE_MOTION, True)

    def close(self) -> None:
        """Release the context and the window, then GLFW. Safe to call twice."""
        if self._handle is None:
            return
        self.ctx.release()
        glfw.destroy_window(self._handle)
        self._handle = None
        glfw.terminate()

    def __enter__(self) -> GameWindow:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

"""The performance HUD, toggled with F3 (SPEC 4.5, 4.7).

`hud_lines` turns frame statistics into text; `PerformanceHud` draws that text
as a Dear ImGui overlay in the top-left corner. The HUD only reads state.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from types import TracebackType
from typing import Any

from imgui_bundle import imgui

from entropy.core.profiler import FrameSummary, GpuMemory

_GIB = 2**30
SLOWDOWN_PREFIX = "world slowed"
_WARNING_COLOUR = (1.0, 0.65, 0.2, 1.0)

_NAME_WIDTH = 20
"""Stage names are padded to this width, so CPU and GPU times line up."""


def hud_lines(
    summary: FrameSummary,
    memory: GpuMemory | None,
    *,
    interop: str,
    time_scale: float = 1.0,
    notes: Sequence[str] = (),
) -> list[str]:
    """The HUD text: rate, frame time, time per stage, VRAM, interop path, notes.

    Stage times come twice, labelled: CPU wall times, and GPU times measured by
    CUDA events and OpenGL timer queries, which arrive a few frames late (see
    `entropy.core.profiler`). When frames
    lose more than 5 % of the simulated time they asked for (frames below
    30 fps hit the 1/30 s clamp, SPEC 5.9), a line starting with
    `SLOWDOWN_PREFIX` says so; pausing or slow motion is not a slowdown.
    """
    lines = [
        f"{summary.fps:.1f} fps",
        f"frame {summary.frame_ms_mean:.2f} ms (worst {summary.frame_ms_worst:.2f} ms)",
    ]
    lines.append("CPU ms per stage")
    lines += [
        f"  {stage:<{_NAME_WIDTH}}{ms:6.2f} ms" for stage, ms in summary.stages_ms_mean.items()
    ]
    if summary.gpu_ms_mean:
        lines.append("GPU ms per stage")
        lines += [
            f"  {stage:<{_NAME_WIDTH}}{ms:6.2f} ms" for stage, ms in summary.gpu_ms_mean.items()
        ]
    else:
        lines.append("GPU ms per stage: none yet")
    if memory is None:
        lines.append("VRAM n/a (no CUDA device)")
    else:
        warp = "n/a" if memory.warp_bytes is None else f"{memory.warp_bytes / _GIB:.2f} GiB"
        lines.append(
            f"VRAM {memory.device_used_bytes / _GIB:.2f} / {memory.device_total_bytes / _GIB:.2f}"
            f" GiB (ENTROPY {warp})"
        )
    lines.append(f"interop {interop}")
    if time_scale == 0.0:
        lines.append("paused")
    elif time_scale != 1.0:
        lines.append(f"time scale {time_scale:.2f}x")
    if summary.frames and summary.time_kept < 0.95:
        lines.append(
            f"{SLOWDOWN_PREFIX}: {summary.time_kept:.0%} of the set speed".replace("%", " %")
        )
    lines += list(notes)
    return lines


class PerformanceHud:
    """A Dear ImGui overlay on a GLFW window, drawn into the current framebuffer.

    Release it (or leave its `with` block) before the window closes: the
    backends need the window and its OpenGL context.
    """

    def __init__(self, glfw_window: Any, *, glsl_version: str = "#version 450") -> None:
        address = ctypes.cast(glfw_window, ctypes.c_void_p).value if glfw_window else None
        if address is None:
            raise ValueError("the HUD needs an open GLFW window")
        self.visible = True
        self._released = False
        self._context = imgui.create_context()
        io = imgui.get_io()
        io.set_ini_filename("")  # no imgui.ini next to the game
        # The HUD takes no input, so it must not change the cursor the game manages.
        io.config_flags |= imgui.ConfigFlags_.no_mouse_cursor_change
        try:
            imgui.backends.glfw_init_for_opengl(address, False)
            try:
                imgui.backends.opengl3_init(glsl_version)
            except BaseException:
                imgui.backends.glfw_shutdown()
                raise
        except BaseException:
            imgui.destroy_context(self._context)
            raise
        try:
            self._scale_for(glfw_window)
        except BaseException:
            self.release()
            raise

    def _scale_for(self, glfw_window: Any) -> None:
        """Scale the text with the display's scaling (150 % in Windows: 1.5)."""
        import glfw  # loaded by the window already; kept local to `ui`

        x_scale, _ = glfw.get_window_content_scale(glfw_window)
        imgui.get_style().font_scale_dpi = max(1.0, float(x_scale))

    def toggle(self) -> None:
        self.visible = not self.visible

    def draw(self, lines: Sequence[str]) -> None:
        """Draw this frame's HUD over whatever the current framebuffer holds."""
        imgui.set_current_context(self._context)
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        if self.visible:
            imgui.set_next_window_pos((10.0, 10.0))
            imgui.set_next_window_bg_alpha(0.6)
            flags = (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.always_auto_resize
                | imgui.WindowFlags_.no_inputs
                | imgui.WindowFlags_.no_saved_settings
            )
            imgui.begin("Performance", None, flags)
            for line in lines:
                if line.startswith(SLOWDOWN_PREFIX):
                    imgui.text_colored(_WARNING_COLOUR, line)
                else:
                    imgui.text_unformatted(line)
            imgui.end()
        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

    def release(self) -> None:
        if self._released:
            return
        imgui.set_current_context(self._context)
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context(self._context)
        self._released = True

    def __enter__(self) -> PerformanceHud:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

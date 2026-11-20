"""The performance HUD, toggled with F3 (SPEC 4.5, 4.7).

`hud_lines` turns frame statistics into text; `PerformanceHud` draws that text
as a Dear ImGui overlay in the top-left corner. The HUD only reads state.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

from imgui_bundle import imgui

from entropy.core.profiler import FrameSummary, GpuMemory

_GIB = 2**30


def hud_lines(
    summary: FrameSummary,
    memory: GpuMemory | None,
    *,
    interop: str,
    notes: Sequence[str] = (),
) -> list[str]:
    """The HUD text: rate, frame time, time per stage, VRAM, interop path, notes."""
    lines = [
        f"{summary.fps:.1f} fps",
        f"frame {summary.frame_ms_mean:.2f} ms (worst {summary.frame_ms_worst:.2f} ms)",
    ]
    lines += [f"  {stage:<11}{ms:6.2f} ms" for stage, ms in summary.stages_ms_mean.items()]
    if memory is None:
        lines.append("VRAM n/a (no CUDA device)")
    else:
        lines.append(
            f"VRAM {memory.device_used_bytes / _GIB:.2f} / {memory.device_total_bytes / _GIB:.2f}"
            f" GiB (ENTROPY {memory.warp_bytes / _GIB:.2f} GiB)"
        )
    lines.append(f"interop {interop}")
    lines += list(notes)
    return lines


class PerformanceHud:
    """A Dear ImGui overlay on a GLFW window, drawn into the current framebuffer."""

    def __init__(self, glfw_window: Any, *, glsl_version: str = "#version 450") -> None:
        self.visible = True
        self.last_vertex_count = 0
        self._context = imgui.create_context()
        imgui.get_io().set_ini_filename("")  # no imgui.ini next to the game
        address = ctypes.cast(glfw_window, ctypes.c_void_p).value
        if address is None:
            raise ValueError("the HUD needs an open GLFW window")
        imgui.backends.glfw_init_for_opengl(address, False)
        imgui.backends.opengl3_init(glsl_version)
        self._released = False

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
                imgui.text_unformatted(line)
            imgui.end()
        imgui.render()
        draw_data = imgui.get_draw_data()
        self.last_vertex_count = draw_data.total_vtx_count
        imgui.backends.opengl3_render_draw_data(draw_data)

    def release(self) -> None:
        if self._released:
            return
        imgui.set_current_context(self._context)
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context(self._context)
        self._released = True

"""The window and the Dear ImGui backend use one GLFW library (ADR 0011).

imgui-bundle ships the GLFW its ImGui backend is linked against. Importing
`entropy` points pyGLFW at that library (PYGLFW_LIBRARY) without importing
imgui-bundle. Checked in fresh interpreters, in both import orders, by
counting the GLFW libraries actually loaded and by starting the HUD on a
game window.
"""

import subprocess
import sys

import pytest

_PROBE = """
import ctypes, os, sys
{imports}
from entropy.core.config import WindowSettings
from entropy.render.context import GameWindow
from entropy.ui.hud import PerformanceHud

def glfw_libraries():
    if sys.platform == "win32":
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32EnumProcessModules.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.K32GetModuleFileNameExW.argtypes = [
            wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD,
        ]
        process = kernel32.GetCurrentProcess()
        modules = (wintypes.HMODULE * 1024)()
        needed = wintypes.DWORD()
        size = ctypes.sizeof(modules)
        kernel32.K32EnumProcessModules(process, modules, size, ctypes.byref(needed))
        names = []
        for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
            name = ctypes.create_unicode_buffer(1024)
            kernel32.K32GetModuleFileNameExW(process, module, name, 1024)
            names.append(name.value.lower())
        return sorted(n for n in names if n.endswith("glfw3.dll"))
    with open("/proc/self/maps") as maps:
        return sorted(set(line.split()[-1] for line in maps if "libglfw" in line))

major, minor = os.environ.get("ENTROPY_TEST_GL_VERSION", "4.6").split(".")
version = (int(major), int(minor))
settings = WindowSettings(width_px=64, height_px=64, vsync=False, gl_version=version)
with GameWindow(settings, visible=False) as window:
    hud = PerformanceHud(window.handle)
    hud.draw(["probe"])
    hud.release()
print(len(glfw_libraries()))
"""


@pytest.mark.render
@pytest.mark.parametrize(
    "imports",
    ["import entropy\nimport glfw", "import glfw\nimport entropy"],
    ids=["entropy-first", "glfw-first"],
)
def test_one_glfw_library_whatever_the_import_order(imports: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(imports=imports)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "1"


def test_importing_entropy_does_not_load_imgui_bundle() -> None:
    """Start-up cost: only the code that draws the HUD loads imgui-bundle."""
    code = "import sys, entropy.core.config; print('imgui_bundle' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )

    assert result.stdout.strip() == "False"

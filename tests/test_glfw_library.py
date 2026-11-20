"""pyGLFW and the Dear ImGui backend share one GLFW library (SPEC 4.7, ADR 0004).

imgui-bundle ships the GLFW its ImGui backend is linked against and, when
imported, points pyGLFW at it (`PYGLFW_LIBRARY`). With two GLFW libraries in
one process the HUD would drive a window that its GLFW never created.
"""

import subprocess
import sys
from pathlib import Path

import entropy


def test_importing_entropy_points_pyglfw_at_the_imgui_bundle_glfw() -> None:
    """In a fresh interpreter, so no earlier import hides the order."""
    code = (
        "import entropy, glfw, imgui_bundle, pathlib;"
        "print(pathlib.Path(glfw._glfw._name).parent == pathlib.Path(imgui_bundle.__file__).parent)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )

    assert result.stdout.strip() == "True"


def test_package_is_importable() -> None:
    assert Path(entropy.__file__).name == "__init__.py"

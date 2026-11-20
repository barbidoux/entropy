"""ENTROPY: real-time physics sandbox."""

import os
import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


def _point_pyglfw_at_imgui_bundle_glfw() -> None:
    """Make pyGLFW load the GLFW library imgui-bundle ships (ADR 0011).

    Only sets PYGLFW_LIBRARY, before any `import glfw`, without importing
    imgui-bundle; an existing value is kept.
    """
    spec = find_spec("imgui_bundle")
    if spec is None or spec.origin is None or os.environ.get("PYGLFW_LIBRARY"):
        return
    names = {"win32": ("glfw3.dll",), "darwin": ("libglfw.3.dylib",)}
    for name in names.get(sys.platform, ("libglfw.so.3",)):
        library = Path(spec.origin).parent / name
        if library.is_file():
            os.environ["PYGLFW_LIBRARY"] = str(library)
            return


_point_pyglfw_at_imgui_bundle_glfw()

__version__: str = version("entropy")

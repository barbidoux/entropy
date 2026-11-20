"""ENTROPY: real-time physics sandbox."""

from importlib.metadata import version

# imgui-bundle ships the GLFW library its Dear ImGui backend is linked against and,
# on import, points pyGLFW at it (PYGLFW_LIBRARY). It must come before any
# `import glfw`, so the window and the HUD share one GLFW (ADR 0004).
import imgui_bundle  # noqa: F401  # isort: skip

__version__: str = version("entropy")

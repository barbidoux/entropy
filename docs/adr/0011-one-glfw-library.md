# ADR 0011: One GLFW library, imgui-bundle's, chosen without importing imgui-bundle

- Status: accepted
- Date: 2026-11-20

## Context

The window uses pyGLFW (`import glfw`), and the HUD uses imgui-bundle's Dear ImGui GLFW
backend (ADR 0004). Both packages ship a GLFW library: pyGLFW its own build (GLFW 3.4.0
here), imgui-bundle the build its backend is linked against (GLFW 3.5.1 here). The HUD
backend must drive the window GLFW created.

On Windows the loader reuses an already loaded DLL of the same module name, so whichever
`glfw3.dll` loads first serves both; Linux behaves the same by soname. Importing
imgui-bundle makes it point pyGLFW at its own library (`PYGLFW_LIBRARY`), but importing it
costs about 0.3 s and loads its whole native module, which every `entropy` import would
pay if it sat in the package root.

## Decision

- `entropy/__init__.py` sets `PYGLFW_LIBRARY` to imgui-bundle's GLFW library, found with
  `importlib.util.find_spec` without importing imgui-bundle, unless the variable is
  already set. Any `import entropy...` therefore fixes the choice before `glfw` loads.
- Only `entropy.ui.hud` imports imgui-bundle.
- A test starts a fresh interpreter in both import orders (`entropy` first, `glfw` first),
  opens a window, starts the HUD, and checks that exactly one GLFW library is loaded.
  Another checks that importing `entropy` does not load imgui-bundle.

## Alternatives considered

- **Import imgui-bundle in the package root:** correct, but 0.3 s and a native module on
  every import, including the CPU test suite and the benchmark harness.
- **Leave the order to chance:** works today because of DLL reuse, but the GLFW version
  in use would then depend on which module a script happens to import first.

## Consequences

- The game always uses imgui-bundle's GLFW build, the one its backend was compiled for.
- `PYGLFW_LIBRARY` is visible to child processes started by ENTROPY.

## How we would know it is wrong

The library-count test reports two GLFW libraries, or a pyGLFW or imgui-bundle upgrade
changes how the library is chosen.

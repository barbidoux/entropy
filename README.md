# ENTROPY

A real-time 3D physics sandbox where water, fire, smoke, explosions and destructible
structures obey physical laws and act on each other, at the scale of one house.

> Flood a house, set it on fire, blow it up, and watch every drop, flame and brick
> behave as it would in reality.

**Status:** milestone M0 (Foundations): a window, a free camera, an empty lit scene, the
performance HUD, and the measurements the solver budgets rest on. No physics yet; the
water tank comes with M1.

## Playing

On the laptop, from the clone:

```powershell
uv run entropy              # open the game
uv run entropy-shortcut     # once: put an ENTROPY shortcut on the desktop
```

| Action | Keys (set in `content/settings.toml`) |
| --- | --- |
| Look around | click in the window, then move the mouse; Esc gives the mouse back |
| Fly | Z Q S D, set for an AZERTY keyboard; on QWERTY write W A S D in `[controls]` |
| Up, down | Space, Ctrl |
| Performance HUD | F3 |
| Quit | close the window |

## Documents

- [Specification](docs/SPEC.md): vision, physics, architecture, test strategy, roadmap.
- [Architecture decision records](docs/adr/)
- [Benchmarks](docs/benchmarks/)
- [Changelog](CHANGELOG.md)

## Stack

Python 3.12, [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels on CPU and GPU,
OpenGL 4.6 through moderngl and GLFW with zero-copy CUDA interop, Dear ImGui through
imgui-bundle (ADR 0004), developed test-first.

Target machine: Windows laptop with an NVIDIA RTX 4090 Laptop GPU (16 GB), run natively,
not in WSL.

## Developing

```bash
uv sync                     # create the environment
uv run pytest               # CPU suite (with render tests): runs anywhere, must always pass
xvfb-run -a uv run pytest   # the same on Linux without a display
uv run pytest -m gpu        # GPU suite: on a machine with a CUDA device
uv run pytest -m perf       # benchmarks, laptop only (docs/benchmarks/README.md)
uv run pytest -m soak       # long stability run, laptop only
uv run ruff check .         # lint
uv run pyright              # types
```

Every change follows the loop in [SPEC section 10](docs/SPEC.md#10-development-workflow):
failing test first, code until green, pull request, CI, independent review, squash merge.

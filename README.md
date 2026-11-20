# ENTROPY

A real-time 3D physics sandbox where water, fire, smoke, explosions and destructible
structures obey physical laws and act on each other, at the scale of one house.

> Flood a house, set it on fire, blow it up, and watch every drop, flame and brick
> behave as it would in reality.

**Status:** specification done, milestone M0 (Foundations) next. Nothing is playable yet.

## Documents

- [Specification](docs/SPEC.md): vision, physics, architecture, test strategy, roadmap.
- [Architecture decision records](docs/adr/)
- [Changelog](CHANGELOG.md)

## Stack

Python 3.12, [NVIDIA Warp](https://github.com/NVIDIA/warp) kernels on CPU and GPU,
OpenGL 4.6 rendering (from M0), developed test-first.

Target machine: Windows laptop with an NVIDIA RTX 4090 Laptop GPU (16 GB), run natively,
not in WSL.

## Developing

```bash
uv sync                 # create the environment
uv run pytest           # CPU suite: runs anywhere, must always pass
uv run pytest -m gpu    # GPU suite: on a machine with a CUDA device
uv run ruff check .     # lint
uv run pyright          # types
```

Every change follows the loop in [SPEC section 10](docs/SPEC.md#10-development-workflow):
failing test first, code until green, pull request, CI, independent review, squash merge.

# ENTROPY — maintainer guide

This file is the operating manual for any Claude session working on this repository.
Read it fully before changing anything. The specification is `docs/SPEC.md`, the
reference copy (ADR 0006). The Claude doc linked at its top is where it started; never
re-export it over SPEC.md. Matt tells the maintainer when he changes it there.

## Who does what

- **Matt** is the product owner and play-tester. He decides priorities, plays each
  milestone and says whether it is fun. He does not read or review code: never ask him
  to. Talk to him in French, informally ("tu"), in plain words, with a short note on what
  to try when there is something to play.
- **Claude (main session)** is the maintainer: writes tests and code, runs the suites,
  opens and merges pull requests, keeps the spec, ADRs and changelog current.
- **Independent reviewer**: a separate agent with a fresh context reviews every pull
  request against the spec and the tests before merge. The session that wrote a change
  never approves it. Findings are fixed or answered before merging.
- **Physics auditor**: once per milestone, a separate agent re-derives the validation
  scenarios and tolerances (SPEC 9.4) from the references.

## Non-negotiable rules

1. **TDD.** No production code without a failing test written first and seen failing.
   Red, green, refactor. Physics features start from their validation test (SPEC 9.4).
2. **Device-agnostic.** Every allocation takes a `device`. Solver tests are parametrized
   over available devices. The whole default suite must pass on CPU.
3. **SI units, Z up, kelvin.** Units in docstrings; ambiguous names carry the unit
   (`dt_s`, `temperature_k`). Gravity is 9.81 m/s² along −Z.
4. **No hard-coded material constants.** Constants live in `content/materials/*.toml`
   with a `source` field; tests read them from there.
5. **Dependencies point down** (SPEC 8.1). Renderer, UI and exporter never write
   simulation state.
6. **No global mutable state.** Configuration is immutable dataclasses loaded from TOML.
7. **Never fake physics silently.** A simplification is named in the spec and in code.
8. **NaN guard.** Every solver checks its state after stepping; a dump becomes a
   regression test.

## The change loop

1. Take the next issue of the current milestone (GitHub issues, one per user story).
2. Branch: `feat/…`, `fix/…`, `docs/…`, `test/…`.
3. Write the failing test, run `uv run pytest`, see it fail.
4. Code until green, refactor.
5. `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run pyright`
6. Open a pull request (Conventional Commits title). CI must be green.
7. Kernel changes: run `uv run pytest -m gpu` and benchmarks on the laptop, attach results.
8. Independent review by a separate agent; fix findings.
9. Squash merge; add a `CHANGELOG.md` entry; write an ADR for any decision.
10. Milestone gate: soak test, performance report, play session for Matt.

## Commands

```bash
uv sync                     # environment (Python 3.12)
uv run pytest               # CPU suite (default markers exclude gpu, perf, soak)
uv run pytest -m gpu        # GPU suite, laptop only
uv run pytest -m perf       # benchmarks against SPEC section 3 budgets, laptop only
uv run coverage run -m pytest && uv run coverage report   # 90 % floor, as in CI
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## The laptop

- Dell Precision 7780, RTX 4090 Laptop GPU (16 GB), i9-13950HX, 64 GB RAM, Windows.
- The game runs natively on Windows, never in WSL (CUDA–OpenGL interop).
- Claude reaches it through the Claude desktop app once Matt connects the clone's folder.
  Suggested location: `GITREPOS\entropy` in his home folder (open decision, SPEC 13).
- Run benchmarks plugged in, in performance power mode, after a 5-minute warm-up.

## Where things are

- `docs/SPEC.md` — specification, the reference copy (ADR 0006).
- `docs/adr/` — architecture decision records, numbered, never edited once accepted.
- `CHANGELOG.md` — one entry per merged change.
- `src/entropy/` — the package; layout in SPEC 10.2.
- `tests/` — unit, property, validation, golden, render, gameplay, perf.

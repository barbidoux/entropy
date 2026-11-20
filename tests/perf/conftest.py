"""Session set-up for `uv run pytest -m perf` (laptop only, method in ADR 0007).

Every test under `tests/perf/` is a benchmark: it is marked `perf` here, so
it never runs in the default suite or CI, and it waits for the session
warm-up. The warm-up loads every kernel module first, then runs a streaming
load for `ENTROPY_WARMUP_S` seconds (default 300, SPEC section 3). A shorter
warm-up is fine while developing a benchmark, but its results are unofficial
and go to `out/benchmarks/` instead of `docs/benchmarks/results/`.
"""

import os
import warnings
from pathlib import Path

import pytest
import warp as wp

from bench.harness import (
    WARMUP_REQUIRED_S,
    BenchSession,
    Conditions,
    read_power_state,
    warm_up,
)

REPO_ROOT = Path(__file__).parents[2]
PERF_DIR = Path(__file__).parent
DEVICE = "cuda:0"
REHEAT_S = 10.0


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if PERF_DIR in item.path.parents:
            item.add_marker(pytest.mark.perf)


@pytest.fixture(scope="session")
def bench_session() -> BenchSession:
    """Warmed-up GPU and the conditions every result of this session starts from."""
    assert wp.is_cuda_available(), "Benchmarks need the laptop's CUDA GPU."
    warmup_target_s = float(os.environ.get("ENTROPY_WARMUP_S", WARMUP_REQUIRED_S))
    wp.force_load(device=DEVICE)  # compile now, so the GPU does not cool down later
    warmup_s = warm_up(warmup_target_s, device=DEVICE)
    power = read_power_state()
    conditions = Conditions(
        on_ac=power.on_ac,
        power_mode=power.power_mode,
        warmup_s=warmup_s,
        energy_saver=power.energy_saver,
    )
    if not conditions.official:
        reasons = "; ".join(conditions.reasons_not_official())
        warnings.warn(f"Unofficial benchmark session ({reasons})", stacklevel=1)
    return BenchSession(conditions=conditions, repo_root=REPO_ROOT)


@pytest.fixture(autouse=True)
def _warmed_up(bench_session: BenchSession) -> None:
    """Every benchmark runs after the warm-up, then a short re-heat.

    The re-heat (`ENTROPY_REHEAT_S`, default 10 s) brings the GPU back to its
    sustained state after a benchmark that barely loaded it.
    """
    warm_up(float(os.environ.get("ENTROPY_REHEAT_S", REHEAT_S)), device=DEVICE)

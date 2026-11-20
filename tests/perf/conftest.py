"""Session set-up for `uv run pytest -m perf` (laptop only).

The first benchmark of a session triggers the warm-up: the GPU runs a
streaming load for `ENTROPY_WARMUP_S` seconds (default 300, SPEC section 3).
A shorter warm-up is fine while developing a benchmark, but its results are
unofficial and go to `out/benchmarks/` instead of `docs/benchmarks/results/`.
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
DEVICE = "cuda:0"


@pytest.fixture(scope="session")
def bench_session() -> BenchSession:
    """Warmed-up GPU and the conditions every result of this session is recorded with."""
    assert wp.is_cuda_available(), "Benchmarks need the laptop's CUDA GPU."
    warmup_target_s = float(os.environ.get("ENTROPY_WARMUP_S", WARMUP_REQUIRED_S))
    power = read_power_state()
    warmup_s = warm_up(warmup_target_s, device=DEVICE)
    conditions = Conditions(on_ac=power.on_ac, power_mode=power.power_mode, warmup_s=warmup_s)
    if not conditions.official:
        reasons = "; ".join(conditions.reasons_not_official())
        warnings.warn(f"Unofficial benchmark session ({reasons})", stacklevel=1)
    return BenchSession(conditions=conditions, repo_root=REPO_ROOT)

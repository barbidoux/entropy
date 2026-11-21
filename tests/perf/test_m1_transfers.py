"""M1 measurement: the APIC transfers at 1 million particles, in random order and sorted
by cell (issue #65, SPEC 3's liquid row).

The Test Tank's grid (72 x 72 x 40 cells of 6.25 cm) holds 1,048,576 particles 2 m
deep at 8 per cell, as in the liquid pass's benchmark. Particles get random velocities
and affine matrices, so every weight is used. The passes are timed interleaved, round
by round (ADR 0022): particles to grid and grid to particles for each order, and the
sort itself (`CellSort`, once per frame in the liquid).
"""

import numpy as np

from bench.harness import BenchSession, summarize, time_interleaved
from entropy.fields.mac import GridSpec, MacVelocity
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.solver import fill_box
from entropy.solvers.liquid.sorting import CellSort
from entropy.solvers.liquid.transfer import grid_to_particles, particles_to_grid

DEVICE = "cuda:0"
DX_M = 0.0625
SPEC = GridSpec(shape=(72, 72, 40), dx_m=DX_M, origin_m=(-2.25, -2.25, 0.0))
LIQUID_BUDGET_MS = 9.0  # SPEC 3, the liquid's whole frame
MASS_KG = 998.21 * DX_M**3 / 8  # the transfers' cost does not depend on it


def test_transfers_at_one_million_particles(bench_session: BenchSession) -> None:
    points = fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), DX_M, 8, seed=1)
    assert len(points) == 1_048_576
    rng = np.random.default_rng(0)
    points = points[rng.permutation(len(points))]
    velocities = rng.normal(0.0, 0.5, points.shape)
    affine = rng.normal(0.0, 0.1, (len(points), 3, 3))
    shuffled = LiquidParticles.from_numpy(
        points, velocities=velocities, affine=affine, device=DEVICE
    )
    ordered = LiquidParticles.from_numpy(points, device=DEVICE)
    sort = CellSort(SPEC, capacity=len(points), device=DEVICE)
    sort.sort(shuffled, ordered)
    scratch = LiquidParticles.from_numpy(points, device=DEVICE)
    velocity, mass = MacVelocity.zeros(SPEC, DEVICE), MacVelocity.zeros(SPEC, DEVICE)

    bench_session.start("transfers_1m")
    timed = time_interleaved(
        {
            "p2g_random": lambda: particles_to_grid(shuffled, MASS_KG, velocity, mass),
            "g2p_random": lambda: grid_to_particles(velocity, scratch),
            "p2g_sorted": lambda: particles_to_grid(ordered, MASS_KG, velocity, mass),
            "g2p_sorted": lambda: grid_to_particles(velocity, ordered),
            "sort": lambda: sort.sort(shuffled, scratch),
        },
        rounds=50,
        device=DEVICE,
    )
    summaries = {name: summarize(samples) for name, samples in timed.items()}
    ms = {name: s.median_s * 1e3 for name, s in summaries.items()}
    sorted_ms = ms["p2g_sorted"] + ms["g2p_sorted"]
    random_ms = ms["p2g_random"] + ms["g2p_random"]
    path = bench_session.record(
        "transfers_1m",
        {
            "particles": len(points),
            "grid": list(SPEC.shape),
            "summaries": {name: s.as_ms() for name, s in summaries.items()},
            "transfers_per_substep_random_ms": random_ms,
            "transfers_per_substep_sorted_ms": sorted_ms,
            "sort_per_frame_ms": ms["sort"],
            "liquid_budget_ms": LIQUID_BUDGET_MS,
        },
    )
    print(
        f"\ntransfers at 1 M particles, per substep: random order {random_ms:.2f} ms "
        f"(P2G {ms['p2g_random']:.2f}, G2P {ms['g2p_random']:.2f}); sorted {sorted_ms:.2f} ms "
        f"(P2G {ms['p2g_sorted']:.2f}, G2P {ms['g2p_sorted']:.2f}); sort {ms['sort']:.2f} ms "
        f"per frame -> {path}"
    )
    assert sorted_ms + ms["sort"] < random_ms  # sorting pays for itself at one substep

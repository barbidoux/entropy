"""M1 measurement: the APIC transfers at 1 million particles, by particle order (issue
#65, ADR 0025, SPEC 3's liquid row).

The Test Tank's grid (72 x 72 x 40 cells of 6.25 cm) holds 1,048,576 particles 2 m
deep at 8 per cell, as in the liquid pass's benchmark. Particles get random velocities
and affine matrices, so every weight is used. Timed interleaved, round by round
(ADR 0022):
- particles to grid and grid to particles in random order, in the order the bucket and
  `fill_box` pour (a lattice), and sorted by half-cell as the liquid keeps them;
- the sort itself, of shuffled particles and of particles already sorted (the liquid's
  case each frame), and the copy of the three arrays the sort replaces in the liquid.
"""

import numpy as np
import warp as wp

from bench.harness import BenchSession, summarize, time_interleaved
from entropy.fields.mac import GridSpec, MacVelocity
from entropy.solvers.liquid.particles import LiquidParticles
from entropy.solvers.liquid.solver import fill_box
from entropy.solvers.liquid.sorting import CellSort
from entropy.solvers.liquid.transfer import grid_to_particles, particles_to_grid
from entropy.world.materials import load_material

DEVICE = "cuda:0"
DX_M = 0.0625
SPEC = GridSpec(shape=(72, 72, 40), dx_m=DX_M, origin_m=(-2.25, -2.25, 0.0))
LIQUID_BUDGET_MS = 9.0  # SPEC 3, the liquid's whole frame
MASS_KG = load_material("water").density_kg_per_m3 * DX_M**3 / 8


def test_transfers_at_one_million_particles(bench_session: BenchSession) -> None:
    poured_points = fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), DX_M, 8, seed=1)
    assert len(poured_points) == 1_048_576
    rng = np.random.default_rng(0)
    velocities = rng.normal(0.0, 0.5, poured_points.shape)
    affine = rng.normal(0.0, 0.1, (len(poured_points), 3, 3))
    poured = LiquidParticles.from_numpy(
        poured_points, velocities=velocities, affine=affine, device=DEVICE
    )
    shuffle = rng.permutation(len(poured_points))
    points = poured_points[shuffle]
    shuffled = LiquidParticles.from_numpy(
        points, velocities=velocities[shuffle], affine=affine[shuffle], device=DEVICE
    )
    ordered = LiquidParticles.from_numpy(points, device=DEVICE)
    sort = CellSort(SPEC, capacity=len(points), device=DEVICE)
    sort.sort(shuffled, ordered)
    # Grid-to-particles writes its particles' velocities: in random order, into a copy,
    # so `shuffled` keeps feeding particles-to-grid the same state; the sort's own output
    # goes to a buffer no timed transfer reads.
    random_g2p = LiquidParticles.from_numpy(points, device=DEVICE)
    poured_g2p = LiquidParticles.from_numpy(poured_points, device=DEVICE)
    sort_out = LiquidParticles.from_numpy(points, device=DEVICE)

    def copy_three() -> None:  # what the liquid did before it sorted (`_keep_before`)
        for source, target in (
            (ordered.position_m, sort_out.position_m),
            (ordered.velocity_m_per_s, sort_out.velocity_m_per_s),
            (ordered.affine_per_s, sort_out.affine_per_s),
        ):
            wp.copy(target, source)

    velocity, mass = MacVelocity.zeros(SPEC, DEVICE), MacVelocity.zeros(SPEC, DEVICE)

    bench_session.start("transfers_1m")
    timed = time_interleaved(
        {
            "p2g_random": lambda: particles_to_grid(shuffled, MASS_KG, velocity, mass),
            "g2p_random": lambda: grid_to_particles(velocity, random_g2p),
            "p2g_poured": lambda: particles_to_grid(poured, MASS_KG, velocity, mass),
            "g2p_poured": lambda: grid_to_particles(velocity, poured_g2p),
            "p2g_sorted": lambda: particles_to_grid(ordered, MASS_KG, velocity, mass),
            "g2p_sorted": lambda: grid_to_particles(velocity, ordered),
            "sort_shuffled": lambda: sort.sort(shuffled, sort_out),
            "sort_sorted": lambda: sort.sort(ordered, sort_out),
            "copy_replaced": copy_three,
        },
        rounds=50,
        device=DEVICE,
    )
    summaries = {name: summarize(samples) for name, samples in timed.items()}
    ms = {name: s.median_s * 1e3 for name, s in summaries.items()}
    sorted_ms = ms["p2g_sorted"] + ms["g2p_sorted"]
    random_ms = ms["p2g_random"] + ms["g2p_random"]
    poured_ms = ms["p2g_poured"] + ms["g2p_poured"]
    net_sort_ms = ms["sort_sorted"] - ms["copy_replaced"]
    path = bench_session.record(
        "transfers_1m",
        {
            "particles": len(points),
            "grid": list(SPEC.shape),
            "summaries": {name: s.as_ms() for name, s in summaries.items()},
            "transfers_per_substep_random_ms": random_ms,
            "transfers_per_substep_poured_ms": poured_ms,
            "transfers_per_substep_sorted_ms": sorted_ms,
            "sort_shuffled_ms": ms["sort_shuffled"],
            "sort_net_per_frame_ms": net_sort_ms,
            "liquid_budget_ms": LIQUID_BUDGET_MS,
        },
    )
    print(
        f"\ntransfers at 1 M particles, per substep: random order {random_ms:.2f} ms "
        f"(P2G {ms['p2g_random']:.2f}, G2P {ms['g2p_random']:.2f}); as poured {poured_ms:.2f} "
        f"ms (P2G {ms['p2g_poured']:.2f}); sorted by half-cell {sorted_ms:.2f} ms (P2G "
        f"{ms['p2g_sorted']:.2f}, G2P {ms['g2p_sorted']:.2f}); sort {ms['sort_shuffled']:.2f} "
        f"ms shuffled, {ms['sort_sorted']:.2f} ms sorted, net {net_sort_ms:.2f} ms a frame "
        f"-> {path}"
    )
    # The liquid's order costs no more than the order water is poured in, and the sort
    # pays for itself at one substep even against that.
    assert ms["p2g_sorted"] <= 1.1 * ms["p2g_poured"]
    assert sorted_ms + net_sort_ms < random_ms

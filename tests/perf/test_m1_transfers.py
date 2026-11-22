"""M1 measurement: the APIC transfers at 1 million particles, by particle order (issue
#65, ADR 0025, SPEC 3's liquid row).

The Test Tank's grid (72 x 72 x 40 cells of 6.25 cm) holds 1,048,576 particles 2 m
deep at 8 per cell, as in the liquid pass's benchmark. Particles get random velocities
and affine matrices, so every weight is used. Timed interleaved, round by round
(ADR 0022):
- particles to grid and grid to particles in random order, in the bucket's and the
  dam's order (layers from the bottom, then x, y fastest), in `fill_box`'s lattice
  (already in half-cell order: a reference), and sorted by half-cell as the liquid
  keeps them;
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
    lattice_points = fill_box((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), DX_M, 8, seed=1)
    assert len(lattice_points) == 1_048_576
    rng = np.random.default_rng(0)
    velocities = rng.normal(0.0, 0.5, lattice_points.shape)
    affine = rng.normal(0.0, 0.1, (len(lattice_points), 3, 3))
    lattice = LiquidParticles.from_numpy(
        lattice_points, velocities=velocities, affine=affine, device=DEVICE
    )
    # The bucket's order (`block_of`, `dam_of`): layers from the bottom, then x, y fastest.
    sites = np.floor((lattice_points - (-2.0, -2.0, 0.0)) / (DX_M / 2)).astype(np.int64)
    layered = np.lexsort((sites[:, 1], sites[:, 0], sites[:, 2]))
    bucket_points = lattice_points[layered]
    bucket = LiquidParticles.from_numpy(
        bucket_points, velocities=velocities[layered], affine=affine[layered], device=DEVICE
    )
    shuffle = rng.permutation(len(lattice_points))
    points = lattice_points[shuffle]
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
    lattice_g2p = LiquidParticles.from_numpy(lattice_points, device=DEVICE)
    bucket_g2p = LiquidParticles.from_numpy(bucket_points, device=DEVICE)
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
            "p2g_bucket": lambda: particles_to_grid(bucket, MASS_KG, velocity, mass),
            "g2p_bucket": lambda: grid_to_particles(velocity, bucket_g2p),
            "p2g_lattice": lambda: particles_to_grid(lattice, MASS_KG, velocity, mass),
            "g2p_lattice": lambda: grid_to_particles(velocity, lattice_g2p),
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
    bucket_ms = ms["p2g_bucket"] + ms["g2p_bucket"]
    lattice_ms = ms["p2g_lattice"] + ms["g2p_lattice"]
    net_sort_ms = ms["sort_sorted"] - ms["copy_replaced"]
    path = bench_session.record(
        "transfers_1m",
        {
            "particles": len(points),
            "grid": list(SPEC.shape),
            "summaries": {name: s.as_ms() for name, s in summaries.items()},
            "transfers_per_substep_random_ms": random_ms,
            "transfers_per_substep_bucket_ms": bucket_ms,
            "transfers_per_substep_lattice_ms": lattice_ms,
            "transfers_per_substep_sorted_ms": sorted_ms,
            "sort_shuffled_ms": ms["sort_shuffled"],
            "sort_net_per_frame_ms": net_sort_ms,
            "liquid_budget_ms": LIQUID_BUDGET_MS,
        },
    )
    print(
        f"\ntransfers at 1 M particles, per substep: random order {random_ms:.2f} ms "
        f"(P2G {ms['p2g_random']:.2f}, G2P {ms['g2p_random']:.2f}); bucket's order "
        f"{bucket_ms:.2f} ms (P2G {ms['p2g_bucket']:.2f}); lattice {lattice_ms:.2f} ms (P2G "
        f"{ms['p2g_lattice']:.2f}); sorted by half-cell {sorted_ms:.2f} ms (P2G "
        f"{ms['p2g_sorted']:.2f}, G2P {ms['g2p_sorted']:.2f}); sort {ms['sort_shuffled']:.2f} "
        f"ms shuffled, {ms['sort_sorted']:.2f} ms sorted, net {net_sort_ms:.2f} ms a frame "
        f"-> {path}"
    )
    # The half-cell key keeps the lattice's cost (a whole-cell key does not), and the
    # sort pays for itself at one substep against the orders the tools pour in.
    assert ms["p2g_sorted"] <= 1.1 * ms["p2g_lattice"]
    assert sorted_ms + net_sort_ms < min(bucket_ms, random_ms)

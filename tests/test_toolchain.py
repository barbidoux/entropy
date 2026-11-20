"""Walking-skeleton tests: the package imports and the Warp toolchain runs.

These are the first tests of the repository. They prove, on any machine,
that the package is installable and that a Warp kernel compiles and runs on
the CPU, including the float atomic additions every particle-to-grid
transfer relies on (SPEC section 5.2).
"""

import numpy as np
import pytest
import warp as wp

import entropy


def test_package_exposes_a_semantic_version() -> None:
    parts = entropy.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


@wp.kernel
def _scatter_add(positions: wp.array(dtype=float), grid: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(grid, int(positions[i]), 1.0)


def test_warp_float_atomics_work_on_cpu() -> None:
    positions = wp.array(np.array([0.5, 1.2, 1.7, 2.1], dtype=np.float32), device="cpu")
    grid = wp.zeros(4, dtype=float, device="cpu")

    wp.launch(_scatter_add, dim=positions.shape[0], inputs=[positions, grid], device="cpu")

    np.testing.assert_array_equal(grid.numpy(), [1.0, 2.0, 1.0, 0.0])


@wp.kernel
def _sum(values: wp.array(dtype=float), total: wp.array(dtype=float)):
    wp.atomic_add(total, 0, values[wp.tid()])


def test_cpu_atomic_sums_are_bit_reproducible() -> None:
    """SPEC section 9.6: CPU runs are bit-for-bit reproducible."""
    rng = np.random.default_rng(seed=0)
    values = wp.array(rng.standard_normal(100_000).astype(np.float32) * 1e3, device="cpu")

    results = set()
    for _ in range(3):
        total = wp.zeros(1, dtype=float, device="cpu")
        wp.launch(_sum, dim=values.shape[0], inputs=[values, total], device="cpu")
        results.add(total.numpy()[0].tobytes())

    assert len(results) == 1


@pytest.mark.gpu
def test_cuda_device_is_available() -> None:
    """Runs on the laptop only (`uv run pytest -m gpu`)."""
    assert wp.is_cuda_available(), "No CUDA device: update the NVIDIA driver, then reinstall Warp."

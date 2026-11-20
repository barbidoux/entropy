"""Frame statistics for the performance HUD (`entropy.core.profiler`)."""

import pytest
import warp as wp

from entropy.core.profiler import FrameStats, gpu_memory


def test_steady_sixty_hertz() -> None:
    stats = FrameStats(window=10)
    for _ in range(10):
        stats.record(1 / 60, {"simulation": 0.004, "rendering": 0.006})

    summary = stats.summary()

    assert summary.fps == pytest.approx(60.0)
    assert summary.frame_ms_mean == pytest.approx(1000 / 60)
    assert summary.stages_ms_mean == pytest.approx({"simulation": 4.0, "rendering": 6.0})


def test_worst_frame_is_reported() -> None:
    stats = FrameStats(window=10)
    for frame_s in (0.016, 0.050, 0.016):
        stats.record(frame_s, {})

    assert stats.summary().frame_ms_worst == pytest.approx(50.0)


def test_old_frames_leave_the_window() -> None:
    stats = FrameStats(window=3)
    stats.record(0.100, {"simulation": 0.090})
    for _ in range(3):
        stats.record(0.010, {"simulation": 0.001})

    summary = stats.summary()

    assert summary.frame_ms_worst == pytest.approx(10.0)
    assert summary.stages_ms_mean["simulation"] == pytest.approx(1.0)


def test_a_stage_missing_from_some_frames_averages_over_its_own_frames() -> None:
    stats = FrameStats(window=4)
    stats.record(0.010, {"simulation": 0.002})
    stats.record(0.010, {})

    assert stats.summary().stages_ms_mean["simulation"] == pytest.approx(2.0)


def test_no_frames_yet() -> None:
    summary = FrameStats().summary()

    assert summary.fps == 0.0
    assert summary.frames == 0


def test_window_must_hold_a_frame() -> None:
    with pytest.raises(ValueError, match="window"):
        FrameStats(window=0)


def test_no_gpu_memory_on_cpu() -> None:
    assert gpu_memory("cpu") is None


@pytest.mark.gpu
def test_gpu_memory_counts_warp_allocations() -> None:
    before = gpu_memory("cuda:0")
    block = wp.zeros(64 * 2**20, dtype=wp.uint8, device="cuda:0")
    after = gpu_memory("cuda:0")

    assert before is not None and after is not None
    assert after.warp_bytes - before.warp_bytes >= 64 * 2**20
    assert 0 < after.device_used_bytes < after.device_total_bytes
    assert after.device_total_bytes > 15 * 2**30
    del block

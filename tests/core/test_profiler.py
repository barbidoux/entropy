"""Frame statistics for the performance HUD (`entropy.core.profiler`)."""

import sys

import pytest
import warp as wp

from entropy.core.profiler import (
    FrameStats,
    StageTimer,
    cuda_stage_timer,
    gpu_memory,
    host_memory_bytes,
)


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

    assert summary.frames == 3
    assert summary.frame_ms_worst == pytest.approx(10.0)
    assert summary.stages_ms_mean["simulation"] == pytest.approx(1.0)


def test_world_speed_is_simulated_over_real_time() -> None:
    stats = FrameStats(window=10)
    stats.record(0.02, {}, advance_s=0.01)
    stats.record(0.02, {}, advance_s=0.01)

    assert stats.summary().world_speed == pytest.approx(0.5)


def test_time_kept_compares_simulated_with_requested_time() -> None:
    """Frames that hit the 1/30 s clamp keep less time than they asked for."""
    stats = FrameStats(window=10)
    stats.record(0.1, {}, advance_s=1 / 30, requested_s=0.1)
    stats.record(0.1, {}, advance_s=1 / 30, requested_s=0.1)

    assert stats.summary().time_kept == pytest.approx(1 / 3)


def test_paused_frames_do_not_count_against_time_kept() -> None:
    stats = FrameStats(window=10)
    stats.record(1 / 60, {}, advance_s=0.0, requested_s=0.0)
    stats.record(1 / 60, {}, advance_s=1 / 60, requested_s=1 / 60)

    assert stats.summary().time_kept == pytest.approx(1.0)


def test_world_speed_defaults_to_zero_without_advances() -> None:
    stats = FrameStats(window=10)
    stats.record(0.02, {})

    assert stats.summary().world_speed == 0.0


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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process memory API")
def test_host_memory_sees_a_new_allocation() -> None:
    before = host_memory_bytes()
    block = bytearray(64 * 2**20)  # committed and touched
    after = host_memory_bytes()

    assert before is not None and after is not None
    assert after - before >= 60 * 2**20
    del block


@pytest.mark.skipif(sys.platform == "win32", reason="other systems")
def test_host_memory_is_none_elsewhere() -> None:
    assert host_memory_bytes() is None


@pytest.mark.gpu
def test_gpu_memory_counts_warp_allocations() -> None:
    before = gpu_memory("cuda:0")
    block = wp.zeros(64 * 2**20, dtype=wp.uint8, device="cuda:0")
    after = gpu_memory("cuda:0")

    assert before is not None and after is not None
    assert before.warp_bytes is not None and after.warp_bytes is not None
    assert after.warp_bytes - before.warp_bytes >= 64 * 2**20
    assert after.warp_high_bytes is not None and after.warp_high_bytes >= after.warp_bytes
    assert 0 < after.device_used_bytes < after.device_total_bytes
    assert after.device_total_bytes == wp.get_device("cuda:0").total_memory
    del block


class FakeTimings:
    """A GPU timing backend whose spans finish when the test says so."""

    def __init__(self, duration_s: float, *, lag_frames: int = 0) -> None:
        self.duration_s = duration_s
        self.lag_frames = lag_frames
        self.finished = False
        self.begun = 0
        self.ended = 0

    def begin(self) -> int:
        self.begun += 1
        return self.begun

    def end(self, token: int) -> None:
        self.ended += 1

    def ready(self, token: int, frames_ago: int) -> bool:
        return self.finished and frames_ago >= self.lag_frames

    def take_elapsed_s(self, token: int) -> float:
        return self.duration_s


def test_gpu_times_come_once_the_gpu_has_finished_the_frame() -> None:
    timings = FakeTimings(0.002)
    timer = StageTimer(timings)
    with timer.stage("liquid"):
        pass
    timer.end_frame()

    assert timer.collect() == []
    timings.finished = True
    assert timer.collect() == [{"liquid": pytest.approx(0.002)}]
    assert timer.collect() == []


def test_stages_of_the_same_name_add_up_within_a_frame() -> None:
    """Substeps of one solver count as one stage."""
    timings = FakeTimings(0.001)
    timings.finished = True
    timer = StageTimer(timings)
    for _ in range(3):
        with timer.stage("liquid"):
            pass
    timer.end_frame()

    assert timer.collect() == [{"liquid": pytest.approx(0.003)}]


def test_finished_frames_come_one_by_one_oldest_first() -> None:
    timings = FakeTimings(0.001)
    timer = StageTimer(timings)
    with timer.stage("liquid"):
        pass
    timer.end_frame()
    with timer.stage("gas"):
        pass
    timer.end_frame()
    timings.finished = True

    assert timer.collect() == [{"liquid": pytest.approx(0.001)}, {"gas": pytest.approx(0.001)}]


def test_results_can_be_read_a_number_of_frames_later() -> None:
    """OpenGL queries cannot say when they are ready, so they are read late."""
    timings = FakeTimings(0.001, lag_frames=2)
    timings.finished = True
    timer = StageTimer(timings)
    with timer.stage("scene"):
        pass
    timer.end_frame()
    assert timer.collect() == []
    timer.end_frame()
    assert timer.collect() == []
    timer.end_frame()

    assert timer.collect() == [{"scene": pytest.approx(0.001)}]


def test_a_stage_left_by_an_exception_is_still_closed() -> None:
    timings = FakeTimings(0.001)
    timer = StageTimer(timings)

    with pytest.raises(RuntimeError), timer.stage("liquid"):
        raise RuntimeError("solver failed")

    assert timings.ended == 1


def test_without_a_gpu_nothing_is_timed() -> None:
    timer = StageTimer(None)
    with timer.stage("liquid"):
        pass
    timer.end_frame()

    assert timer.collect() == []


def test_cuda_stage_timer_on_cpu_times_nothing() -> None:
    timer = cuda_stage_timer("cpu")
    with timer.stage("liquid"):
        pass
    timer.end_frame()

    assert timer.collect() == []


def test_gpu_times_are_averaged_per_stage() -> None:
    stats = FrameStats(window=10)
    stats.record(1 / 60, {})
    stats.record_gpu({"liquid": 0.002, "scene": 0.001})
    stats.record_gpu({"liquid": 0.004})

    assert stats.summary().gpu_ms_mean == pytest.approx({"liquid": 3.0, "scene": 1.0})


def test_gpu_times_leave_the_window_like_frames() -> None:
    stats = FrameStats(window=2)
    stats.record(1 / 60, {})
    for liquid_s in (0.010, 0.001, 0.001):
        stats.record_gpu({"liquid": liquid_s})

    assert stats.summary().gpu_ms_mean == pytest.approx({"liquid": 1.0})


@pytest.mark.gpu
def test_cuda_events_time_a_kernel() -> None:
    timer = cuda_stage_timer("cuda:0")
    block = wp.zeros(64 * 2**20, dtype=float, device="cuda:0")
    with timer.stage("fill"):
        block.fill_(1.0)
    timer.end_frame()
    wp.synchronize_device("cuda:0")

    (frame,) = timer.collect()

    # 256 MiB written at no more than the 576 GB/s DRAM peak take at least 0.46 ms.
    assert 0.00046 < frame["fill"] < 0.1

"""Text of the performance HUD (`entropy.ui.hud`), SPEC 4.7."""

import dataclasses

from entropy.core.profiler import FrameSummary, GpuMemory
from entropy.ui.hud import SLOWDOWN_PREFIX, hud_lines

SUMMARY = FrameSummary(
    frames=60,
    fps=59.94,
    frame_ms_mean=16.68,
    frame_ms_worst=21.2,
    stages_ms_mean={"logic": 0.42, "simulation": 0.0, "rendering": 1.31, "present": 14.9},
    world_speed=1.0,
    time_kept=1.0,
)
MEMORY = GpuMemory(
    warp_bytes=512 * 2**20, device_used_bytes=3 * 2**30, device_total_bytes=16 * 2**30
)


def test_hud_shows_rate_frame_times_and_every_stage() -> None:
    text = "\n".join(hud_lines(SUMMARY, MEMORY, interop="zero-copy"))

    assert "59.9 fps" in text
    assert "16.68 ms" in text
    assert "worst 21.20 ms" in text
    for stage, ms in (("logic", "0.42"), ("simulation", "0.00"), ("rendering", "1.31")):
        assert stage in text
        assert f"{ms} ms" in text
    assert "slowed" not in text
    assert "time scale" not in text


def test_hud_labels_cpu_and_gpu_stage_times() -> None:
    """SPEC 4.7: GPU time per stage, next to the CPU stage times, labelled as such."""
    summary = dataclasses.replace(
        SUMMARY, gpu_ms_mean={"test pattern (CUDA)": 0.05, "scene (OpenGL)": 0.35}
    )

    lines = hud_lines(summary, MEMORY, interop="zero-copy")

    cpu, gpu = lines.index("CPU ms per stage"), lines.index("GPU ms per stage")
    assert cpu < lines.index(next(line for line in lines if "rendering" in line)) < gpu
    assert any("scene (OpenGL)" in line and "0.35 ms" in line for line in lines[gpu:])
    assert any("test pattern (CUDA)" in line and "0.05 ms" in line for line in lines[gpu:])


def test_cpu_and_gpu_times_line_up() -> None:
    summary = dataclasses.replace(SUMMARY, gpu_ms_mean={"scene (OpenGL)": 0.35})

    lines = hud_lines(summary, MEMORY, interop="zero-copy")

    cpu = next(line for line in lines if "logic" in line)
    gpu = next(line for line in lines if "scene (OpenGL)" in line)
    assert len(cpu) == len(gpu)


def test_hud_says_when_no_gpu_time_has_come_yet() -> None:
    assert "GPU ms per stage: none yet" in hud_lines(SUMMARY, MEMORY, interop="zero-copy")


def test_hud_shows_vram_and_the_interop_path() -> None:
    text = "\n".join(hud_lines(SUMMARY, MEMORY, interop="host-copy"))

    assert "VRAM 3.00 / 16.00 GiB" in text
    assert "ENTROPY 0.50 GiB" in text
    assert "interop host-copy" in text


def test_hud_without_a_gpu_says_so() -> None:
    assert "VRAM n/a" in "\n".join(hud_lines(SUMMARY, None, interop="host-copy"))


def test_hud_without_warp_pool_figures() -> None:
    memory = dataclasses.replace(MEMORY, warp_bytes=None)

    assert "ENTROPY n/a" in "\n".join(hud_lines(SUMMARY, memory, interop="zero-copy"))


def test_hud_says_when_the_world_runs_slower_than_asked() -> None:
    """SPEC 5.9: below 30 fps the clamp slows the world down, and the HUD says so."""
    slowed = dataclasses.replace(SUMMARY, time_kept=0.62)

    lines = hud_lines(slowed, MEMORY, interop="zero-copy")

    assert f"{SLOWDOWN_PREFIX}: 62 % of the set speed" in lines


def test_a_small_shortfall_is_not_reported() -> None:
    almost = dataclasses.replace(SUMMARY, time_kept=0.97)

    assert "slowed" not in "\n".join(hud_lines(almost, MEMORY, interop="zero-copy"))


def test_slow_motion_and_pause_are_shown_as_such() -> None:
    slow = "\n".join(hud_lines(SUMMARY, MEMORY, interop="zero-copy", time_scale=0.1))
    paused = "\n".join(hud_lines(SUMMARY, MEMORY, interop="zero-copy", time_scale=0.0))

    assert "time scale 0.10x" in slow
    assert "paused" in paused
    assert "slowed" not in slow + paused


def test_hud_shows_extra_notes() -> None:
    lines = hud_lines(SUMMARY, None, interop="zero-copy", notes=["200,000 points"])

    assert lines[-1] == "200,000 points"

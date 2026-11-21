"""Checkpoints (`entropy.sim.checkpoints`, SPEC 4.6, #55): a ring of solver states."""

import pytest

from entropy.sim.checkpoints import CheckpointRing


class _Solver:
    def __init__(self) -> None:
        self.state = 0
        self.restored: list[int] = []

    def checkpoint(self) -> int:
        return self.state

    def restore(self, state: int) -> None:
        self.restored.append(state)
        self.state = state


def test_a_checkpoint_is_taken_every_ten_simulated_seconds() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)

    for second in range(1, 36):
        solver.state = second
        ring.advance(1.0)

    assert ring.saved_times_s() == [10.0, 20.0, 30.0]
    assert [states[0] for _, states in ring.entries] == [10, 20, 30]


def test_the_ring_keeps_the_last_six() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=1.0, keep=6)

    for second in range(1, 10):
        solver.state = second
        ring.advance(1.0)

    assert ring.saved_times_s() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_save_now_and_restore_the_latest() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)
    solver.state = 7
    ring.advance(3.0)
    ring.save_now()
    solver.state = 99

    restored_s = ring.restore_latest()

    assert restored_s == pytest.approx(3.0)
    assert solver.state == 7
    assert ring.clock_s == pytest.approx(3.0)


def test_restoring_with_no_checkpoint_does_nothing() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)

    assert ring.restore_latest() is None
    assert solver.restored == []


def test_freezing_drops_what_was_saved_since_the_last_step() -> None:
    """SPEC 5.9, 4.6: the NaN guard reads a step's state only at the next frame, so a
    checkpoint saved after the step that went non-finite holds NaN. When the world
    freezes, those are dropped; the ones before, checked by a later step, are kept."""
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)
    solver.state = 1
    ring.advance(10.0)  # saved at 10 s; the next step found it finite
    ring.advance(5.0)  # this step's state is the one that went non-finite
    solver.state = 2
    ring.save_now()  # saved at 15 s, before the freeze was seen

    ring.drop_unchecked()

    assert ring.saved_times_s() == [10.0]
    assert ring.restore_latest() == 10.0 and solver.state == 1

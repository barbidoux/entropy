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

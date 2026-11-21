"""Checkpoints (`entropy.sim.checkpoints`, SPEC 4.6, #55): a ring of solver states."""

from pathlib import Path

import pytest

from entropy.sim.checkpoints import CheckpointRing, Rewind
from entropy.sim.simulation import Simulation
from entropy.solvers.interface import NonFiniteStateError


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


def test_the_ring_s_defaults_are_the_spec_s() -> None:
    """SPEC 4.6: a checkpoint every 10 s of simulated time, a ring of 6."""
    ring = CheckpointRing([_Solver()])

    assert ring.every_s == 10.0
    assert ring.entries.maxlen == 6


def test_restoring_moves_the_clock_and_the_next_save_back() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)
    for second in range(1, 26):
        solver.state = second
        ring.advance(1.0)  # saves at 10 s and 20 s

    assert ring.restore_latest() == 20.0
    assert ring.clock_s == 20.0
    ring.advance(9.0)
    assert ring.saved_times_s() == [10.0, 20.0]  # the next save is 10 s after 20 s
    ring.advance(1.0)
    assert ring.saved_times_s() == [10.0, 20.0, 30.0]


def test_restoring_again_goes_one_checkpoint_further_back() -> None:
    """F9 twice in a row: the checkpoint before the latest, and so on to the oldest."""
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)
    for second in range(1, 31):
        solver.state = second
        ring.advance(1.0)  # saves at 10, 20 and 30 s

    assert [ring.restore_latest() for _ in range(4)] == [30.0, 20.0, 10.0, 10.0]
    assert solver.state == 10


def test_running_on_from_an_earlier_checkpoint_forgets_the_later_ones() -> None:
    solver = _Solver()
    ring = CheckpointRing([solver], every_s=10.0, keep=6)
    for second in range(1, 31):
        solver.state = second
        ring.advance(1.0)
    ring.restore_latest()
    ring.restore_latest()  # back at 20 s

    ring.advance(1.0)

    assert ring.saved_times_s() == [10.0, 20.0]
    ring.advance(9.0)  # the next save is 10 s after the restored 20 s, not after 30 s
    assert ring.saved_times_s() == [10.0, 20.0, 30.0]


class _AsyncSolver(_Solver):
    """Offers `checkpoint_async`, reusing what it is handed (pinned buffers, in the liquid)."""

    def __init__(self) -> None:
        super().__init__()
        self.reused: list[object] = []

    def checkpoint_async(self, reuse: object = None) -> object:
        self.reused.append(reuse)
        return ("state", self.state)


def test_a_full_ring_hands_the_checkpoint_it_drops_over_for_reuse() -> None:
    """Allocating pinned memory for every save stalls a frame (#74's review, S5): the
    checkpoint that falls off the ring lends its buffers to the new one."""
    solver = _AsyncSolver()
    ring = CheckpointRing([solver], every_s=1.0, keep=2)

    for _ in range(4):
        ring.advance(1.0)

    assert solver.reused == [None, None, ("state", 0), ("state", 0)]


class _Liquid(_Solver):
    """A solver whose state goes non-finite when told, found at its next pacing read, as
    the liquid's NaN guard finds it (one frame late)."""

    def __init__(self) -> None:
        super().__init__()
        self.poisoned = False
        self.found = False

    def max_advance_s(self) -> float:
        if self.poisoned:
            self.found = True
            raise NonFiniteStateError(Path("out/nan_dumps/fake.npz"))
        return 1.0

    def step(self, dt_s: float, ctx: object) -> None:
        self.state += 1

    def restore(self, state: int) -> None:
        super().restore(state)
        self.poisoned = False


def _rewind() -> tuple[Rewind, _Liquid]:
    liquid = _Liquid()
    simulation = Simulation([liquid])
    return Rewind(simulation, CheckpointRing([liquid], every_s=10.0, keep=6)), liquid


def test_there_is_always_a_checkpoint_at_the_start() -> None:
    """A freeze in the first 10 s, with no F5, still has somewhere to go back to."""
    rewind, _ = _rewind()

    assert rewind.ring.saved_times_s() == [0.0]


def test_saving_is_refused_while_the_world_is_frozen() -> None:
    rewind, liquid = _rewind()
    rewind.step(0.5)
    liquid.poisoned = True
    rewind.step(0.5)  # finds it and freezes

    assert not rewind.save()
    assert rewind.ring.saved_times_s() == [0.0]


def test_a_freeze_drops_what_was_saved_after_the_last_good_step() -> None:
    rewind, liquid = _rewind()
    rewind.step(0.5)
    assert rewind.save()  # at 0.5 s; the next step checks it
    rewind.step(0.5)
    liquid.poisoned = True  # the step to 1.0 s went non-finite
    assert rewind.save()  # at 1.0 s, before the freeze is seen
    rewind.step(0.5)

    assert rewind.simulation.frozen_by is not None
    assert rewind.ring.saved_times_s() == [0.0, 0.5]


def test_restoring_checks_the_last_step_first() -> None:
    """#74's review, S2: paused after a step that went non-finite, F5 then F9. The save
    is unchecked; restoring it would freeze again with nowhere left to go. The restore
    reads the last step's verdict first, drops it, and goes back to the good one."""
    rewind, liquid = _rewind()
    rewind.step(0.5)
    rewind.save()  # 0.5 s, then checked
    rewind.step(0.5)
    liquid.state = 99
    liquid.poisoned = True  # the step to 1.0 s went non-finite; nothing read it yet
    rewind.save()  # paused: F5 at 1.0 s

    assert rewind.restore() == 0.5

    assert liquid.found and liquid.state == 1
    assert rewind.simulation.frozen_by is None
    assert rewind.ring.saved_times_s() == [0.0, 0.5]

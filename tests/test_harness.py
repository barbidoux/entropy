"""The test harness itself: device parametrization (SPEC 9.1 rule 3)."""

from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")

_INI = """
[pytest]
markers =
    gpu: needs a CUDA device
"""

_TEST_USING_DEVICE = """
def test_runs_on(device):
    assert device in ("cpu", "cuda:0")
"""


@pytest.fixture
def harness(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(CONFTEST.read_text(encoding="utf-8"))
    pytester.makeini(_INI)
    pytester.makepyfile(_TEST_USING_DEVICE)
    return pytester


def test_device_fixture_runs_on_cpu_in_the_default_suite(harness: pytest.Pytester) -> None:
    result = harness.runpytest("-m", "not gpu")

    result.assert_outcomes(passed=1, deselected=1)
    result.stdout.no_fnmatch_line("*test_runs_on?cuda*PASSED*")


def test_device_fixture_adds_a_cuda_case_marked_gpu(harness: pytest.Pytester) -> None:
    result = harness.runpytest("-m", "gpu", "--collect-only", "-q")

    result.stdout.fnmatch_lines(["*test_runs_on?cuda?*"])
    result.stdout.no_fnmatch_line("*test_runs_on?cpu?*")

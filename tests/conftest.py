"""Shared fixtures for the whole suite."""

import pytest

# SPEC 9.1 rule 3: solver tests run on CPU everywhere, and on CUDA with `-m gpu`.
DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu),
]


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    """Warp device name for a device-parametrized test."""
    return request.param

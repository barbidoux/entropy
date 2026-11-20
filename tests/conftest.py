"""Shared fixtures for the whole suite."""

import os

import pytest

from entropy.core.config import WindowSettings

# SPEC 9.1 rule 3: solver tests run on CPU everywhere, and on CUDA with `-m gpu`.
DEVICES = (
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda:0", id="cuda", marks=pytest.mark.gpu),
)


@pytest.fixture(params=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    """Warp device name for a device-parametrized test."""
    return request.param


@pytest.fixture
def window_settings() -> WindowSettings:
    """A small hidden-window setup for `render` tests.

    It asks for the game's OpenGL 4.6, or for `ENTROPY_TEST_GL_VERSION`: CI sets
    4.5, the highest Mesa's software renderer offers (ADR 0009).
    """
    requested = os.environ.get("ENTROPY_TEST_GL_VERSION", "4.6")
    try:
        major, minor = requested.split(".")
    except ValueError:
        raise pytest.UsageError(
            f"ENTROPY_TEST_GL_VERSION must look like 4.5, got {requested!r}"
        ) from None
    return WindowSettings(
        width_px=320,
        height_px=200,
        title="ENTROPY test",
        vsync=False,
        gl_version=(int(major), int(minor)),
    )

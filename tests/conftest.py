"""Shared fixtures for the lab test suite."""

import pytest


@pytest.fixture(scope="session")
def fast_compile():
    """Run PyTensor in FAST_COMPILE mode for tiny test models (no C compilation)."""
    import pytensor

    with pytensor.config.change_flags(mode="FAST_COMPILE"):
        yield

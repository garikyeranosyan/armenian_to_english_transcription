"""Fixtures shared by the end-to-end tests."""

import pytest

from live_hy_en.output import configure_terminal
from live_hy_en.pipeline import load_models
from replay import block_network


@pytest.fixture(scope="session", autouse=True)
def utf8_output():
    """Print the replays' text as UTF-8, as the program does, also when the
    output goes to a file (which would otherwise get the ANSI code page)."""
    configure_terminal()


@pytest.fixture(scope="session")
def offline_models():
    """Load every model once, with the network blocked; return them and any
    network attempts made while loading.

    One shared set: two sets at once don't fit in the laptop's GPU memory and
    slow every model down.
    """
    attempts: list = []
    with pytest.MonkeyPatch.context() as patch:
        block_network(patch, attempts)
        models = load_models()
    return models, attempts


@pytest.fixture(scope="session")
def models(offline_models):
    """The loaded models."""
    return offline_models[0]

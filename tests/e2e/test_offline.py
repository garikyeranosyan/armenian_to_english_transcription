"""Local models only (PLAN.md §5.1): every model loads and runs with the network
blocked, so nothing is downloaded or sent to a cloud service at runtime.
"""

import pytest

from asr_benchmark import load_fleurs
from replay import block_network, needs_laptop

pytestmark = [pytest.mark.e2e, needs_laptop]


def test_models_load_and_run_with_the_network_blocked(offline_models, monkeypatch):
    models, load_attempts = offline_models
    run_attempts: list = []
    [clip] = load_fleurs(1)
    block_network(monkeypatch, run_attempts)
    armenian = models.transcriber.transcribe(clip.samples)
    english = models.translator.translate(armenian)
    # Armenian isn't printed: a redirected Windows console can't encode it.
    print(f"\nOffline translation: {english}")

    assert armenian and english
    assert load_attempts == [], "loading the models tried to reach the network"
    assert run_attempts == [], "running the models tried to reach the network"

"""Soak test (PLAN.md §5.1, runs reliably): 20+ minutes of replayed conversation
with no crash, no memory growth and no growing delay. It takes as long as the
audio, so it only runs when asked for:
    .\\.venv\\Scripts\\python.exe -m pytest -m soak -s
Progress shows on the console every minute and is saved to SOAK_LOG; no
redirection is needed (PowerShell's Out-File would hide it until the end).
"""

import sys
import threading
from pathlib import Path

import pytest

from asr_benchmark import load_fleurs
from live_hy_en.metrics import MAX_LATENCY_S
from replay import (
    assert_latency_targets,
    build_tracks,
    entries_of,
    log_to,
    needs_laptop,
    p95_latency_s,
    place_alternating,
    process_memory_bytes,
    replay,
    report,
)

pytestmark = [pytest.mark.soak, needs_laptop]

SOAK_S = 20 * 60
# FLEURS clips average about 11.5 s, plus the gap: enough clips for SOAK_S.
SOAK_CLIP_COUNT = 110
MEMORY_SAMPLE_S = 10.0
# Memory may settle during the first quarter; after that it must stay flat.
MAX_MEMORY_GROWTH_BYTES = 200 * 1024 * 1024
# The last quarter's p95 delay may not exceed the first quarter's by more.
MAX_DELAY_GROWTH_S = 0.5
PROGRESS_EVERY_S = 60.0
SOAK_LOG = Path(__file__).resolve().parents[1] / "outputs" / "soak_output.txt"


def test_twenty_minute_call_runs_without_crash_leak_or_growing_delay(models):
    with log_to(SOAK_LOG):
        _soak(models)


def _soak(models) -> None:
    placed, audio = place_alternating(load_fleurs(SOAK_CLIP_COUNT))
    assert placed[-1].end_s >= SOAK_S, "not enough audio for the soak length"

    samples: list[int] = []
    finished = threading.Event()

    def sample_memory() -> None:
        while not finished.wait(MEMORY_SAMPLE_S):
            memory = process_memory_bytes()
            if memory is not None:
                samples.append(memory)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        print(
            f"\nReplaying {placed[-1].end_s / 60:.0f} min of audio in real time; "
            "results at the end.",
            flush=True,
        )
        shown, pipeline, _ = replay(
            models, build_tracks(placed, audio), progress_every_s=PROGRESS_EVERY_S
        )
    finally:
        finished.set()
        sampler.join()
    report(f"Soak, {placed[-1].end_s / 60:.0f} min", placed, shown, pipeline)

    lost = [clip for clip in placed if not entries_of(clip, shown)]
    assert lost == [], f"{len(lost)} phrases were lost"
    assert_latency_targets(pipeline, MAX_LATENCY_S)

    timings = pipeline.metrics.timings
    quarter = len(timings) // 4
    first_s, last_s = (
        p95_latency_s(timings[:quarter]),
        p95_latency_s(timings[-quarter:]),
    )
    print(f"p95 delay: first quarter {first_s:.2f} s, last quarter {last_s:.2f} s")
    assert last_s <= first_s + MAX_DELAY_GROWTH_S, "delay grows over the call"

    if sys.platform == "win32":
        assert samples, "memory was not measured"
    if samples:
        settled = samples[len(samples) // 4]
        growth = max(samples[len(samples) // 4 :]) - settled
        print(f"Memory after settling: grew {growth / 2**20:.0f} MB")
        assert growth <= MAX_MEMORY_GROWTH_BYTES, "memory grows over the call"

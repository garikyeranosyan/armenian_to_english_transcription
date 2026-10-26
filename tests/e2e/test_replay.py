"""End-to-end replay with the real models (PLAN.md §5.2 layer 3).

Recorded Armenian speech (FLEURS hy_am, downloaded by the ASR benchmark) is fed
through the full pipeline at real-time speed in place of live capture. Run on the
laptop with:
    .\\.venv\\Scripts\\python.exe -m pytest -m e2e -s
"""

import numpy as np
import pytest

from asr_benchmark import load_fleurs, normalize
from live_hy_en.capture import SAMPLE_RATE_HZ, Source
from live_hy_en.metrics import MAX_LATENCY_S
from live_hy_en.segmenter import MAX_SEGMENT_S
from live_hy_en.speakers import MIC_LABEL
from replay import (
    CLIP_MARGIN_S,
    GAP_S,
    PlacedClip,
    assert_latency_targets,
    build_tracks,
    entries_of,
    needs_laptop,
    place_alternating,
    print_latency,
    replay,
    report,
)

pytestmark = [pytest.mark.e2e, needs_laptop]

CLIP_COUNT = 12
OVERLAP_PAIR_COUNT = 4
MONOLOGUE_CLIP_COUNT = 6
# Breaths between sentences: shorter than the pause that ends a phrase.
MONOLOGUE_GAP_S = 0.3
# Share of the monologue's words that must come out as some word, not be lost.
MIN_WORDS_KEPT = 0.8


@pytest.fixture(scope="module")
def clips():
    return load_fleurs(CLIP_COUNT + 2 * OVERLAP_PAIR_COUNT + MONOLOGUE_CLIP_COUNT)


def test_alternating_call_is_transcribed_live_within_the_limit(models, clips):
    placed, audio = place_alternating(clips[:CLIP_COUNT])
    shown, pipeline, playback_end_s = replay(models, build_tracks(placed, audio))
    report("Alternating call", placed, shown, pipeline)

    assert all(entries_of(clip, shown) for clip in placed), "a phrase was lost"
    for item in shown:
        expected_me = any(
            clip.source is Source.MIC
            and clip.start_s - CLIP_MARGIN_S <= item.start_s <= clip.end_s
            for clip in placed
        )
        assert (item.entry.speaker == MIC_LABEL) == expected_me
        assert item.entry.english
    # Live: entries appear while the call goes on, not at the end.
    assert sum(item.shown_s < playback_end_s for item in shown) >= len(shown) - 2
    assert_latency_targets(pipeline, MAX_LATENCY_S)


def test_both_sides_talking_at_once_stay_within_the_limit(models, clips):
    placed, audio, cursor_s = [], [], GAP_S
    pairs = clips[CLIP_COUNT : CLIP_COUNT + 2 * OVERLAP_PAIR_COUNT]
    for mine, theirs in zip(pairs[::2], pairs[1::2], strict=True):
        for source, clip in ((Source.MIC, mine), (Source.SYSTEM, theirs)):
            placed.append(
                PlacedClip(source, cursor_s, cursor_s + clip.duration_s, clip.reference)
            )
            audio.append(clip.samples)
        cursor_s += max(mine.duration_s, theirs.duration_s) + GAP_S
    shown, pipeline, _ = replay(models, build_tracks(placed, audio))
    report("Both sides at once", placed, shown, pipeline)

    assert all(entries_of(clip, shown) for clip in placed), "a phrase was lost"
    # One phrase waits for the other's transcription (one model run at a time,
    # DECISIONS.md pipeline.py 1), so the queue's 0.3 s budget can't hold here.
    assert_latency_targets(pipeline, MAX_LATENCY_S, also_accepted=frozenset({"queue"}))


def test_long_monologue_is_cut_into_phrases_within_the_limit(models, clips):
    # PLAN.md §5.1 stress: one call-side speaker talks for about a minute.
    monologue = clips[-MONOLOGUE_CLIP_COUNT:]
    breath = np.zeros(round(MONOLOGUE_GAP_S * SAMPLE_RATE_HZ), dtype=np.float32)
    speech = np.concatenate([part for c in monologue for part in (c.samples, breath)])
    placed = [
        PlacedClip(
            Source.SYSTEM,
            GAP_S,
            GAP_S + speech.size / SAMPLE_RATE_HZ,
            " ".join(c.reference for c in monologue),
        )
    ]
    shown, pipeline, _ = replay(models, build_tracks(placed, [speech]))
    print(f"\n=== Long monologue ({speech.size / SAMPLE_RATE_HZ:.0f} s) ===")
    for item in shown:
        print(f"[{item.start_s:6.1f} s] {item.entry.speaker}: {item.entry.english}")
    print_latency(pipeline)

    reference_words = len(normalize(placed[0].reference).split())
    shown_words = sum(len(normalize(i.entry.armenian).split()) for i in shown)
    assert shown_words >= MIN_WORDS_KEPT * reference_words, "speech was lost"
    for timing in pipeline.metrics.timings:
        assert timing.end_s - timing.start_s <= MAX_SEGMENT_S
    assert_latency_targets(pipeline, MAX_LATENCY_S)

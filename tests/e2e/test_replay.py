"""End-to-end replay with the real models (PLAN.md §5.2 layer 3).

Recorded Armenian speech (FLEURS hy_am, downloaded by the ASR benchmark) is fed
through the full pipeline at real-time speed in place of live capture. Run on the
laptop with:
    .\\.venv\\Scripts\\python.exe -m pytest -m e2e -s
"""

import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from asr_benchmark import DATA_DIR, error_rates, load_fleurs
from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, SessionClock, Source
from live_hy_en.metrics import MAX_LATENCY_S, format_summary
from live_hy_en.output import TranscriptEntry
from live_hy_en.pipeline import Pipeline, load_models
from live_hy_en.segmenter import VAD_MODEL_PATH, WINDOW_SAMPLES
from live_hy_en.speakers import MIC_LABEL, SPEAKER_MODEL_PATH
from live_hy_en.translator import MT_MODEL, MT_MODELS_DIR

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not all(
            path.exists()
            for path in (
                VAD_MODEL_PATH,
                SPEAKER_MODEL_PATH,
                MT_MODELS_DIR / MT_MODEL,
                DATA_DIR / "test.tar.gz",
            )
        ),
        reason="needs the downloaded models and FLEURS data on the laptop",
    ),
]

CLIP_COUNT = 12
OVERLAP_PAIR_COUNT = 4
GAP_S = 1.5
CHUNK_S = WINDOW_SAMPLES / SAMPLE_RATE_HZ
# A clip's entries are those that start within it, allowing for VAD padding.
CLIP_MARGIN_S = 0.5


@dataclass(frozen=True)
class PlacedClip:
    """One recording placed on a source's track, in seconds from replay start."""

    source: Source
    start_s: float
    end_s: float
    reference: str


@dataclass(frozen=True)
class ShownEntry:
    entry: TranscriptEntry
    start_s: float
    shown_s: float


@pytest.fixture(scope="module")
def models():
    return load_models()


@pytest.fixture(scope="module")
def clips():
    return load_fleurs(CLIP_COUNT + 2 * OVERLAP_PAIR_COUNT)


def _tracks(placed: list[PlacedClip], audio: list[np.ndarray]) -> dict:
    length_s = max(clip.end_s for clip in placed) + GAP_S
    tracks = {
        source: np.zeros(round(length_s * SAMPLE_RATE_HZ), dtype=np.float32)
        for source in Source
    }
    for clip, samples in zip(placed, audio, strict=True):
        start = round(clip.start_s * SAMPLE_RATE_HZ)
        tracks[clip.source][start : start + samples.size] = samples
    return tracks


def _replay(models, tracks: dict) -> tuple[list[ShownEntry], Pipeline, float]:
    """Feed the tracks in real time, as capture would; return what was shown."""
    clock = SessionClock()
    shown: list[ShownEntry] = []
    replay_start_s = 0.0

    def show(entry: TranscriptEntry) -> None:
        start_s = (entry.spoken_at - started_at).total_seconds() - replay_start_s
        shown.append(ShownEntry(entry, start_s, clock.elapsed_s() - replay_start_s))

    started_at = datetime.now()
    pipeline = Pipeline(models, clock, started_at, show=show)
    pipeline.start()
    chunk = WINDOW_SAMPLES
    replay_start_s = clock.elapsed_s()
    length = len(tracks[Source.MIC])
    for index, offset in enumerate(range(0, length - chunk + 1, chunk)):
        due_s = replay_start_s + (index + 1) * CHUNK_S
        time.sleep(max(0.0, due_s - clock.elapsed_s()))
        for source, track in tracks.items():
            samples = track[offset : offset + chunk]
            # Loopback sends nothing while the call is silent; the mic always does.
            if source is Source.SYSTEM and not samples.any():
                continue
            frame_start_s = clock.elapsed_s() - CHUNK_S
            pipeline.frames[source].put(AudioFrame(source, frame_start_s, samples))
    playback_end_s = clock.elapsed_s() - replay_start_s
    pipeline.stop()
    return shown, pipeline, playback_end_s


def _entries_of(clip: PlacedClip, shown: list[ShownEntry]) -> list[ShownEntry]:
    return [
        item
        for item in shown
        if clip.start_s - CLIP_MARGIN_S <= item.start_s <= clip.end_s + CLIP_MARGIN_S
        and (item.entry.speaker == MIC_LABEL) == (clip.source is Source.MIC)
    ]


def _report(title: str, placed, shown, pipeline) -> None:
    print(f"\n=== {title} ===")
    pairs = []
    for clip in placed:
        entries = _entries_of(clip, shown)
        hypothesis = " ".join(item.entry.armenian for item in entries)
        pairs.append((clip.reference, hypothesis))
        print(
            f"[{clip.start_s:6.1f}-{clip.end_s:6.1f} s] {clip.source.value:6} "
            f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}: "
            + " | ".join(f"{i.entry.speaker}: {i.entry.english}" for i in entries)
        )
    wer, cer = error_rates(pairs)
    print(f"WER {wer:.1%}, CER {cer:.1%} over {len(placed)} clips")
    print(format_summary(pipeline.metrics.summary()))


def test_alternating_call_is_transcribed_live_within_the_limit(models, clips):
    placed, audio, cursor_s = [], [], GAP_S
    for index, clip in enumerate(clips[:CLIP_COUNT]):
        # One in three phrases is mine; the rest come from the call.
        source = Source.MIC if index % 3 == 0 else Source.SYSTEM
        placed.append(
            PlacedClip(source, cursor_s, cursor_s + clip.duration_s, clip.reference)
        )
        audio.append(clip.samples)
        cursor_s += clip.duration_s + GAP_S
    shown, pipeline, playback_end_s = _replay(models, _tracks(placed, audio))
    _report("Alternating call", placed, shown, pipeline)

    assert all(_entries_of(clip, shown) for clip in placed), "a phrase was lost"
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
    summary = pipeline.metrics.summary()
    assert summary.max_s <= MAX_LATENCY_S


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
    shown, pipeline, _ = _replay(models, _tracks(placed, audio))
    _report("Both sides at once", placed, shown, pipeline)

    assert all(_entries_of(clip, shown) for clip in placed), "a phrase was lost"
    assert pipeline.metrics.summary().max_s <= MAX_LATENCY_S

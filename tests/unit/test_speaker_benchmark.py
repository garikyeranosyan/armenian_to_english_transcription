"""Tests for the speaker benchmark (PLAN.md §5.1: stable call-side speaker labels)."""

import random

import numpy as np
import pytest

from live_hy_en.capture import Source
from speaker_benchmark import (
    SAMPLE_RATE_HZ,
    CachedEncoder,
    Recording,
    Score,
    SpeechSegment,
    add_scores,
    choose_recordings,
    format_table,
    parse_cv_tsv,
    score_call,
    shorten,
    simulate_calls,
    to_call_quality,
)

CV_TSV = (
    "client_id\tpath\tsentence_id\tsentence\tsentence_domain\tup_votes\t"
    "down_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
    'aaa\tcommon_voice_hy-AM_1.mp3\t1\tԲարև "ձեզ"\t\t2\t0\t\t\t\t\thy-AM\t\n'
    "bbb\tcommon_voice_hy-AM_2.mp3\t2\tԻնչպես եք\t\t2\t0\t\t\t\t\thy-AM\t\n"
)


def _segment(seconds: float = 2.0, lead_in_s: float = 0.0) -> SpeechSegment:
    lead_in = np.zeros(round(lead_in_s * SAMPLE_RATE_HZ), dtype=np.float32)
    speech = np.linspace(0.1, 0.9, round(seconds * SAMPLE_RATE_HZ), dtype=np.float32)
    return SpeechSegment(
        Source.SYSTEM, 0.0, seconds, np.concatenate([lead_in, speech]), lead_in.size
    )


def test_split_file_gives_each_recording_its_speaker():
    recordings = parse_cv_tsv(CV_TSV)
    assert recordings == [
        Recording("common_voice_hy-AM_1.mp3", "aaa"),
        Recording("common_voice_hy-AM_2.mp3", "bbb"),
    ]


def test_speakers_with_most_recordings_are_chosen():
    recordings = [Recording(f"{s}{i}.mp3", s) for s in "abc" for i in range(ord(s))]
    recordings += [Recording(f"d{i}.mp3", "d") for i in range(2)]
    chosen = choose_recordings(recordings, speaker_count=2, clips_per_speaker=3)
    assert [r.speaker_id for r in chosen] == ["c"] * 3 + ["b"] * 3


def test_merges_and_splits_are_counted():
    # B's phrase sits under A's label 1 (merged); A once gets label 3 (split).
    speakers = ["A", "A", "B", "A", "A"]
    labels = ["1", "1", "1", "3", "1"]
    score = score_call(speakers, labels)
    assert (score.merged_count, score.split_count) == (1, 1)
    assert (score.speaker_count, score.label_count) == (2, 2)
    assert score.merged_rate == pytest.approx(1 / 5)


def test_perfect_labels_score_zero():
    score = score_call(["A", "B", "A"], ["1", "2", "1"])
    assert score.merged_rate == score.split_rate == 0


def test_scores_of_several_calls_add_up():
    total = add_scores([Score(4, 1, 0, 2, 2), Score(6, 0, 2, 3, 4)])
    assert total == Score(10, 1, 2, 5, 6)


def test_short_reply_is_cut_from_the_speech_alone():
    segment = _segment(seconds=2.0, lead_in_s=0.3)
    reply = shorten(segment, 0.5)
    assert reply.speech_samples.size == round(0.5 * SAMPLE_RATE_HZ)
    assert reply.speech_samples.min() > 0  # no padding left in
    assert reply.end_s - reply.start_s == pytest.approx(0.5)


def test_calls_alternate_between_their_own_speakers():
    phrases = {speaker: [_segment()] for speaker in "abcdefg"}
    calls = simulate_calls(phrases, 6, 12, random.Random(0))
    assert [len({t.speaker_id for t in call}) for call in calls] == [2, 3, 4] * 2
    for call in calls:
        assert len(call) == 12
        pairs = zip(call, call[1:], strict=False)
        assert all(a.speaker_id != b.speaker_id for a, b in pairs)


def test_calls_repeat_with_the_same_seed():
    phrases = {speaker: [_segment()] for speaker in "abcd"}
    first = simulate_calls(phrases, 3, 8, random.Random(1))
    second = simulate_calls(phrases, 3, 8, random.Random(1))
    assert [[t.speaker_id for t in c] for c in first] == [
        [t.speaker_id for t in c] for c in second
    ]


def test_call_codec_keeps_length_and_changes_the_sound():
    rng = np.random.default_rng(0)
    samples = (0.1 * rng.standard_normal(SAMPLE_RATE_HZ)).astype(np.float32)
    decoded = to_call_quality(samples)
    assert decoded.dtype == np.float32
    assert decoded.size == pytest.approx(samples.size, abs=SAMPLE_RATE_HZ // 50)
    assert not np.allclose(decoded[: samples.size], samples, atol=1e-3)


def test_each_voiceprint_is_computed_once():
    class CountingEncoder:
        calls = 0

        def embed(self, samples: np.ndarray) -> np.ndarray:
            CountingEncoder.calls += 1
            return np.ones(3)

    encoder = CachedEncoder(CountingEncoder())
    samples = _segment().speech_samples
    encoder.embed(samples)
    encoder.embed(samples.copy())
    assert CountingEncoder.calls == 1


def test_table_has_a_row_per_threshold():
    results = [(0.4, Score(10, 1, 2, 4, 5)), (0.5, Score(10, 0, 3, 4, 6))]
    table = format_table(results, call_count=2).splitlines()
    assert len(table) == 4
    assert table[2] == "| 0.40 | 10.0% | 20.0% | 2.50 | 2.00 |"

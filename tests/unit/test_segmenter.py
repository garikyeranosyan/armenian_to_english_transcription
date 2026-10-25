"""Tests for segmentation (PLAN.md §2 principle 2 and §3 module 3)."""

import numpy as np
import pytest

from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, Source
from live_hy_en.segmenter import (
    END_SILENCE_MS,
    LONG_PHRASE_S,
    MAX_SEGMENT_S,
    MIN_SPEECH_MS,
    POST_SPEECH_PAD_MS,
    PRE_SPEECH_PAD_MS,
    PRE_SPEECH_PAD_WINDOWS,
    SHORT_PAUSE_MS,
    VAD_MODEL_PATH,
    WINDOW_S,
    WINDOW_SAMPLES,
    Segmenter,
    SileroVad,
    SpeechSegment,
    VadModelError,
)

# Resampler output arrives in uneven bursts, never in whole VAD windows.
FRAME_SAMPLES = 490
SPEECH_LEVEL = 0.5
# A soft first sound the detector doesn't yet call speech.
SOFT_LEVEL = 0.02
LOUD_RMS = 0.05
PRE_PAD_S = PRE_SPEECH_PAD_MS / 1000
POST_PAD_S = POST_SPEECH_PAD_MS / 1000


class FakeDetector:
    """Calls loud windows speech and quiet windows silence."""

    def __init__(self) -> None:
        self.reset_count = 0

    def speech_probability(self, window: np.ndarray) -> float:
        assert window.size == WINDOW_SAMPLES
        return 0.9 if np.sqrt(np.mean(window**2)) > LOUD_RMS else 0.05

    def reset(self) -> None:
        self.reset_count += 1


def _speech(seconds: float) -> np.ndarray:
    return np.full(round(seconds * SAMPLE_RATE_HZ), SPEECH_LEVEL, dtype=np.float32)


def _soft(seconds: float) -> np.ndarray:
    return np.full(round(seconds * SAMPLE_RATE_HZ), SOFT_LEVEL, dtype=np.float32)


def _duration_s(segment: SpeechSegment) -> float:
    return segment.samples.size / SAMPLE_RATE_HZ


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SAMPLE_RATE_HZ), dtype=np.float32)


def _frames(audio: np.ndarray, start_s: float = 0.0) -> list[AudioFrame]:
    return [
        AudioFrame(
            Source.SYSTEM,
            start_s + offset / SAMPLE_RATE_HZ,
            audio[offset : offset + FRAME_SAMPLES],
        )
        for offset in range(0, audio.size, FRAME_SAMPLES)
    ]


def _run(segmenter: Segmenter, frames: list[AudioFrame]) -> list[SpeechSegment]:
    segments = []
    for frame in frames:
        segments += segmenter.process(frame)
    return segments


@pytest.fixture
def segmenter() -> Segmenter:
    return Segmenter(FakeDetector(), Source.SYSTEM)


def test_silence_gives_no_segments(segmenter):
    assert _run(segmenter, _frames(_silence(5))) == []


def test_phrase_ends_after_a_pause(segmenter):
    audio = np.concatenate([_silence(1), _speech(2), _silence(1)])
    [segment] = _run(segmenter, _frames(audio))
    assert segment.source is Source.SYSTEM
    assert segment.start_s == pytest.approx(1.0, abs=2 * WINDOW_S)
    assert segment.end_s == pytest.approx(3.0, abs=2 * WINDOW_S)


def test_audio_around_the_speech_is_kept(segmenter):
    audio = np.concatenate([_silence(1), _speech(2), _silence(1)])
    [segment] = _run(segmenter, _frames(audio))
    assert _duration_s(segment) == pytest.approx(
        PRE_PAD_S + 2 + POST_PAD_S, abs=2 * WINDOW_S
    )


def test_soft_start_of_the_first_word_is_kept(segmenter):
    audio = np.concatenate([_silence(1), _soft(0.2), _speech(2), _silence(1)])
    [segment] = _run(segmenter, _frames(audio))
    soft_samples = np.count_nonzero(segment.samples == SOFT_LEVEL)
    assert soft_samples == round(0.2 * SAMPLE_RATE_HZ)


def test_speech_samples_leave_out_the_padding(segmenter):
    audio = np.concatenate([_silence(1), _speech(2), _silence(1)])
    [segment] = _run(segmenter, _frames(audio))
    speech = segment.speech_samples
    assert speech.size / SAMPLE_RATE_HZ == pytest.approx(
        segment.end_s - segment.start_s
    )
    # All of the speech, and at most a window of silence at either end.
    assert np.count_nonzero(speech) == round(2 * SAMPLE_RATE_HZ)


def test_padding_does_not_delay_the_end_time(segmenter):
    # Latency is measured from end_s, so it must stay where the speech stopped.
    audio = np.concatenate([_speech(2), _silence(1)])
    [segment] = _run(segmenter, _frames(audio))
    assert segment.end_s == pytest.approx(2.0, abs=2 * WINDOW_S)


def test_pause_is_detected_within_the_latency_budget(segmenter):
    audio = np.concatenate([_speech(2), _silence(1)])
    frames = _frames(audio)
    for frame in frames:
        if segmenter.process(frame):
            detected_s = frame.start_s + frame.samples.size / SAMPLE_RATE_HZ
            break
    pause_s = detected_s - 2.0
    assert pause_s == pytest.approx(END_SILENCE_MS / 1000, abs=2 * WINDOW_S)


def test_short_pause_inside_a_phrase_does_not_split_it(segmenter):
    audio = np.concatenate([_speech(1), _silence(0.2), _speech(1), _silence(1)])
    assert len(_run(segmenter, _frames(audio))) == 1


def test_two_phrases_give_two_segments(segmenter):
    audio = np.concatenate([_speech(1), _silence(0.8), _speech(1), _silence(1)])
    first, second = _run(segmenter, _frames(audio))
    assert first.end_s < second.start_s
    # The second phrase's lead-in is the pause, not the end of the first phrase.
    lead_in = second.samples[: PRE_SPEECH_PAD_WINDOWS * WINDOW_SAMPLES]
    assert lead_in.size and not lead_in.any()


def test_long_phrase_ends_at_a_short_pause(segmenter):
    gap_s = 1.5 * SHORT_PAUSE_MS / 1000
    first_s = LONG_PHRASE_S + 1
    audio = np.concatenate([_speech(first_s), _silence(gap_s), _speech(2), _silence(1)])
    first, second = _run(segmenter, _frames(audio))
    assert first.end_s == pytest.approx(first_s, abs=2 * WINDOW_S)
    assert second.start_s == pytest.approx(first_s + gap_s, abs=2 * WINDOW_S)


def test_short_phrase_does_not_end_at_a_short_pause(segmenter):
    gap_s = 1.5 * SHORT_PAUSE_MS / 1000
    audio = np.concatenate([_speech(1), _silence(gap_s), _speech(1), _silence(1)])
    assert len(_run(segmenter, _frames(audio))) == 1


def test_tiny_burst_is_dropped(segmenter):
    burst_s = MIN_SPEECH_MS / 1000 / 2
    audio = np.concatenate([_silence(1), _speech(burst_s), _silence(1)])
    assert _run(segmenter, _frames(audio)) == []


def test_long_monologue_is_cut_at_the_maximum_length(segmenter):
    audio = np.concatenate([_speech(24), _silence(1)])
    segments = _run(segmenter, _frames(audio))
    assert len(segments) == 3
    assert all(s.end_s - s.start_s <= MAX_SEGMENT_S for s in segments)
    # Cut pieces join up with no audio lost or repeated; only the end is padded.
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert later.start_s == pytest.approx(earlier.end_s)
    total_s = sum(_duration_s(s) for s in segments)
    assert total_s == pytest.approx(24 + POST_PAD_S, abs=2 * WINDOW_S)


def test_gap_in_the_stream_ends_the_phrase():
    # Loopback sends nothing while the call is silent, so frames just stop.
    detector = FakeDetector()
    segmenter = Segmenter(detector, Source.SYSTEM)
    segments = _run(segmenter, _frames(_speech(2)))
    assert segments == []
    [segment] = _run(segmenter, _frames(_speech(1), start_s=12.0))
    assert segment.end_s == pytest.approx(2.0, abs=2 * WINDOW_S)
    assert detector.reset_count == 1


def test_times_follow_the_frames_after_a_gap(segmenter):
    _run(segmenter, _frames(_speech(1)))
    [_, second] = _run(segmenter, _frames(_speech(1), start_s=20.0)) + segmenter.flush()
    assert second.start_s == pytest.approx(20.0, abs=2 * WINDOW_S)
    # Nothing from before the gap is used as the lead-in.
    assert _duration_s(second) == pytest.approx(1.0, abs=2 * WINDOW_S)


def test_flush_returns_the_open_segment(segmenter):
    _run(segmenter, _frames(_speech(2)))
    [segment] = segmenter.flush()
    assert segment.end_s - segment.start_s == pytest.approx(2.0, abs=2 * WINDOW_S)
    assert segmenter.flush() == []


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(VadModelError, match="Download it from"):
        SileroVad(tmp_path / "missing.onnx")


needs_model = pytest.mark.skipif(
    not VAD_MODEL_PATH.is_file(), reason="Silero VAD model not downloaded"
)


@needs_model
def test_silero_scores_silence_as_not_speech():
    vad = SileroVad()
    probabilities = [
        vad.speech_probability(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
        for _ in range(20)
    ]
    assert max(probabilities) < 0.1


@needs_model
def test_silero_reset_repeats_its_scores():
    vad = SileroVad()
    noise = np.random.default_rng(0).normal(0, 0.1, WINDOW_SAMPLES * 10)
    windows = noise.astype(np.float32).reshape(-1, WINDOW_SAMPLES)
    first = [vad.speech_probability(window) for window in windows]
    vad.reset()
    second = [vad.speech_probability(window) for window in windows]
    assert first == pytest.approx(second)

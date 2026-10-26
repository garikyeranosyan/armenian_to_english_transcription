"""Tests for speaker labels (PLAN.md §5.1: mic is always Me; call labels are stable)."""

import numpy as np
import pytest

from live_hy_en.capture import SAMPLE_RATE_HZ, Source
from live_hy_en.segmenter import SpeechSegment
from live_hy_en.speakers import (
    MEL_BANDS,
    MIC_LABEL,
    SPEAKER_MODEL_URL,
    ModelLoadError,
    SpeakerTracker,
    WeSpeakerEncoder,
    compute_fbank,
)

# Fake voiceprints: two clearly different voices, and a variation of voice A.
VOICE_A = np.array([1.0, 0.0, 0.0])
VOICE_A_AGAIN = np.array([0.9, 0.2, 0.0])
VOICE_B = np.array([0.0, 1.0, 0.0])
VOICE_C = np.array([0.0, 0.0, 1.0])


class FakeEncoder:
    """Returns the voiceprint stored under each segment's first sample value."""

    def __init__(self) -> None:
        self.voiceprints: dict[float, np.ndarray] = {}
        self.calls = 0

    def segment(
        self,
        voiceprint: np.ndarray,
        source=Source.SYSTEM,
        speech_s: float = 1.0,
        lead_in_s: float = 0.0,
    ) -> SpeechSegment:
        key = float(len(self.voiceprints) + 1)
        self.voiceprints[key] = voiceprint
        speech = np.full(round(speech_s * SAMPLE_RATE_HZ), key, dtype=np.float32)
        lead_in = np.zeros(round(lead_in_s * SAMPLE_RATE_HZ), dtype=np.float32)
        return SpeechSegment(
            source, 0.0, speech_s, np.concatenate([lead_in, speech]), lead_in.size
        )

    def embed(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.embedded = samples
        return self.voiceprints.get(float(samples[0]), VOICE_A)


@pytest.fixture
def encoder():
    return FakeEncoder()


def _labels(tracker: SpeakerTracker, segments) -> list[str]:
    return [tracker.assign_speaker(segment) for segment in segments]


def test_microphone_is_always_me_and_skips_the_speaker_model(encoder):
    tracker = SpeakerTracker(encoder)
    segments = [encoder.segment(v, Source.MIC) for v in (VOICE_A, VOICE_B)]
    assert _labels(tracker, segments) == [MIC_LABEL, MIC_LABEL] == ["Me", "Me"]
    assert encoder.calls == 0


def test_first_call_side_voice_is_speaker_1(encoder):
    tracker = SpeakerTracker(encoder)
    assert tracker.assign_speaker(encoder.segment(VOICE_A)) == "Speaker 1"


def test_two_voices_get_different_labels(encoder):
    tracker = SpeakerTracker(encoder)
    segments = [encoder.segment(VOICE_A), encoder.segment(VOICE_B)]
    assert _labels(tracker, segments) == ["Speaker 1", "Speaker 2"]


def test_returning_speaker_keeps_their_first_label(encoder):
    tracker = SpeakerTracker(encoder)
    voices = [VOICE_A, VOICE_B, VOICE_A_AGAIN, VOICE_C, VOICE_B, VOICE_A]
    labels = _labels(tracker, [encoder.segment(v) for v in voices])
    assert labels == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
        "Speaker 3",
        "Speaker 2",
        "Speaker 1",
    ]


def test_my_speech_does_not_create_or_shift_call_side_speakers(encoder):
    tracker = SpeakerTracker(encoder)
    segments = [
        encoder.segment(VOICE_A),
        encoder.segment(VOICE_B, Source.MIC),
        encoder.segment(VOICE_B),
    ]
    assert _labels(tracker, segments) == ["Speaker 1", "Me", "Speaker 2"]


def test_threshold_decides_between_same_and_new_speaker(encoder):
    similarity = float(VOICE_A @ VOICE_A_AGAIN / np.linalg.norm(VOICE_A_AGAIN))
    strict = SpeakerTracker(encoder, similarity_threshold=similarity + 0.01)
    lenient = SpeakerTracker(encoder, similarity_threshold=similarity - 0.01)
    for tracker, second in ((strict, "Speaker 2"), (lenient, "Speaker 1")):
        segments = [encoder.segment(VOICE_A), encoder.segment(VOICE_A_AGAIN)]
        assert _labels(tracker, segments) == ["Speaker 1", second]


def test_closest_speaker_wins_when_several_are_similar(encoder):
    tracker = SpeakerTracker(encoder, similarity_threshold=0.1)
    between_closer_to_b = np.array([0.4, 0.9, 0.0])
    segments = [
        encoder.segment(VOICE_A),
        encoder.segment(VOICE_C),
        encoder.segment(VOICE_B),
        encoder.segment(between_closer_to_b),
    ]
    # B is only created because a threshold of 0.1 still rejects A and C for it.
    labels = _labels(tracker, segments)
    assert labels[-1] == labels[2]


def test_voiceprint_is_made_from_the_speech_without_padding(encoder):
    tracker = SpeakerTracker(encoder)
    segment = encoder.segment(VOICE_A, speech_s=2.0, lead_in_s=0.3)
    tracker.assign_speaker(segment)
    assert encoder.embedded.size == 2 * SAMPLE_RATE_HZ
    assert encoder.embedded.all()


def test_warm_up_runs_the_model_once_on_a_second_of_silence(encoder):
    captured = []
    encoder.embed = lambda samples: captured.append(samples) or VOICE_A
    SpeakerTracker(encoder).warm_up()
    [samples] = captured
    assert samples.size == SAMPLE_RATE_HZ
    assert not samples.any()


def test_warm_up_failure_gives_clear_error():
    class BrokenEncoder:
        def embed(self, samples: np.ndarray) -> np.ndarray:
            raise RuntimeError("Invalid input shape")

    with pytest.raises(ModelLoadError, match="Speaker model failed.*input shape"):
        SpeakerTracker(BrokenEncoder()).warm_up()


def test_missing_model_says_where_to_download_it(tmp_path):
    with pytest.raises(ModelLoadError, match="not found") as error:
        WeSpeakerEncoder(tmp_path / "missing.onnx")
    assert SPEAKER_MODEL_URL in str(error.value)


def test_features_have_one_mean_normalised_row_per_10_ms():
    rng = np.random.default_rng(0)
    samples = (0.1 * rng.standard_normal(SAMPLE_RATE_HZ)).astype(np.float32)
    features = compute_fbank(samples)
    # 25 ms frames every 10 ms, without padding the edges: 98 frames in 1 s.
    assert features.shape == (98, MEL_BANDS)
    assert features.dtype == np.float32
    assert np.allclose(features.mean(axis=0), 0, atol=1e-4)

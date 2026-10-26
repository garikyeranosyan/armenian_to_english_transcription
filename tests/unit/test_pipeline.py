"""Tests for the pipeline (PLAN.md §5.1: both sources, live output, bounded delays)."""

import threading
import time as time_module
from datetime import datetime, timedelta

import numpy as np
import pytest

import live_hy_en.pipeline as pipeline_module
from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, SessionClock, Source
from live_hy_en.config import AudioDevice, Settings
from live_hy_en.metrics import Stage
from live_hy_en.pipeline import Models, Pipeline, load_models, run
from live_hy_en.segmenter import WINDOW_SAMPLES
from live_hy_en.speakers import SpeakerTracker
from live_hy_en.transcriber import Transcriber
from live_hy_en.translator import Translator

STARTED_AT = datetime(2026, 10, 24, 14, 0, 0)
WAIT_S = 5.0
# Each fake phrase is speech at a distinct loudness, so the fakes below can tell
# phrases apart from their audio alone.
TEXT_BY_LOUDNESS = {0.2: "Բարև", 0.3: "Լավ եմ", 0.4: "Շնորհակալություն", 0.5: "Այո"}
SILENT_TEXT_LOUDNESS = 0.6
FAILING_LOUDNESS = 0.7


def _loudness(samples: np.ndarray) -> float:
    return round(float(np.abs(samples).max()), 1)


class FakeDetector:
    """Calls any window louder than near-silence speech."""

    def speech_probability(self, window: np.ndarray) -> float:
        return 1.0 if np.abs(window).max() > 0.01 else 0.0

    def reset(self) -> None:
        pass


class FakeRecognizer:
    """Returns the Armenian text for each phrase's loudness."""

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, samples: np.ndarray) -> str:
        self.calls += 1
        loudness = _loudness(samples)
        if loudness == FAILING_LOUDNESS:
            raise RuntimeError("CUDA error: an illegal memory access")
        if loudness == SILENT_TEXT_LOUDNESS:
            return ""
        return TEXT_BY_LOUDNESS.get(loudness, "Բարև")


class FakeEncoder:
    """Gives each loudness its own voice, so each makes a new call-side speaker."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, samples: np.ndarray) -> np.ndarray:
        self.calls += 1
        voice = np.zeros(8)
        voice[int(_loudness(samples) * 10)] = 1.0
        return voice


class FakeTextTranslator:
    def translate(self, text: str) -> str:
        return f"EN<{text}>"


def _models(recognizer=None, encoder=None, text_translator=None) -> Models:
    return Models(
        transcriber=Transcriber(recognizer or FakeRecognizer()),
        speakers=SpeakerTracker(encoder or FakeEncoder()),
        translator=Translator(text_translator or FakeTextTranslator()),
        make_detector=FakeDetector,
    )


def _phrase(
    source: Source, start_s: float, loudness: float, speech_s: float = 0.64
) -> list[AudioFrame]:
    """Frames of one phrase followed by a second of silence, from `start_s`."""
    speech = np.full(round(speech_s * SAMPLE_RATE_HZ), loudness, dtype=np.float32)
    samples = np.concatenate([speech, np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)])
    return [
        AudioFrame(source, start_s + offset / SAMPLE_RATE_HZ, samples[offset:end])
        for offset in range(0, samples.size, WINDOW_SAMPLES)
        if (end := offset + WINDOW_SAMPLES) <= samples.size
    ]


class Shown:
    """Collects shown entries and signals when a number of them has arrived."""

    def __init__(self) -> None:
        self.entries = []
        self._changed = threading.Condition()

    def __call__(self, entry) -> None:
        with self._changed:
            self.entries.append(entry)
            self._changed.notify_all()

    def wait_for(self, count: int) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: len(self.entries) >= count, WAIT_S)


@pytest.fixture
def shown():
    return Shown()


def _feed(pipeline: Pipeline, frames: list[AudioFrame]) -> None:
    for frame in frames:
        pipeline.frames[frame.source].put(frame)


def _run(models: Models, shown: Shown, frames: list[AudioFrame], **options):
    pipeline = Pipeline(models, SessionClock(), STARTED_AT, show=shown, **options)
    pipeline.start()
    _feed(pipeline, frames)
    pipeline.stop()
    return shown.entries


def test_both_sources_are_transcribed_translated_and_shown(shown):
    frames = _phrase(Source.MIC, 1.0, 0.2) + _phrase(Source.SYSTEM, 3.0, 0.3)
    entries = sorted(_run(_models(), shown, frames), key=lambda e: e.spoken_at)
    assert [(e.speaker, e.armenian, e.english) for e in entries] == [
        ("Me", "Բարև", "EN<Բարև>"),
        ("Speaker 1", "Լավ եմ", "EN<Լավ եմ>"),
    ]


def test_entry_time_is_the_phrase_start_as_time_of_day(shown):
    [entry] = _run(_models(), shown, _phrase(Source.SYSTEM, 72.0, 0.2))
    assert entry.spoken_at == STARTED_AT + timedelta(seconds=72.0)


def test_different_call_side_voices_get_their_own_labels(shown):
    frames = (
        _phrase(Source.SYSTEM, 1.0, 0.2)
        + _phrase(Source.SYSTEM, 3.0, 0.3)
        + _phrase(Source.SYSTEM, 5.0, 0.2)
    )
    entries = _run(_models(), shown, frames)
    assert [e.speaker for e in entries] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_entries_are_shown_while_the_session_runs(shown):
    pipeline = Pipeline(_models(), SessionClock(), STARTED_AT, show=shown)
    pipeline.start()
    _feed(pipeline, _phrase(Source.SYSTEM, 1.0, 0.2))
    # The mic sends nothing at all, and the call side is still shown.
    assert shown.wait_for(1)
    pipeline.stop()


def test_phrase_ends_when_a_source_falls_silent(shown):
    # Loopback stops sending when the call goes quiet, with no pause frames.
    pipeline = Pipeline(_models(), SessionClock(), STARTED_AT, show=shown)
    pipeline.start()
    _feed(pipeline, _phrase(Source.SYSTEM, 1.0, 0.2)[:10])  # speech only
    assert shown.wait_for(1)
    pipeline.stop()
    assert len(shown.entries) == 1


def test_a_phrase_still_open_at_shutdown_is_shown(shown):
    frames = _phrase(Source.MIC, 1.0, 0.2)[:10]  # speech only, no pause after
    [entry] = _run(_models(), shown, frames)
    assert entry.armenian == "Բարև"


def test_empty_transcripts_are_not_shown_and_create_no_speaker(shown):
    encoder = FakeEncoder()
    frames = _phrase(Source.SYSTEM, 1.0, SILENT_TEXT_LOUDNESS) + _phrase(
        Source.SYSTEM, 3.0, 0.2
    )
    entries = _run(_models(encoder=encoder), shown, frames)
    assert [(e.speaker, e.armenian) for e in entries] == [("Speaker 1", "Բարև")]
    assert encoder.calls == 1


def test_a_failing_phrase_does_not_stop_the_session(shown, caplog):
    frames = _phrase(Source.SYSTEM, 1.0, FAILING_LOUDNESS) + _phrase(
        Source.SYSTEM, 3.0, 0.2
    )
    entries = _run(_models(), shown, frames)
    assert [e.armenian for e in entries] == ["Բարև"]
    assert "Could not transcribe the system phrase at 1.0 s" in caplog.text
    assert "illegal memory access" in caplog.text


def test_a_failing_translation_does_not_stop_the_session(shown, caplog):
    class FlakyTextTranslator:
        def translate(self, text: str) -> str:
            if text == "Բարև":
                raise RuntimeError("out of memory")
            return f"EN<{text}>"

    frames = _phrase(Source.SYSTEM, 1.0, 0.2) + _phrase(Source.SYSTEM, 3.0, 0.3)
    entries = _run(_models(text_translator=FlakyTextTranslator()), shown, frames)
    assert [e.english for e in entries] == ["EN<Լավ եմ>"]
    assert "Could not translate or print the phrase at 1.0 s" in caplog.text


def test_full_queue_waits_and_warns_but_drops_nothing(shown, caplog):
    release = threading.Event()

    class SlowRecognizer(FakeRecognizer):
        def recognize(self, samples: np.ndarray) -> str:
            release.wait(WAIT_S)
            return super().recognize(samples)

    loudness_order = [0.2, 0.3, 0.4, 0.5]
    frames = [
        frame
        for index, loudness in enumerate(loudness_order)
        for frame in _phrase(Source.SYSTEM, 1.0 + 2 * index, loudness)
    ]
    pipeline = Pipeline(
        _models(recognizer=SlowRecognizer()),
        SessionClock(),
        STARTED_AT,
        show=shown,
        max_queued_segments=1,
    )
    pipeline.start()
    _feed(pipeline, frames)
    stopper = threading.Thread(target=pipeline.stop)
    stopper.start()
    # Give the segmenter time to fill the one-slot queue and wait on it.
    stopper.join(0.5)
    release.set()
    stopper.join(WAIT_S)
    assert [e.armenian for e in shown.entries] == [
        TEXT_BY_LOUDNESS[loudness] for loudness in loudness_order
    ]
    assert "Transcription is falling behind" in caplog.text


def test_translation_overlaps_the_next_transcription(shown):
    second_transcription_started = threading.Event()
    overlapped = []

    class CountingRecognizer(FakeRecognizer):
        def recognize(self, samples: np.ndarray) -> str:
            if self.calls == 1:
                second_transcription_started.set()
            return super().recognize(samples)

    class WaitingTextTranslator(FakeTextTranslator):
        def translate(self, text: str) -> str:
            # With one worker doing both, the second phrase couldn't start yet.
            overlapped.append(second_transcription_started.wait(WAIT_S))
            return super().translate(text)

    frames = _phrase(Source.SYSTEM, 1.0, 0.2) + _phrase(Source.SYSTEM, 3.0, 0.3)
    models = _models(
        recognizer=CountingRecognizer(), text_translator=WaitingTextTranslator()
    )
    _run(models, shown, frames)
    assert overlapped[0] is True


def test_every_shown_phrase_is_timed_through_each_stage(shown, caplog):
    caplog.set_level("DEBUG", logger="live_hy_en.metrics")
    # The frames below are from 1 s to 5 s, so start the clock past them.
    clock = SessionClock(now=lambda: 100.0 + time_module.monotonic())
    pipeline = Pipeline(_models(), clock, STARTED_AT, show=shown)
    pipeline.start()
    _feed(pipeline, _phrase(Source.MIC, 1.0, 0.2) + _phrase(Source.SYSTEM, 3.0, 0.3))
    pipeline.stop()
    summary = pipeline.metrics.summary()
    assert summary.phrase_count == len(shown.entries) == 2
    assert summary.over_limit_count == 0
    lines = [r.getMessage() for r in caplog.records if r.name == "live_hy_en.metrics"]
    assert len(lines) == 2
    assert all(stage.value in line for line in lines for stage in Stage)


class FakeWarmable:
    """Stands in for one model wrapper and records its warm-up."""

    def __init__(self, model) -> None:
        self.model = model
        self.warmed_up = False

    def warm_up(self) -> None:
        self.warmed_up = True


def test_load_models_warms_up_every_model_before_listening(monkeypatch, capsys):
    for name in (
        "FasterWhisperRecognizer",
        "WeSpeakerEncoder",
        "CTranslate2Translator",
    ):
        monkeypatch.setattr(pipeline_module, name, lambda: object())
    for name in ("Transcriber", "SpeakerTracker", "Translator"):
        monkeypatch.setattr(pipeline_module, name, FakeWarmable)
    models = load_models()
    assert models.transcriber.warmed_up
    assert models.speakers.warmed_up
    assert models.translator.warmed_up
    assert "Loading models" in capsys.readouterr().out


DEVICE_MIC = AudioDevice(1, "Headset Microphone", 1, 48_000)
DEVICE_LOOPBACK = AudioDevice(7, "Speakers [Loopback]", 2, 48_000)


class FakeCapture:
    """Stands in for AudioCapture and records how it was used."""

    instances: list["FakeCapture"] = []

    def __init__(self, audio, device, source, clock, frames) -> None:
        self.device = device
        self.source = source
        self.frames = frames
        self.started = self.stopped = False
        FakeCapture.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_run_captures_both_devices_until_ctrl_c(monkeypatch, capsys):
    FakeCapture.instances = []
    monkeypatch.setattr(pipeline_module, "load_models", _models)
    monkeypatch.setattr(pipeline_module, "AudioCapture", FakeCapture)

    def ctrl_c(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module.time, "sleep", ctrl_c)
    run(Settings(mic=DEVICE_MIC, loopback=DEVICE_LOOPBACK), audio=object())
    mic, system = FakeCapture.instances
    assert (mic.device, mic.source) == (DEVICE_MIC, Source.MIC)
    assert (system.device, system.source) == (DEVICE_LOOPBACK, Source.SYSTEM)
    assert all(capture.started and capture.stopped for capture in (mic, system))
    output = capsys.readouterr().out
    assert "Listening to 'Headset Microphone' and 'Speakers [Loopback]'" in output
    assert "Stopping" in output
    assert "No phrases were shown" in output

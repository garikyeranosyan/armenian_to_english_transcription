"""Tests for transcription (PLAN.md §3 module 5, §6 risk 1: wrong or empty text)."""

from pathlib import Path
from types import SimpleNamespace

import ctranslate2
import faster_whisper
import numpy as np
import pytest

from live_hy_en.capture import SAMPLE_RATE_HZ
from live_hy_en.transcriber import (
    ASR_MODEL,
    CPU_COMPUTE_TYPE,
    GPU_COMPUTE_TYPE,
    LANGUAGE,
    TEMPERATURES,
    FasterWhisperRecognizer,
    ModelLoadError,
    Transcriber,
    is_armenian,
)

SEGMENT = np.full(SAMPLE_RATE_HZ * 2, 0.1, dtype=np.float32)


class FakeRecognizer:
    """Returns a fixed transcript and records the audio it was given."""

    def __init__(self, text: str = "Բարև, ինչպե՞ս ես։") -> None:
        self.text = text
        self.calls: list[np.ndarray] = []

    def recognize(self, samples: np.ndarray) -> str:
        self.calls.append(samples)
        return self.text


def test_armenian_speech_becomes_armenian_text():
    recognizer = FakeRecognizer("Բարև, ինչպե՞ս ես։")
    assert Transcriber(recognizer).transcribe(SEGMENT) == "Բարև, ինչպե՞ս ես։"
    assert recognizer.calls[0] is SEGMENT


def test_extra_whitespace_is_removed():
    transcriber = Transcriber(FakeRecognizer("  Լավ   եմ,\n շնորհակալություն։ "))
    assert transcriber.transcribe(SEGMENT) == "Լավ եմ, շնորհակալություն։"


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_output_is_dropped(text):
    assert Transcriber(FakeRecognizer(text)).transcribe(SEGMENT) is None


@pytest.mark.parametrize("text", ["Barev, inchpes es?", "Привет, как дела?"])
def test_wrong_script_output_is_dropped(text, caplog):
    assert Transcriber(FakeRecognizer(text)).transcribe(SEGMENT) is None
    assert "not in Armenian script" in caplog.text


def test_armenian_with_a_foreign_name_is_kept():
    transcriber = Transcriber(FakeRecognizer("Ես օգտագործում եմ Zoom"))
    assert transcriber.transcribe(SEGMENT) == "Ես օգտագործում եմ Zoom"


def test_warm_up_runs_the_model_once_on_a_second_of_silence():
    recognizer = FakeRecognizer()
    Transcriber(recognizer).warm_up()
    [samples] = recognizer.calls
    assert samples.dtype == np.float32
    assert samples.size == SAMPLE_RATE_HZ
    assert not samples.any()


def test_warm_up_failure_gives_clear_error():
    class BrokenRecognizer:
        def recognize(self, samples: np.ndarray) -> str:
            raise RuntimeError("Could not load library cudnn_ops64_9.dll")

    with pytest.raises(ModelLoadError, match="first run.*cudnn_ops64_9"):
        Transcriber(BrokenRecognizer()).warm_up()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Բարև, ինչպե՞ս ես։", True),
        ("", True),
        ("2026", True),
        ("Barev, inchpes es?", False),
        ("Привет, как дела?", False),
        ("Ես օգտագործում եմ Zoom", True),
    ],
)
def test_is_armenian_flags_the_wrong_script(text, expected):
    assert is_armenian(text) is expected


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel and records how it was used."""

    instances: list["FakeWhisperModel"] = []

    def __init__(self, model, **options) -> None:
        self.model = model
        self.options = options
        self.transcribe_options: dict = {}
        FakeWhisperModel.instances.append(self)

    def transcribe(self, samples, **options):
        self.transcribe_options = options
        segments = [
            SimpleNamespace(text=" Բարև, "),
            SimpleNamespace(text="ինչպե՞ս ես։ "),
        ]
        return iter(segments), None


@pytest.fixture
def fake_whisper(monkeypatch):
    FakeWhisperModel.instances = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


def _gpu_count(monkeypatch, count: int) -> None:
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: count)


def test_default_model_is_the_benchmark_choice(fake_whisper, monkeypatch):
    _gpu_count(monkeypatch, 1)
    FasterWhisperRecognizer()
    assert fake_whisper.instances[0].model == ASR_MODEL
    assert Path(ASR_MODEL).name == "buzzasr-armenian"


def test_any_model_can_be_plugged_in(fake_whisper, monkeypatch):
    _gpu_count(monkeypatch, 1)
    FasterWhisperRecognizer("path/to/armenian-fine-tune")
    assert fake_whisper.instances[0].model == "path/to/armenian-fine-tune"


def test_gpu_is_used_when_present(fake_whisper, monkeypatch):
    _gpu_count(monkeypatch, 1)
    FasterWhisperRecognizer()
    options = fake_whisper.instances[0].options
    assert options["device"] == "cuda"
    assert options["compute_type"] == GPU_COMPUTE_TYPE


def test_cpu_is_the_fallback_without_a_gpu(fake_whisper, monkeypatch, caplog):
    _gpu_count(monkeypatch, 0)
    FasterWhisperRecognizer()
    options = fake_whisper.instances[0].options
    assert options["device"] == "cpu"
    assert options["compute_type"] == CPU_COMPUTE_TYPE
    assert "No CUDA GPU" in caplog.text


def test_explicit_cpu_gets_a_cpu_compute_type(fake_whisper):
    FasterWhisperRecognizer(device="cpu")
    assert fake_whisper.instances[0].options["compute_type"] == CPU_COMPUTE_TYPE


def test_token_is_passed_to_the_download(fake_whisper):
    FasterWhisperRecognizer(device="cpu", token="hf_secret")
    assert fake_whisper.instances[0].options["use_auth_token"] == "hf_secret"


def test_language_is_fixed_to_armenian_and_segments_are_joined(fake_whisper):
    recognizer = FasterWhisperRecognizer(device="cpu")
    assert recognizer.recognize(SEGMENT) == "Բարև, ինչպե՞ս ես։"
    assert fake_whisper.instances[0].transcribe_options["language"] == LANGUAGE == "hy"


def test_load_failure_gives_clear_error(monkeypatch):
    def failing_model(model, **options):
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(faster_whisper, "WhisperModel", failing_model)
    with pytest.raises(ModelLoadError, match="buzzasr-armenian.*on cuda.*CUDA driver"):
        FasterWhisperRecognizer(device="cuda")


def test_unclear_phrases_are_retried_at_most_twice(fake_whisper):
    # Each retry is a full decode; six attempts broke the 5 s limit (PLAN.md §2).
    FasterWhisperRecognizer("any-model", device="cpu").recognize(SEGMENT)
    temperatures = fake_whisper.instances[0].transcribe_options["temperature"]
    assert temperatures == TEMPERATURES
    assert temperatures[0] == 0.0 and len(temperatures) == 3

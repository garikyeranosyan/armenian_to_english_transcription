"""Tests for transcription (PLAN.md §3 module 5, §6 risk 1: wrong or empty text)."""

import numpy as np
import pytest

from live_hy_en.capture import SAMPLE_RATE_HZ
from live_hy_en.transcriber import Transcriber, is_armenian

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

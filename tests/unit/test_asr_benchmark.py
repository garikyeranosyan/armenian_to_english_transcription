"""Tests for the ASR benchmark (PLAN.md §5.1: word/character error rate on Armenian)."""

import io

import numpy as np
import pytest
import soundfile

import asr_benchmark
from asr_benchmark import (
    ASR_BUDGET_S,
    HF_TOKEN_NAME,
    SAMPLE_RATE_HZ,
    Clip,
    ClipResult,
    download,
    edit_distance,
    error_rates,
    evaluate,
    format_table,
    load_hf_token,
    normalize,
    parse_fleurs_tsv,
    read_wav,
    summarize,
)


def _clip(clip_id: str = "1", seconds: float = 4.0, reference: str = "բարև") -> Clip:
    samples = np.zeros(round(seconds * SAMPLE_RATE_HZ), dtype=np.float32)
    return Clip(clip_id, samples, reference)


def _wav_bytes(
    samples: np.ndarray, rate_hz: int, subtype: str, wav_format: str = "WAV"
) -> bytes:
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, rate_hz, subtype=subtype, format=wav_format)
    return buffer.getvalue()


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Բարև, ինչպե՞ս ես։") == "բարև ինչպես ես"


def test_normalize_keeps_in_word_question_mark_from_splitting_the_word():
    assert normalize("ինչպե՞ս").split() == ["ինչպես"]


def test_normalize_treats_both_spellings_of_yev_alike():
    assert normalize("Երևան եւ Գյումրի") == normalize("Երեւան և Գյումրի")


@pytest.mark.parametrize(
    ("reference", "hypothesis", "distance"),
    [("abc", "abc", 0), ("abc", "abd", 1), ("abc", "", 3), ("", "ab", 2)],
)
def test_edit_distance(reference, hypothesis, distance):
    assert edit_distance(reference, hypothesis) == distance


def test_error_rates_count_words_and_characters():
    wer, cer = error_rates([("լավ եմ ես", "լավ էմ ես")])
    assert wer == pytest.approx(1 / 3)
    assert cer == pytest.approx(1 / 9)


def test_error_rates_are_pooled_over_the_corpus():
    # Pooled: 1 wrong word out of 4 in total, not the mean of 100 % and 0 %.
    wer, _ = error_rates([("բարև", "բարի"), ("լավ եմ ես", "լավ եմ ես")])
    assert wer == pytest.approx(1 / 4)


def test_punctuation_differences_are_not_errors():
    assert error_rates([("Բարև, ինչպե՞ս ես։", "բարև ինչպես ես")]) == (0.0, 0.0)


def test_parse_fleurs_tsv_maps_file_to_id_and_raw_transcript():
    tsv = (
        "1001\t123.wav\tԲարև, աշխարհ։\tբարև աշխարհ\tb a r e v\t64000\tFEMALE\n"
        "1002\t456.wav\tԼավ եմ։\tլավ եմ\tl a v\t32000\tMALE\n"
    )
    assert parse_fleurs_tsv(tsv) == {
        "123.wav": ("1001", "Բարև, աշխարհ։"),
        "456.wav": ("1002", "Լավ եմ։"),
    }


@pytest.mark.parametrize(
    ("subtype", "wav_format"),
    [
        ("PCM_16", "WAV"),
        ("PCM_24", "WAV"),
        ("FLOAT", "WAV"),
        # WAVE_FORMAT_EXTENSIBLE headers, which the wave module rejects.
        ("PCM_16", "WAVEX"),
        ("FLOAT", "WAVEX"),
    ],
)
def test_read_wav_decodes_common_formats(subtype, wav_format):
    tone = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, SAMPLE_RATE_HZ))
    samples = read_wav(_wav_bytes(tone, SAMPLE_RATE_HZ, subtype, wav_format))
    assert samples.dtype == np.float32
    assert samples.size == SAMPLE_RATE_HZ
    assert samples == pytest.approx(tone, abs=1e-3)


def test_read_wav_gives_16k_mono_from_48k_stereo():
    tone = 0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, 48_000))
    stereo = np.column_stack([tone, tone])
    samples = read_wav(_wav_bytes(stereo, 48_000, "FLOAT"))
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert samples.size == pytest.approx(SAMPLE_RATE_HZ, rel=0.01)
    assert np.abs(samples).max() == pytest.approx(0.5, abs=0.01)


def test_evaluate_warms_up_once_then_times_every_clip():
    calls = []

    def fake_transcribe(samples: np.ndarray) -> str:
        calls.append(samples.size)
        return "բարև"

    clips = [_clip("1"), _clip("2")]
    results = evaluate(fake_transcribe, clips)
    assert len(calls) == 3
    assert [r.clip.clip_id for r in results] == ["1", "2"]
    assert all(r.elapsed_s >= 0 for r in results)


def test_summarize_reports_accuracy_and_speed():
    results = [
        ClipResult(_clip("1", reference="բարև ձեզ"), "բարև ձեզ", 1.0),
        ClipResult(_clip("2", reference="լավ եմ"), "Lav em", ASR_BUDGET_S + 1),
    ]
    summary = summarize("model", results)
    assert summary.wer == pytest.approx(0.5)
    assert summary.wrong_script_count == 1
    assert summary.over_budget_count == 1
    assert summary.max_s == ASR_BUDGET_S + 1
    assert summary.real_time_factor == pytest.approx((1.0 + ASR_BUDGET_S + 1) / 8.0)


def test_format_table_has_one_row_per_model():
    results = [ClipResult(_clip(), "բարև", 0.5)]
    table = format_table([summarize("small", results), summarize("large", results)])
    assert table.count("\n") == 3
    assert "| small |" in table


class FakeReply(io.BytesIO):
    """Stands in for the response of urllib.request.urlopen."""


@pytest.fixture
def no_env_token(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_NAME, raising=False)


def test_token_is_read_from_the_env_file(tmp_path, no_env_token):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{HF_TOKEN_NAME}=hf_from_file\n", encoding="utf-8")
    assert load_hf_token(env_file) == "hf_from_file"


def test_token_already_in_the_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv(HF_TOKEN_NAME, "hf_from_environment")
    env_file = tmp_path / ".env"
    env_file.write_text(f"{HF_TOKEN_NAME}=hf_from_file\n", encoding="utf-8")
    assert load_hf_token(env_file) == "hf_from_environment"


@pytest.mark.parametrize("env_text", [None, f"{HF_TOKEN_NAME}=\n"])
def test_missing_or_empty_token_gives_none(tmp_path, no_env_token, env_text):
    env_file = tmp_path / ".env"
    if env_text is not None:
        env_file.write_text(env_text, encoding="utf-8")
    assert load_hf_token(env_file) is None


@pytest.mark.parametrize(
    ("token", "expected_header"), [("hf_secret", "Bearer hf_secret"), (None, None)]
)
def test_download_sends_the_token_only_when_set(
    tmp_path, monkeypatch, token, expected_header
):
    requests = []

    def fake_urlopen(request):
        requests.append(request)
        return FakeReply(b"data")

    monkeypatch.setattr(asr_benchmark.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "test.tsv"
    download("https://example.org/test.tsv", target, token)
    assert requests[0].get_header("Authorization") == expected_header
    assert target.read_bytes() == b"data"
    assert list(tmp_path.iterdir()) == [target]

"""Tests for converting Armenian ASR fine-tunes (PLAN.md §6 risk 1: model choice)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ctranslate2.converters
import pytest

import convert_asr
from convert_asr import COPY_FILES, PROMPT_TOKENS, TOKENIZER_FILE, convert, main

STANDARD_TOKENS = {token: index for index, token in enumerate(PROMPT_TOKENS)}


VOCABULARY = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|hy|>"]
ARMENIAN_ID = VOCABULARY.index("<|hy|>")


class FakeConverter:
    """Stands in for TransformersConverter: writes the files it would write."""

    calls: list[dict] = []
    lang_ids: list[int] = []

    def __init__(self, hub_id: str, **options) -> None:
        FakeConverter.calls.append({"hub_id": hub_id, **options})

    def convert(self, output_dir: str, **options) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True)
        (output / "model.bin").write_bytes(b"weights")
        config = {"lang_ids": FakeConverter.lang_ids}
        (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output / "vocabulary.json").write_text(
            json.dumps(VOCABULARY), encoding="utf-8"
        )


class FakeBackendTokenizer:
    def __init__(self, tokens: dict[str, int]) -> None:
        self.tokens = tokens

    def token_to_id(self, token: str) -> int | None:
        return self.tokens.get(token)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.tokens), encoding="utf-8")


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    FakeConverter.calls = []
    FakeConverter.lang_ids = [VOCABULARY.index("<|en|>"), ARMENIAN_ID]
    tokens = dict(STANDARD_TOKENS)
    auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda hub_id: SimpleNamespace(
            backend_tokenizer=FakeBackendTokenizer(tokens)
        )
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto_tokenizer)
    )
    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", FakeConverter)
    monkeypatch.setattr(convert_asr, "load_hf_token", lambda: None)
    monkeypatch.setattr(convert_asr, "ASR_MODELS_DIR", tmp_path)
    return SimpleNamespace(models_dir=tmp_path, tokens=tokens)


def test_fine_tune_is_converted_with_what_faster_whisper_needs(fakes):
    model_dir = convert("whisper-turbo-armenian", fakes.models_dir)
    assert model_dir == fakes.models_dir / "whisper-turbo-armenian"
    assert (model_dir / "model.bin").is_file()
    assert (model_dir / TOKENIZER_FILE).is_file()
    [call] = FakeConverter.calls
    assert call["hub_id"] == "Chillarmo/whisper-large-v3-turbo-armenian"
    # Without it, large-v3 models get 80 mel bands instead of 128.
    assert "preprocessor_config.json" in call["copy_files"] == COPY_FILES


def test_tokenizer_without_the_armenian_prompt_fails_and_leaves_nothing(fakes, capsys):
    del fakes.tokens["<|hy|>"]
    assert main(["whisper-turbo-armenian"]) == 1
    assert "lacks prompt tokens: <|hy|>" in capsys.readouterr().out
    assert list(fakes.models_dir.iterdir()) == []


def test_model_without_language_tokens_is_rejected_as_english_only(fakes, capsys):
    # What happened with BuzzASR/armenian: its tokenizer has its own token order.
    FakeConverter.lang_ids = []
    assert main(["whisper-turbo-armenian"]) == 1
    assert "would run it as English-only" in capsys.readouterr().out
    assert list(fakes.models_dir.iterdir()) == []


def test_converted_model_is_not_converted_again(fakes):
    (fakes.models_dir / "whisper-turbo-armenian").mkdir()
    assert main([]) == 0
    assert FakeConverter.calls == []

"""Tests for converting the MT candidates (PLAN.md §6 risk 12: models ready offline)."""

from pathlib import Path

import ctranslate2.converters
import pytest

import convert_mt
from convert_mt import convert, main, parse_args, tokenizer_files
from live_hy_en.translator import MT_MODELS


class FakeConverter:
    """Stands in for TransformersConverter; writes a marker file, or fails."""

    calls: list[dict] = []
    failing_hub_ids: set[str] = set()

    def __init__(self, hub_id: str, **options) -> None:
        self.hub_id = hub_id
        FakeConverter.calls.append({"hub_id": hub_id, **options})

    def convert(self, output_dir: str, **options) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True)
        (output / "model.bin").write_bytes(b"weights")
        if self.hub_id in FakeConverter.failing_hub_ids:
            raise RuntimeError("Repository not found")


@pytest.fixture
def fake_converter(monkeypatch, tmp_path):
    FakeConverter.calls = []
    FakeConverter.failing_hub_ids = set()
    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", FakeConverter)
    monkeypatch.setattr(convert_mt, "MT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(convert_mt, "load_hf_token", lambda: None)
    return tmp_path


def test_tokenizer_files_are_copied_with_the_weights():
    assert tokenizer_files(MT_MODELS["nllb-200-distilled-600M"]) == [
        "sentencepiece.bpe.model"
    ]
    assert tokenizer_files(MT_MODELS["opus-mt-hy-en"]) == ["source.spm", "target.spm"]


def test_model_is_converted_into_its_own_folder(fake_converter):
    output_dir = convert("opus-mt-hy-en", fake_converter)
    assert output_dir == fake_converter / "opus-mt-hy-en"
    assert (output_dir / "model.bin").is_file()
    assert not (fake_converter / "opus-mt-hy-en.part").exists()
    [call] = FakeConverter.calls
    assert call["hub_id"] == "Helsinki-NLP/opus-mt-hy-en"
    assert call["copy_files"] == ["source.spm", "target.spm"]


def test_converted_model_is_not_converted_again(fake_converter):
    (fake_converter / "opus-mt-hy-en").mkdir()
    convert("opus-mt-hy-en", fake_converter)
    assert FakeConverter.calls == []


def test_all_candidates_are_converted_by_default(fake_converter):
    assert main([]) == 0
    assert sorted(path.name for path in fake_converter.iterdir()) == sorted(MT_MODELS)


def test_one_failure_does_not_stop_the_others(fake_converter, capsys):
    FakeConverter.failing_hub_ids = {"Helsinki-NLP/opus-mt-hy-en"}
    assert main(["opus-mt-hy-en", "m2m100_1.2B"]) == 1
    assert (fake_converter / "m2m100_1.2B").is_dir()
    assert not (fake_converter / "opus-mt-hy-en").exists()
    assert not (fake_converter / "opus-mt-hy-en.part").exists()
    assert "opus-mt-hy-en: conversion failed: Repository not found" in (
        capsys.readouterr().out
    )


def test_unknown_model_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["google-translate"])

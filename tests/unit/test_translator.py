"""Tests for translation (PLAN.md §3 module 6, §5.1: each segment gets an EN line)."""

from types import SimpleNamespace

import ctranslate2
import pytest
import sentencepiece

from live_hy_en.translator import (
    CPU_COMPUTE_TYPE,
    END_OF_TEXT,
    GPU_COMPUTE_TYPE,
    MT_MODEL,
    MT_MODELS,
    WARM_UP_TEXT,
    CTranslate2Translator,
    ModelLoadError,
    Translator,
)


class FakeTextTranslator:
    """Returns a fixed translation and records the text it was given."""

    def __init__(self, english: str = "Hi, how are you?") -> None:
        self.english = english
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return self.english


def test_armenian_text_becomes_english_text():
    model = FakeTextTranslator("Hi, how are you?")
    assert Translator(model).translate("Բարև, ինչպե՞ս ես։") == "Hi, how are you?"
    assert model.calls == ["Բարև, ինչպե՞ս ես։"]


def test_extra_whitespace_is_removed():
    translator = Translator(FakeTextTranslator("  I'm   fine,\n thank you. "))
    assert translator.translate("Լավ եմ") == "I'm fine, thank you."


def test_empty_translation_is_reported(caplog):
    assert Translator(FakeTextTranslator("  ")).translate("Լավ եմ") == ""
    assert "no output" in caplog.text


def test_warm_up_translates_armenian_once():
    model = FakeTextTranslator()
    Translator(model).warm_up()
    assert model.calls == [WARM_UP_TEXT]


def test_warm_up_failure_gives_clear_error():
    class BrokenTextTranslator:
        def translate(self, text: str) -> str:
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(ModelLoadError, match="first run.*CUDA out of memory"):
        Translator(BrokenTextTranslator()).warm_up()


class FakeSentencePiece:
    """Splits on spaces, and joins pieces back, recording which file it loaded."""

    def __init__(self, model_file: str) -> None:
        self.model_file = model_file

    def encode(self, text: str, out_type=str) -> list[str]:
        return text.split()

    def decode(self, pieces: list[str]) -> str:
        return " ".join(pieces)


class FakeCt2Translator:
    """Stands in for ctranslate2.Translator; echoes its hypothesis tokens."""

    instances: list["FakeCt2Translator"] = []
    hypothesis: list[str] = []

    def __init__(self, model_path: str, **options) -> None:
        self.model_path = model_path
        self.options = options
        self.calls: list[tuple] = []
        FakeCt2Translator.instances.append(self)

    def translate_batch(self, source, **options):
        self.calls.append((source, options))
        return [SimpleNamespace(hypotheses=[list(FakeCt2Translator.hypothesis)])]


@pytest.fixture
def fake_ct2(monkeypatch, tmp_path):
    FakeCt2Translator.instances = []
    FakeCt2Translator.hypothesis = ["Hi", "there"]
    monkeypatch.setattr(ctranslate2, "Translator", FakeCt2Translator)
    monkeypatch.setattr(sentencepiece, "SentencePieceProcessor", FakeSentencePiece)
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    for name in MT_MODELS:
        (tmp_path / name).mkdir()
    return tmp_path


def _load(models_dir, model: str = MT_MODEL, **options) -> CTranslate2Translator:
    return CTranslate2Translator(model, models_dir=models_dir, **options)


def test_nllb_is_told_armenian_in_and_english_out(fake_ct2):
    FakeCt2Translator.hypothesis = ["eng_Latn", "Hi", "there"]
    translator = _load(fake_ct2, "nllb-200-distilled-600M")
    assert translator.translate("Բարև ձեզ") == "Hi there"
    [(source, options)] = FakeCt2Translator.instances[0].calls
    assert source == [["hye_Armn", "Բարև", "ձեզ", END_OF_TEXT]]
    assert options["target_prefix"] == [["eng_Latn"]]


def test_m2m100_is_told_armenian_in_and_english_out(fake_ct2):
    FakeCt2Translator.hypothesis = ["__en__", "Hi"]
    translator = _load(fake_ct2, "m2m100_1.2B")
    assert translator.translate("Բարև") == "Hi"
    [(source, options)] = FakeCt2Translator.instances[0].calls
    assert source == [["__hy__", "Բարև", END_OF_TEXT]]
    assert options["target_prefix"] == [["__en__"]]


def test_madlad_gets_the_english_tag_in_its_text(fake_ct2):
    translator = _load(fake_ct2, "madlad400-3b-mt")
    assert translator.translate("Բարև") == "Hi there"
    [(source, options)] = FakeCt2Translator.instances[0].calls
    assert source == [["<2en>", "Բարև", END_OF_TEXT]]
    assert options["target_prefix"] is None


def test_opus_mt_decodes_with_its_target_vocabulary(fake_ct2):
    translator = _load(fake_ct2, "opus-mt-hy-en")
    assert translator.translate("Բարև") == "Hi there"
    assert translator._source_spm.model_file.endswith("source.spm")
    assert translator._target_spm.model_file.endswith("target.spm")


def test_gpu_is_used_when_present(fake_ct2):
    _load(fake_ct2)
    options = FakeCt2Translator.instances[0].options
    assert options == {"device": "cuda", "compute_type": GPU_COMPUTE_TYPE}


def test_cpu_is_the_fallback_without_a_gpu(fake_ct2, monkeypatch, caplog):
    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    _load(fake_ct2)
    options = FakeCt2Translator.instances[0].options
    assert options == {"device": "cpu", "compute_type": CPU_COMPUTE_TYPE}
    assert "No CUDA GPU" in caplog.text


def test_model_is_loaded_from_its_own_folder(fake_ct2):
    _load(fake_ct2, "opus-mt-hy-en")
    assert FakeCt2Translator.instances[0].model_path == str(fake_ct2 / "opus-mt-hy-en")


def test_unknown_model_gives_clear_error(fake_ct2):
    with pytest.raises(ModelLoadError, match="Unknown translation model.*nllb"):
        _load(fake_ct2, "google-translate")


def test_unconverted_model_says_how_to_convert_it(tmp_path):
    with pytest.raises(ModelLoadError, match="convert_mt.py nllb-200-distilled-600M"):
        _load(tmp_path, "nllb-200-distilled-600M")


def test_load_failure_gives_clear_error(fake_ct2, monkeypatch):
    def failing_model(model_path, **options):
        raise RuntimeError("Unsupported model binary version")

    monkeypatch.setattr(ctranslate2, "Translator", failing_model)
    with pytest.raises(ModelLoadError, match="nllb.*on cuda.*binary version"):
        _load(fake_ct2)

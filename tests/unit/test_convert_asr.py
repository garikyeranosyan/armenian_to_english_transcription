"""Tests for converting Armenian ASR fine-tunes (PLAN.md §6 risk 1: model choice)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import ctranslate2.converters
import pytest
from faster_whisper.tokenizer import Tokenizer as WhisperTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE

import convert_asr
from convert_asr import (
    ASR_CANDIDATES,
    COPY_FILES,
    TOKENIZER_FILE,
    VOCABULARY_FILE,
    convert,
    main,
)

# Like OpenAI's multilingual vocabularies, ending with the empty token.
TEXT_TOKENS = ["Ġ", "Ա", "Բ", ""]
# Whisper's special tokens in their usual order, shortened to two languages.
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|hy|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nospeech|>",
    "<|notimestamps|>",
    "<|0.00|>",
    "<|0.02|>",
]
PROMPT = ["<|startoftranscript|>", "<|hy|>", "<|transcribe|>"]


def trained_ids(first_id: int) -> dict[str, int]:
    """Where the model was trained to see each special token, in Whisper's order."""
    return {token: first_id + i for i, token in enumerate(SPECIAL_TOKENS)}


def generation_config(ids: dict[str, int]) -> dict:
    """The decoding settings a model ships with, using its trained IDs."""
    return {
        "decoder_start_token_id": ids["<|startoftranscript|>"],
        "eos_token_id": ids["<|endoftext|>"],
        "no_timestamps_token_id": ids["<|notimestamps|>"],
        "lang_to_id": {"<|hy|>": ids["<|hy|>"]},
    }


# Standard Whisper: the special tokens follow the text tokens.
STANDARD_IDS = trained_ids(len(TEXT_TOKENS))
STANDARD_SIZE = len(TEXT_TOKENS) + len(SPECIAL_TOKENS)
# BuzzASR: they follow a vocabulary that also lists them first.
SPECIAL_FIRST_IDS = trained_ids(len(SPECIAL_TOKENS) + len(TEXT_TOKENS) - 1)
SPECIAL_FIRST_SIZE = 2 * len(SPECIAL_TOKENS) + len(TEXT_TOKENS) - 1


def standard_tokenizer() -> Tokenizer:
    """A Whisper-style tokenizer: text tokens first, then the special tokens."""
    tokenizer = Tokenizer(BPE({t: i for i, t in enumerate(TEXT_TOKENS)}, merges=[]))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


def special_first_tokenizer() -> Tokenizer:
    """BuzzASR's layout: its vocabulary lists the special tokens first and has
    no empty token."""
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + TEXT_TOKENS[:-1])}
    tokenizer = Tokenizer(BPE(vocab, merges=[]))
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


class FakeConverter:
    """Stands in for TransformersConverter: writes the files it would write."""

    calls: list[dict] = []
    model_size = STANDARD_SIZE

    def __init__(self, hub_id: str, **options) -> None:
        FakeConverter.calls.append({"hub_id": hub_id, **options})

    def convert(self, output_dir: str, **options) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True)
        (output / "model.bin").write_bytes(b"weights")
        # The converter copies standard Whisper IDs and finds no languages.
        config = {"lang_ids": [], "suppress_ids_begin": [220, 50257]}
        (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
        vocabulary = [f"token{i}" for i in range(FakeConverter.model_size)]
        (output / VOCABULARY_FILE).write_text(json.dumps(vocabulary), encoding="utf-8")


def use_layout(state: SimpleNamespace, tokenizer: Tokenizer, ids: dict, size: int):
    """Make the fake hub serve a model with this tokenizer and these trained IDs."""
    state.tokenizer = tokenizer
    state.hub_files = {
        "added_tokens.json": ids,
        "generation_config.json": generation_config(ids),
    }
    FakeConverter.model_size = size


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    FakeConverter.calls = []
    state = SimpleNamespace(models_dir=tmp_path)
    use_layout(state, standard_tokenizer(), STANDARD_IDS, STANDARD_SIZE)
    auto_tokenizer = SimpleNamespace(
        from_pretrained=lambda hub_id: SimpleNamespace(
            backend_tokenizer=state.tokenizer
        )
    )
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto_tokenizer)
    )
    monkeypatch.setattr(ctranslate2.converters, "TransformersConverter", FakeConverter)
    monkeypatch.setattr(
        convert_asr, "read_hub_json", lambda hub_id, name: state.hub_files.get(name, {})
    )
    monkeypatch.setattr(convert_asr, "load_hf_token", lambda: None)
    monkeypatch.setattr(convert_asr, "ASR_MODELS_DIR", tmp_path)
    return state


def load_converted(model_dir: Path) -> tuple[WhisperTokenizer, dict, list[str]]:
    """Load the tokenizer as faster-whisper does, plus CTranslate2's tables."""
    tokenizer = WhisperTokenizer(
        Tokenizer.from_file(str(model_dir / TOKENIZER_FILE)),
        True,
        task="transcribe",
        language="hy",
    )
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((model_dir / VOCABULARY_FILE).read_text(encoding="utf-8"))
    return tokenizer, config, vocabulary


def test_both_armenian_fine_tunes_are_candidates():
    assert ASR_CANDIDATES == {
        "whisper-turbo-armenian": "Chillarmo/whisper-large-v3-turbo-armenian",
        "buzzasr-armenian": "BuzzASR/armenian",
    }


def test_fine_tune_is_converted_with_what_faster_whisper_needs(fakes):
    model_dir = convert("whisper-turbo-armenian", fakes.models_dir)
    assert model_dir == fakes.models_dir / "whisper-turbo-armenian"
    assert (model_dir / "model.bin").is_file()
    [call] = FakeConverter.calls
    assert call["hub_id"] == "Chillarmo/whisper-large-v3-turbo-armenian"
    # Without it, large-v3 models get 80 mel bands instead of 128.
    assert "preprocessor_config.json" in call["copy_files"] == COPY_FILES
    tokenizer, config, vocabulary = load_converted(model_dir)
    assert list(tokenizer.sot_sequence) == [STANDARD_IDS[token] for token in PROMPT]
    assert config["lang_ids"] == [STANDARD_IDS["<|en|>"], STANDARD_IDS["<|hy|>"]]
    assert config["suppress_ids_begin"] == [0, STANDARD_IDS["<|endoftext|>"]]
    assert vocabulary == TEXT_TOKENS + SPECIAL_TOKENS


def test_special_tokens_listed_first_get_the_ids_the_model_was_trained_with(fakes):
    # What broke BuzzASR/armenian: faster-whisper ran it as English-only.
    use_layout(fakes, special_first_tokenizer(), SPECIAL_FIRST_IDS, SPECIAL_FIRST_SIZE)
    model_dir = convert("buzzasr-armenian", fakes.models_dir)
    tokenizer, config, vocabulary = load_converted(model_dir)
    ids = SPECIAL_FIRST_IDS
    assert list(tokenizer.sot_sequence) == [ids[token] for token in PROMPT]
    assert tokenizer.eot == ids["<|endoftext|>"]
    assert tokenizer.timestamp_begin == ids["<|0.00|>"]
    assert ids["<|hy|>"] in config["lang_ids"]
    assert vocabulary[ids["<|hy|>"]] == "<|hy|>"
    # The old copies are renamed and never printed as text; the first one marks
    # the model as multilingual for CTranslate2.
    assert vocabulary[0] == ""
    assert vocabulary[1] == "<|unused_1|>"
    text_ids = tokenizer.encode("ԱԲ")
    assert tokenizer.decode([1, *text_ids, tokenizer.eot]) == tokenizer.decode(text_ids)


def test_tokenizer_disagreeing_with_the_generation_config_is_rejected(fakes, capsys):
    fakes.hub_files["generation_config.json"]["decoder_start_token_id"] = 1
    assert main(["whisper-turbo-armenian"]) == 1
    assert "decoder_start_token_id is 1" in capsys.readouterr().out
    assert list(fakes.models_dir.iterdir()) == []


def test_tokenizer_without_the_armenian_prompt_fails_and_leaves_nothing(fakes, capsys):
    tokenizer = Tokenizer(BPE({t: i for i, t in enumerate(TEXT_TOKENS)}, merges=[]))
    tokenizer.add_special_tokens([t for t in SPECIAL_TOKENS if t != "<|hy|>"])
    fakes.tokenizer = tokenizer
    fakes.hub_files = {}
    assert main(["whisper-turbo-armenian"]) == 1
    assert "lacks prompt tokens: <|hy|>" in capsys.readouterr().out
    assert list(fakes.models_dir.iterdir()) == []


def test_converted_model_is_not_converted_again(fakes):
    for name in ASR_CANDIDATES:
        (fakes.models_dir / name).mkdir()
    assert main([]) == 0
    assert FakeConverter.calls == []


def test_vocabulary_without_the_multilingual_marker_is_rejected(fakes, capsys):
    tokenizer = Tokenizer(
        BPE({t: i for i, t in enumerate(TEXT_TOKENS[:-1])}, merges=[])
    )
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    use_layout(fakes, tokenizer, trained_ids(len(TEXT_TOKENS) - 1), STANDARD_SIZE - 1)
    assert main(["whisper-turbo-armenian"]) == 1
    assert "CTranslate2 would run it as English-only" in capsys.readouterr().out
    assert list(fakes.models_dir.iterdir()) == []

"""Convert Armenian Whisper fine-tunes to CTranslate2 once, into models/asr/.

Run from the project folder, then benchmark the folder like any other model:
    python benchmarks/convert_asr.py
    python benchmarks/asr_benchmark.py --models models/asr/whisper-turbo-armenian
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Run as a script, so the program's own package isn't on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_benchmark import load_hf_token  # noqa: E402
from convert_mt import convert_hub_model  # noqa: E402
from live_hy_en.transcriber import ASR_MODELS_DIR  # noqa: E402

# Folder name -> Hugging Face model ID.
ASR_CANDIDATES = {
    "whisper-turbo-armenian": "Chillarmo/whisper-large-v3-turbo-armenian",
    "buzzasr-armenian": "BuzzASR/armenian",
}
# faster-whisper reads the mel band count (128 for large-v3) from this file and
# would assume 80 without it.
COPY_FILES = ["preprocessor_config.json"]
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"
VOCABULARY_FILE = "vocabulary.json"
# The special token IDs the model was trained with, and its decoding settings.
ADDED_TOKENS_FILE = "added_tokens.json"
GENERATION_CONFIG_FILE = "generation_config.json"
END_TOKEN = "<|endoftext|>"
START_TOKEN = "<|startoftranscript|>"
ARMENIAN_TOKEN = "<|hy|>"
TRANSLATE_TOKEN = "<|translate|>"
TRANSCRIBE_TOKEN = "<|transcribe|>"
NO_TIMESTAMPS_TOKEN = "<|notimestamps|>"
FIRST_TIMESTAMP_TOKEN = "<|0.00|>"
# Byte-level BPE's space, which Whisper may not emit as the first token.
SPACE_TOKEN = "Ġ"
# Name given to a vocabulary entry whose special-token name the model uses at
# another ID; it stays special, so it is never printed.
UNUSED_TOKEN = "<|unused_{}|>"
# CTranslate2 counts a Whisper model as multilingual only if its vocabulary has
# this empty token, which OpenAI's multilingual vocabularies keep before the end
# token; without it, faster-whisper replaces Armenian with English.
MULTILINGUAL_MARKER = ""
# Tokens faster-whisper puts in the prompt for Armenian transcription.
PROMPT_TOKENS = [START_TOKEN, ARMENIAN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]


def special_token(token_id: int, content: str) -> dict:
    """Return a tokenizer.json added-token entry for a special token."""
    return {
        "id": token_id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def apply_trained_ids(layout: dict, trained_ids: dict[str, int]) -> dict:
    """Give each special token in a tokenizer.json layout its trained ID.

    BuzzASR's Armenian vocabulary also lists Whisper's special tokens at IDs
    0-1608, which win over its added_tokens.json, while the model was trained
    with them after the text tokens, in Whisper's usual order. The freed IDs are
    never generated; the first becomes the multilingual marker if it is missing.
    """
    vocab = layout["model"]["vocab"]
    renamed = []
    for token, token_id in list(vocab.items()):
        if token in trained_ids and token_id != trained_ids[token]:
            if MULTILINGUAL_MARKER not in vocab:
                vocab[MULTILINGUAL_MARKER] = vocab.pop(token)
                continue
            name = UNUSED_TOKEN.format(token_id)
            vocab[name] = vocab.pop(token)
            renamed.append(special_token(token_id, name))
    kept = [
        entry for entry in layout["added_tokens"] if entry["content"] not in trained_ids
    ]
    trained = [
        special_token(token_id, token) for token, token_id in trained_ids.items()
    ]
    taken = {entry["id"]: entry["content"] for entry in kept}
    taken.update({token_id: token for token, token_id in vocab.items()})
    clashes = sorted(
        token for token, token_id in trained_ids.items() if token_id in taken
    )
    if clashes:
        raise ValueError(
            f"trained IDs of {', '.join(clashes)} are used by other tokens"
        )
    layout["added_tokens"] = sorted(kept + renamed + trained, key=lambda e: e["id"])
    # The template that adds start and end tokens must use the same IDs.
    ids = {entry["content"]: entry["id"] for entry in layout["added_tokens"]}
    post_processor = layout.get("post_processor") or {}
    for special in post_processor.get("special_tokens", {}).values():
        special["ids"] = [ids[token] for token in special["tokens"]]
    return layout


def read_hub_json(hub_id: str, filename: str) -> dict:
    """Return a JSON file from the model's Hugging Face repository, or {} if absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(hub_id, filename)
    except EntryNotFoundError:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_tokenizer(hub_id: str, model_dir: Path):
    """Write the fast tokenizer file faster-whisper loads, with trained token IDs."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    # Some fine-tunes ship only vocab.json and merges.txt; transformers builds
    # the same tokenizer from them, which can then be saved as one file.
    backend = AutoTokenizer.from_pretrained(hub_id).backend_tokenizer
    layout = apply_trained_ids(
        json.loads(backend.to_str()), read_hub_json(hub_id, ADDED_TOKENS_FILE)
    )
    tokenizer = Tokenizer.from_str(json.dumps(layout))
    missing = [token for token in PROMPT_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer lacks prompt tokens: {', '.join(missing)}")
    tokenizer.save(str(model_dir / TOKENIZER_FILE))
    return tokenizer


def write_vocabulary_and_config(tokenizer, model_dir: Path) -> None:
    """Rewrite CTranslate2's token list, languages and first-token rule to match.

    The converter names tokens and finds languages from transformers' IDs, and
    copies the first-token rule as standard Whisper IDs.
    """
    vocabulary_path = model_dir / VOCABULARY_FILE
    size = len(json.loads(vocabulary_path.read_text(encoding="utf-8")))
    vocabulary = [tokenizer.id_to_token(token_id) for token_id in range(size)]
    if None in vocabulary:
        raise ValueError(f"tokenizer has no token for ID {vocabulary.index(None)}")
    vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
    config_path = model_dir / CONFIG_FILE
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Whisper's language tokens sit between the start and translate tokens.
    config["lang_ids"] = list(
        range(
            tokenizer.token_to_id(START_TOKEN) + 1,
            tokenizer.token_to_id(TRANSLATE_TOKEN),
        )
    )
    config["suppress_ids_begin"] = [
        tokenizer.token_to_id(SPACE_TOKEN),
        tokenizer.token_to_id(END_TOKEN),
    ]
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def check_layout(tokenizer, model_dir: Path, generation: dict) -> None:
    """Fail unless faster-whisper will prompt and read the model correctly.

    faster-whisper treats IDs below the end token as text, takes the first
    timestamp as the token after "no timestamps", and runs a model as
    English-only when it has no language IDs or no multilingual marker.
    """
    token_id = tokenizer.token_to_id
    vocab = tokenizer.get_vocab(with_added_tokens=False)
    end_id = token_id(END_TOKEN)
    problems = []
    if any(index >= end_id for index in vocab.values()):
        problems.append("text tokens come after the end token")
    if MULTILINGUAL_MARKER not in vocab:
        problems.append("no empty token, so CTranslate2 would run it as English-only")
    if token_id(FIRST_TIMESTAMP_TOKEN) != token_id(NO_TIMESTAMPS_TOKEN) + 1:
        problems.append("timestamps don't follow the no-timestamps token")
    config = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    if token_id(ARMENIAN_TOKEN) not in config["lang_ids"]:
        problems.append(
            "Armenian is not among its languages, so it would run as English-only"
        )
    trained = {
        "decoder_start_token_id": token_id(START_TOKEN),
        "eos_token_id": end_id,
        "no_timestamps_token_id": token_id(NO_TIMESTAMPS_TOKEN),
    }
    for key, expected in trained.items():
        if key in generation and generation[key] != expected:
            problems.append(f"{key} is {generation[key]}, tokenizer says {expected}")
    armenian_id = generation.get("lang_to_id", {}).get(ARMENIAN_TOKEN)
    if armenian_id not in (None, token_id(ARMENIAN_TOKEN)):
        problems.append(f"{ARMENIAN_TOKEN} was trained as {armenian_id}")
    if problems:
        raise ValueError("converted model would not work: " + "; ".join(problems))


def prepare_for_faster_whisper(hub_id: str, model_dir: Path) -> None:
    """Add the tokenizer file, match the token tables to it, and check them."""
    tokenizer = write_tokenizer(hub_id, model_dir)
    write_vocabulary_and_config(tokenizer, model_dir)
    check_layout(tokenizer, model_dir, read_hub_json(hub_id, GENERATION_CONFIG_FILE))


def convert(name: str, models_dir: Path = ASR_MODELS_DIR) -> Path:
    """Convert one fine-tune, unless it is already there; return its folder."""
    hub_id = ASR_CANDIDATES[name]
    return convert_hub_model(
        hub_id,
        models_dir / name,
        COPY_FILES,
        add_files=lambda model_dir: prepare_for_faster_whisper(hub_id, model_dir),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Convert every candidate; return 1 if any of them failed."""
    names = list(argv) if argv else list(ASR_CANDIDATES)
    # Puts HF_TOKEN from .env into the environment, where the Hub client reads it.
    load_hf_token()
    failed = []
    for name in names:
        try:
            convert(name, ASR_MODELS_DIR)
        except Exception as error:
            print(f"{name}: conversion failed: {error}")
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

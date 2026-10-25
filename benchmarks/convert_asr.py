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

# Folder name -> Hugging Face model ID. Candidates must keep Whisper's standard
# tokenizer layout: BuzzASR/armenian, for one, has its own token order, so it
# converts without language tokens and faster-whisper runs it as English-only.
ASR_CANDIDATES = {
    "whisper-turbo-armenian": "Chillarmo/whisper-large-v3-turbo-armenian",
}
# faster-whisper reads the mel band count (128 for large-v3) from this file and
# would assume 80 without it.
COPY_FILES = ["preprocessor_config.json"]
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "config.json"
VOCABULARY_FILE = "vocabulary.json"
ARMENIAN_TOKEN = "<|hy|>"
# Tokens faster-whisper puts in the prompt for Armenian transcription.
PROMPT_TOKENS = [
    "<|startoftranscript|>",
    "<|hy|>",
    "<|transcribe|>",
    "<|notimestamps|>",
]


def write_tokenizer(hub_id: str, model_dir: Path) -> None:
    """Write the fast tokenizer file faster-whisper loads, and check its prompt."""
    from transformers import AutoTokenizer

    # Some fine-tunes ship only vocab.json and merges.txt; transformers builds
    # the same tokenizer from them, which can then be saved as one file.
    backend = AutoTokenizer.from_pretrained(hub_id).backend_tokenizer
    missing = [token for token in PROMPT_TOKENS if backend.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer lacks prompt tokens: {', '.join(missing)}")
    backend.save(str(model_dir / TOKENIZER_FILE))


def check_armenian_is_supported(model_dir: Path) -> None:
    """Fail unless the converted model lists Armenian among its languages.

    faster-whisper treats a model without language tokens as English-only and
    replaces the requested language with English.
    """
    config = json.loads((model_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    vocabulary = json.loads((model_dir / VOCABULARY_FILE).read_text(encoding="utf-8"))
    armenian_id = (
        vocabulary.index(ARMENIAN_TOKEN) if ARMENIAN_TOKEN in vocabulary else None
    )
    if armenian_id not in config.get("lang_ids", []):
        raise ValueError(
            "converted model has no Armenian language token, so faster-whisper "
            "would run it as English-only"
        )


def prepare_for_faster_whisper(hub_id: str, model_dir: Path) -> None:
    """Add the tokenizer file and check the model will transcribe Armenian."""
    write_tokenizer(hub_id, model_dir)
    check_armenian_is_supported(model_dir)


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

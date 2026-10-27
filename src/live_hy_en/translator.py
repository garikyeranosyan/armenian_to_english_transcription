"""Armenian text to English text, behind an interface any MT model can fill."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from live_hy_en.transcriber import ModelLoadError, add_cuda_dll_dirs, has_cuda_gpu

BEAM_SIZE = 4
# 8-bit weights, so it fits in GPU memory beside the speech model.
GPU_COMPUTE_TYPE = "int8_float16"
CPU_COMPUTE_TYPE = "int8"
END_OF_TEXT = "</s>"
WARM_UP_TEXT = "Բարև, ինչպե՞ս ես։"
# Converted models live here, one folder each (benchmarks/convert_mt.py).
MT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "mt"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptFormat:
    """How one model family marks the source and target language."""

    source_spm: str
    target_spm: str
    # Tokens before the source text, e.g. NLLB's language code.
    source_prefix: tuple[str, ...] = ()
    # Text before the source text, e.g. MADLAD's target-language tag.
    text_prefix: str = ""
    # Tokens the translation must start with; removed from the output.
    target_prefix: tuple[str, ...] = ()


NLLB_FORMAT = PromptFormat(
    "sentencepiece.bpe.model",
    "sentencepiece.bpe.model",
    source_prefix=("hye_Armn",),
    target_prefix=("eng_Latn",),
)
M2M100_FORMAT = PromptFormat(
    "sentencepiece.bpe.model",
    "sentencepiece.bpe.model",
    source_prefix=("__hy__",),
    target_prefix=("__en__",),
)
MADLAD_FORMAT = PromptFormat("spiece.model", "spiece.model", text_prefix="<2en> ")
OPUS_MT_FORMAT = PromptFormat("source.spm", "target.spm")


@dataclass(frozen=True)
class MtModel:
    """A translation candidate: where it is published and how to prompt it."""

    hub_id: str
    prompt_format: PromptFormat


# The candidates from PLAN.md §3, keyed by their folder name in MT_MODELS_DIR.
MT_MODELS = {
    "nllb-200-distilled-600M": MtModel("facebook/nllb-200-distilled-600M", NLLB_FORMAT),
    "nllb-200-distilled-1.3B": MtModel("facebook/nllb-200-distilled-1.3B", NLLB_FORMAT),
    "m2m100_1.2B": MtModel("facebook/m2m100_1.2B", M2M100_FORMAT),
    "madlad400-3b-mt": MtModel("google/madlad400-3b-mt", MADLAD_FORMAT),
    "opus-mt-hy-en": MtModel("Helsinki-NLP/opus-mt-hy-en", OPUS_MT_FORMAT),
}
# Chosen by benchmarks/mt_benchmark.py: the best chrF++ on FLEURS, both from
# correct Armenian (63.2) and from our ASR transcripts (55.0). At p95 0.87 s it
# overruns the 0.5 s translation share, accepted because whole phrases still
# finish far inside the 5 s limit (e2e replay max 2.18 s).
MT_MODEL = "nllb-200-distilled-1.3B"


class TextTranslator(Protocol):
    """An MT model that turns Armenian text into English text."""

    def translate(self, text: str) -> str:
        """Return the English translation of the Armenian text."""
        ...


def build_source(prompt_format: PromptFormat, pieces: list[str]) -> list[str]:
    """Return the model's input tokens for the source text's sentencepieces."""
    return [*prompt_format.source_prefix, *pieces, END_OF_TEXT]


def strip_target_prefix(prompt_format: PromptFormat, tokens: list[str]) -> list[str]:
    """Return the translation's tokens without the forced target prefix."""
    return tokens[len(prompt_format.target_prefix) :]


class CTranslate2Translator:
    """A converted MT model run with CTranslate2 and sentencepiece, GPU first."""

    def __init__(
        self,
        model: str = MT_MODEL,
        *,
        models_dir: Path = MT_MODELS_DIR,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        if model not in MT_MODELS:
            raise ModelLoadError(
                f"Unknown translation model '{model}'; "
                f"choose one of: {', '.join(MT_MODELS)}"
            )
        model_dir = models_dir / model
        if not model_dir.is_dir():
            raise ModelLoadError(
                f"Translation model not found in {model_dir}. Convert it once with: "
                f"python benchmarks/convert_mt.py {model}"
            )
        self._format = MT_MODELS[model].prompt_format
        # The CUDA DLL folders must be registered before CTranslate2 is imported.
        add_cuda_dll_dirs()
        import ctranslate2
        import sentencepiece

        if device is None:
            device = "cuda" if has_cuda_gpu() else "cpu"
        if compute_type is None:
            compute_type = GPU_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE
        try:
            self._model = ctranslate2.Translator(
                str(model_dir), device=device, compute_type=compute_type
            )
            self._source_spm = sentencepiece.SentencePieceProcessor(
                model_file=str(model_dir / self._format.source_spm)
            )
            self._target_spm = sentencepiece.SentencePieceProcessor(
                model_file=str(model_dir / self._format.target_spm)
            )
        except Exception as error:
            raise ModelLoadError(
                f"Could not load translation model '{model}' on {device}: {error}"
            ) from error

    def translate(self, text: str) -> str:
        """Return the English translation of the Armenian text."""
        pieces = self._source_spm.encode(self._format.text_prefix + text, out_type=str)
        target_prefix = self._format.target_prefix
        [result] = self._model.translate_batch(
            [build_source(self._format, pieces)],
            target_prefix=[list(target_prefix)] if target_prefix else None,
            beam_size=BEAM_SIZE,
        )
        tokens = strip_target_prefix(self._format, result.hypotheses[0])
        return self._target_spm.decode(tokens)


class Translator:
    """Translate Armenian transcripts into English."""

    def __init__(self, model: TextTranslator) -> None:
        self._model = model

    def warm_up(self) -> None:
        """Run the model once at startup, so the first phrase isn't slow."""
        try:
            self._model.translate(WARM_UP_TEXT)
        except Exception as error:
            raise ModelLoadError(
                f"Translation model failed its first run: {error}"
            ) from error

    def translate(self, text: str) -> str:
        """Return the English translation of one Armenian transcript."""
        english = " ".join(self._model.translate(text).split())
        if not english:
            logger.warning("Translation model gave no output for: %r", text)
        return english

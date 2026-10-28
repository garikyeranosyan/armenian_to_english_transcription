"""Armenian speech to Armenian text, behind an interface any ASR model can fill."""

import logging
import os
import re
import sys
import sysconfig
from pathlib import Path
from typing import Protocol

import numpy as np

from live_hy_en.capture import SAMPLE_RATE_HZ

# Fine-tuned models converted by benchmarks/convert_asr.py, one folder each.
ASR_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "asr"
# Chosen by benchmarks/asr_benchmark.py: lowest WER on FLEURS hy_am. Any
# faster-whisper model name, Hugging Face model ID or local CTranslate2 folder
# works here.
ASR_MODEL = str(ASR_MODELS_DIR / "buzzasr-armenian")
LANGUAGE = "hy"
# Greedy decoding, as BuzzASR reports its scores: beam search on its 32-layer
# decoder made phrases miss the 5 s limit.
BEAM_SIZE = 1
# Decode at 0, then retry at most twice when the text repeats itself or the model
# is unsure. faster-whisper's default six attempts took 8 s on one phrase.
TEMPERATURES = (0.0, 0.2, 0.4)
# 8-bit weights halve GPU memory: in float16, BuzzASR and NLLB-1.3B together
# overflowed the laptop's GPU and slowed every model past the 5 s limit.
GPU_COMPUTE_TYPE = "int8_float16"
CPU_COMPUTE_TYPE = "int8"
# Output below this share of Armenian letters counts as the wrong script.
MIN_ARMENIAN_RATIO = 0.5
WARM_UP_S = 1.0

ARMENIAN_LETTER = re.compile("[\u0531-\u0556\u0560-\u0588\ufb13-\ufb17]")

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """A model could not be loaded or failed its first run."""


class SpeechRecognizer(Protocol):
    """An ASR model that turns Armenian speech into text.

    It takes 16 kHz mono float32 samples, and its language is fixed to Armenian.
    """

    def recognize(self, samples: np.ndarray) -> str:
        """Return the text spoken in the samples."""
        ...


def add_cuda_dll_dirs() -> None:
    """Let CTranslate2 find the cuBLAS and cuDNN DLLs from NVIDIA's pip wheels."""
    if sys.platform != "win32":
        return
    nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for library in ("cublas", "cudnn"):
        dll_dir = nvidia_dir / library / "bin"
        path = os.environ.get("PATH", "")
        if dll_dir.is_dir() and str(dll_dir) not in path.split(os.pathsep):
            os.add_dll_directory(str(dll_dir))
            # Some CUDA libraries load their own dependencies through PATH.
            os.environ["PATH"] = f"{dll_dir}{os.pathsep}{path}"


def has_cuda_gpu() -> bool:
    """Return whether CTranslate2 sees a CUDA GPU, warning when it doesn't."""
    # Only called after add_cuda_dll_dirs(), which must run before this import.
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        return True
    logger.warning("No CUDA GPU found; running models on the CPU, which is slower.")
    return False


class FasterWhisperRecognizer:
    """A Whisper model run with faster-whisper (CTranslate2), GPU first."""

    def __init__(
        self,
        model: str = ASR_MODEL,
        *,
        device: str | None = None,
        compute_type: str | None = None,
        token: str | None = None,
    ) -> None:
        # The CUDA DLL folders must be registered before CTranslate2 is imported.
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        if device is None:
            device = "cuda" if has_cuda_gpu() else "cpu"
        if compute_type is None:
            compute_type = GPU_COMPUTE_TYPE if device == "cuda" else CPU_COMPUTE_TYPE
        try:
            self._model = WhisperModel(
                model, device=device, compute_type=compute_type, use_auth_token=token
            )
        except Exception as error:
            raise ModelLoadError(
                f"Could not load speech model '{model}' on {device}: {error}"
            ) from error

    def recognize(self, samples: np.ndarray) -> str:
        """Return the Armenian text spoken in the samples."""
        segments, _ = self._model.transcribe(
            samples,
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURES,
            # Segments are independent phrases; earlier text only spreads errors.
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments)


def is_armenian(text: str) -> bool:
    """Return whether most letters in the text are Armenian."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return True
    armenian = sum(1 for char in letters if ARMENIAN_LETTER.match(char))
    return armenian / len(letters) >= MIN_ARMENIAN_RATIO


class Transcriber:
    """Transcribe speech segments, dropping output that is empty or not Armenian."""

    def __init__(self, recognizer: SpeechRecognizer) -> None:
        self._recognizer = recognizer

    def warm_up(self) -> None:
        """Run the model once at startup, so the first phrase isn't slow."""
        silence = np.zeros(round(WARM_UP_S * SAMPLE_RATE_HZ), dtype=np.float32)
        try:
            self._recognizer.recognize(silence)
        except Exception as error:
            # Some GPU libraries (e.g. cuDNN) are only loaded on first use.
            raise ModelLoadError(
                f"Speech model failed its first run: {error}"
            ) from error

    def transcribe(self, samples: np.ndarray) -> str | None:
        """Return the Armenian text of one segment, or None if it isn't usable."""
        text = " ".join(self._recognizer.recognize(samples).split())
        if not text:
            return None
        if not is_armenian(text):
            # Wrong-script output is a known failure on low-resource languages.
            logger.warning("Dropped transcript not in Armenian script: %r", text)
            return None
        return text

"""Armenian speech to Armenian text, behind an interface any ASR model can fill."""

import logging
import re
from typing import Protocol

import numpy as np

from live_hy_en.capture import SAMPLE_RATE_HZ

# Output below this share of Armenian letters counts as the wrong script.
MIN_ARMENIAN_RATIO = 0.5
WARM_UP_S = 1.0

ARMENIAN_LETTER = re.compile("[Ա-Ֆա-ևﬓ-ﬗ]")

logger = logging.getLogger(__name__)


class SpeechRecognizer(Protocol):
    """An ASR model that turns Armenian speech into text.

    The model is chosen by the M1 benchmark (benchmarks/asr_benchmark.py). Its
    adapter takes 16 kHz mono float32 samples and fixes the language to Armenian.
    """

    def recognize(self, samples: np.ndarray) -> str:
        """Return the text spoken in the samples."""
        ...


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
        self._recognizer.recognize(
            np.zeros(round(WARM_UP_S * SAMPLE_RATE_HZ), dtype=np.float32)
        )

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

"""Voice activity detection: cut one source's audio frames into speech segments."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime

from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, Source

VAD_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
VAD_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/"
    "src/silero_vad/data/silero_vad.onnx"
)
# Silero VAD scores fixed windows of 512 samples (32 ms) at 16 kHz.
WINDOW_SAMPLES = 512
# Silero needs the last 64 samples of the previous window as context.
CONTEXT_SAMPLES = 64
SPEECH_THRESHOLD = 0.5
# Lower threshold to end speech, as in Silero's own iterator, so it doesn't flicker.
SILENCE_THRESHOLD = 0.35
END_SILENCE_MS = 700
MIN_SPEECH_MS = 250
# Whisper drops words cut right at a segment's edge, and Silero only scores a soft
# first sound as speech after it has begun, so audio either side is kept too.
PRE_SPEECH_PAD_MS = 300
# Taken from the silence already waited for, so it adds no latency; a phrase
# ended by a short pause (below) keeps only that pause.
POST_SPEECH_PAD_MS = 200
MAX_SEGMENT_S = 10.0
# Once a phrase is this long, a short gap between words ends it, so long speech
# is split there instead of at MAX_SEGMENT_S in the middle of a word. Chosen on
# FLEURS hy_am: cuts inside speech fell from 89 % to 13 % of all cuts.
LONG_PHRASE_S = 5.0
SHORT_PAUSE_MS = 100
# Frames further apart than this mean the source sent nothing (loopback silence).
MAX_FRAME_GAP_S = 0.1

WINDOW_S = WINDOW_SAMPLES / SAMPLE_RATE_HZ
END_SILENCE_WINDOWS = round(END_SILENCE_MS / 1000 / WINDOW_S)
MIN_SPEECH_WINDOWS = round(MIN_SPEECH_MS / 1000 / WINDOW_S)
PRE_SPEECH_PAD_WINDOWS = round(PRE_SPEECH_PAD_MS / 1000 / WINDOW_S)
POST_SPEECH_PAD_WINDOWS = round(POST_SPEECH_PAD_MS / 1000 / WINDOW_S)
MAX_SEGMENT_WINDOWS = int(MAX_SEGMENT_S / WINDOW_S)
LONG_PHRASE_WINDOWS = round(LONG_PHRASE_S / WINDOW_S)
SHORT_PAUSE_WINDOWS = round(SHORT_PAUSE_MS / 1000 / WINDOW_S)


class VadModelError(Exception):
    """The voice activity model could not be loaded."""


@dataclass(frozen=True)
class SpeechSegment:
    """One phrase of speech from one source, in session seconds.

    `start_s` and `end_s` mark the speech itself; `samples` also hold the padding
    around it, so latency is still measured from when the speaker stopped.
    """

    source: Source
    start_s: float
    end_s: float
    samples: np.ndarray


class SpeechDetector(Protocol):
    """Scores 512-sample windows of 16 kHz audio for speech."""

    def speech_probability(self, window: np.ndarray) -> float:
        """Return the probability, 0 to 1, that the window contains speech."""
        ...

    def reset(self) -> None:
        """Forget earlier audio, e.g. after a gap in the stream."""
        ...


class SileroVad:
    """Silero VAD run with onnxruntime on the CPU."""

    def __init__(self, model_path: Path = VAD_MODEL_PATH) -> None:
        if not model_path.is_file():
            raise VadModelError(
                f"Silero VAD model not found at {model_path}. "
                f"Download it from {VAD_MODEL_URL}"
            )
        options = onnxruntime.SessionOptions()
        # One thread is enough for a 32 ms window and leaves cores for capture.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # The CPU keeps the GPU free for transcription and translation.
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._sample_rate = np.array(SAMPLE_RATE_HZ, dtype=np.int64)
        self.reset()

    def speech_probability(self, window: np.ndarray) -> float:
        """Return the speech probability of one 512-sample window."""
        model_input = np.concatenate([self._context, window])[np.newaxis, :]
        probability, self._state = self._session.run(
            None,
            {"input": model_input, "state": self._state, "sr": self._sample_rate},
        )
        self._context = window[-CONTEXT_SAMPLES:]
        return float(probability[0, 0])

    def reset(self) -> None:
        """Clear the model's memory of earlier windows."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)


class Segmenter:
    """Turn one source's frames into speech segments, cut at pauses."""

    def __init__(self, detector: SpeechDetector, source: Source) -> None:
        self._detector = detector
        self._source = source
        self._pending = np.zeros(0, dtype=np.float32)
        self._next_window_s: float | None = None
        # The latest audio before speech starts, kept as the segment's lead-in.
        self._recent_windows: deque[np.ndarray] = deque(maxlen=PRE_SPEECH_PAD_WINDOWS)
        self._is_speaking = False
        self._windows: list[np.ndarray] = []
        self._pad_window_count = 0
        self._segment_start_s = 0.0
        self._speech_window_count = 0

    def process(self, frame: AudioFrame) -> list[SpeechSegment]:
        """Add one frame; return any segments that ended in it."""
        segments = []
        expected_start_s = self._next_window_s
        if expected_start_s is not None:
            expected_start_s += self._pending.size / SAMPLE_RATE_HZ
            if frame.start_s - expected_start_s > MAX_FRAME_GAP_S:
                # No audio arrived for a while, so whatever was said has ended.
                segments += self.flush()
                self._detector.reset()
        if self._next_window_s is None:
            self._next_window_s = frame.start_s
        self._pending = np.concatenate([self._pending, frame.samples])
        while self._pending.size >= WINDOW_SAMPLES:
            window = self._pending[:WINDOW_SAMPLES]
            self._pending = self._pending[WINDOW_SAMPLES:]
            segments += self._process_window(window, self._next_window_s)
            self._next_window_s += WINDOW_S
        return segments

    def flush(self) -> list[SpeechSegment]:
        """End any open segment, e.g. at shutdown or after a gap in the audio."""
        segments = self._end_segment()
        self._is_speaking = False
        self._pending = np.zeros(0, dtype=np.float32)
        self._next_window_s = None
        # Audio from before a gap doesn't lead into what comes after it.
        self._recent_windows.clear()
        return segments

    def _process_window(
        self, window: np.ndarray, start_s: float
    ) -> list[SpeechSegment]:
        probability = self._detector.speech_probability(window)
        segments = []
        if not self._is_speaking:
            if probability >= SPEECH_THRESHOLD:
                self._start_segment(window, start_s)
            else:
                self._recent_windows.append(window)
            return segments
        if len(self._windows) == MAX_SEGMENT_WINDOWS:
            # Cap long monologues so one segment can't break the latency limit.
            segments += self._end_segment()
            self._segment_start_s = start_s
        self._windows.append(window)
        if probability >= SILENCE_THRESHOLD:
            self._speech_window_count = len(self._windows)
        silence_windows = len(self._windows) - self._speech_window_count
        if silence_windows >= self._end_silence_windows():
            silence = self._windows[self._speech_window_count :]
            segments += self._end_segment()
            self._is_speaking = False
            self._recent_windows.extend(silence)
        return segments

    def _end_silence_windows(self) -> int:
        """Return how much silence ends the open phrase: less once it is long."""
        phrase_windows = self._speech_window_count - self._pad_window_count
        if phrase_windows >= LONG_PHRASE_WINDOWS:
            return SHORT_PAUSE_WINDOWS
        return END_SILENCE_WINDOWS

    def _start_segment(self, window: np.ndarray, start_s: float) -> None:
        self._is_speaking = True
        self._windows = [*self._recent_windows, window]
        self._pad_window_count = len(self._recent_windows)
        self._recent_windows.clear()
        self._segment_start_s = start_s
        self._speech_window_count = len(self._windows)

    def _end_segment(self) -> list[SpeechSegment]:
        """Emit the open segment with its padding, unless its speech is too short."""
        speech_windows = self._speech_window_count - self._pad_window_count
        kept_windows = self._windows[
            : self._speech_window_count + POST_SPEECH_PAD_WINDOWS
        ]
        self._windows = []
        self._pad_window_count = 0
        self._speech_window_count = 0
        if speech_windows < MIN_SPEECH_WINDOWS:
            return []
        return [
            SpeechSegment(
                source=self._source,
                start_s=self._segment_start_s,
                end_s=self._segment_start_s + speech_windows * WINDOW_S,
                samples=np.concatenate(kept_windows),
            )
        ]

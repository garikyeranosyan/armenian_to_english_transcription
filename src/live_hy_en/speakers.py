"""Speaker labels: "Me" for the microphone, stable "Speaker N" for system audio."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import kaldi_native_fbank
import numpy as np
import onnxruntime

from live_hy_en.capture import SAMPLE_RATE_HZ, Source
from live_hy_en.segmenter import SpeechSegment
from live_hy_en.transcriber import ModelLoadError

SPEAKER_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "wespeaker_resnet34_LM.onnx"
)
SPEAKER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
MIC_LABEL = "Me"
CALL_LABEL_PREFIX = "Speaker"
# Provisional until tuned on call-quality voice mixes (TODO.md).
SIMILARITY_THRESHOLD = 0.4
# Feature settings WeSpeaker was trained with: 80 log-mel bands over 25 ms
# Hamming windows every 10 ms, on samples in the int16 range, without dither.
MEL_BANDS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
INT16_SCALE = 32768.0
WARM_UP_S = 1.0


class VoiceEncoder(Protocol):
    """A speaker-embedding model: one voiceprint per stretch of speech."""

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Return the voiceprint of 16 kHz mono float32 speech."""
        ...


def compute_fbank(samples: np.ndarray) -> np.ndarray:
    """Return mean-normalised log-mel features, one row per 10 ms frame."""
    options = kaldi_native_fbank.FbankOptions()
    options.frame_opts.samp_freq = SAMPLE_RATE_HZ
    options.frame_opts.frame_length_ms = FRAME_LENGTH_MS
    options.frame_opts.frame_shift_ms = FRAME_SHIFT_MS
    options.frame_opts.dither = 0.0
    options.frame_opts.window_type = "hamming"
    options.mel_opts.num_bins = MEL_BANDS
    fbank = kaldi_native_fbank.OnlineFbank(options)
    fbank.accept_waveform(SAMPLE_RATE_HZ, (samples * INT16_SCALE).tolist())
    fbank.input_finished()
    features = np.array(
        [fbank.get_frame(index) for index in range(fbank.num_frames_ready)],
        dtype=np.float32,
    )
    # Removing the mean cancels the channel, e.g. a phone line's colouring.
    return features - features.mean(axis=0)


class WeSpeakerEncoder:
    """WeSpeaker ResNet34 (VoxCeleb) run with onnxruntime on the CPU."""

    def __init__(self, model_path: Path = SPEAKER_MODEL_PATH) -> None:
        if not model_path.is_file():
            raise ModelLoadError(
                f"Speaker model not found at {model_path}. "
                f"Download it from {SPEAKER_MODEL_URL}"
            )
        try:
            # The CPU keeps the GPU free for transcription and translation.
            self._session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as error:
            raise ModelLoadError(
                f"Could not load speaker model {model_path}: {error}"
            ) from error
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Return the 256-value voiceprint of the speech."""
        features = compute_fbank(samples)[np.newaxis, :, :]
        [embeddings] = self._session.run(None, {self._input_name: features})
        return embeddings[0]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), np.finfo(np.float32).tiny)


@dataclass
class _CallSpeaker:
    label: str
    # Sum of the speaker's unit voiceprints; its direction is the mean voice.
    voiceprint_sum: np.ndarray


class SpeakerTracker:
    """Label each segment's speaker; call-side labels never change once given."""

    def __init__(
        self, encoder: VoiceEncoder, similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> None:
        self._encoder = encoder
        self._similarity_threshold = similarity_threshold
        self._speakers: list[_CallSpeaker] = []

    def warm_up(self) -> None:
        """Run the model once at startup, so the first phrase isn't slow."""
        silence = np.zeros(round(WARM_UP_S * SAMPLE_RATE_HZ), dtype=np.float32)
        try:
            self._encoder.embed(silence)
        except Exception as error:
            raise ModelLoadError(
                f"Speaker model failed its first run: {error}"
            ) from error

    def assign_speaker(self, segment: SpeechSegment) -> str:
        """Return "Me" for the microphone, otherwise the matching "Speaker N"."""
        if segment.source is Source.MIC:
            return MIC_LABEL
        voiceprint = _unit(self._encoder.embed(segment.samples))
        speaker = self._closest_speaker(voiceprint)
        if speaker is None:
            speaker = _CallSpeaker(
                f"{CALL_LABEL_PREFIX} {len(self._speakers) + 1}",
                np.zeros_like(voiceprint),
            )
            self._speakers.append(speaker)
        speaker.voiceprint_sum = speaker.voiceprint_sum + voiceprint
        return speaker.label

    def _closest_speaker(self, voiceprint: np.ndarray) -> "_CallSpeaker | None":
        """Return the most similar known speaker, if similar enough to be them."""
        best, best_similarity = None, self._similarity_threshold
        for speaker in self._speakers:
            similarity = float(_unit(speaker.voiceprint_sum) @ voiceprint)
            if similarity >= best_similarity:
                best, best_similarity = speaker, similarity
        return best

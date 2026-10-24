"""Microphone and WASAPI loopback capture in one common format on a shared clock."""

import logging
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

from live_hy_en.config import AudioDevice

SAMPLE_RATE_HZ = 16_000
# 32 ms per device callback: one Silero VAD window at 16 kHz.
CALLBACK_MS = 32

logger = logging.getLogger(__name__)


class Source(str, Enum):
    """Where a frame of audio was recorded."""

    MIC = "mic"
    SYSTEM = "system"


@dataclass(frozen=True)
class AudioFrame:
    """Mono float32 audio at SAMPLE_RATE_HZ, starting `start_s` into the session."""

    source: Source
    start_s: float
    samples: np.ndarray


class CaptureError(Exception):
    """An audio device could not be recorded."""


class SessionClock:
    """Seconds since the session started; shared by both sources."""

    def __init__(self, now: Callable[[], float] = time.monotonic) -> None:
        self._now = now
        self._start_s = now()

    def elapsed_s(self) -> float:
        """Return the seconds since the clock was created."""
        return self._now() - self._start_s


class FrameConverter:
    """Turn raw device audio into mono float32 at SAMPLE_RATE_HZ."""

    def __init__(self, channels: int, sample_rate_hz: int) -> None:
        self._channels = channels
        # Streaming keeps filter state between callbacks, so chunk edges don't click.
        self._resampler = soxr.ResampleStream(
            sample_rate_hz, SAMPLE_RATE_HZ, 1, dtype="float32"
        )

    def convert(self, raw_audio: bytes) -> np.ndarray:
        """Return the converted samples; may be empty while the resampler fills."""
        interleaved = np.frombuffer(raw_audio, dtype=np.float32)
        mono = interleaved.reshape(-1, self._channels).mean(axis=1)
        return self._resampler.resample_chunk(mono)


class AudioCapture:
    """Record one device and put its AudioFrames on a queue without ever waiting."""

    def __init__(
        self,
        audio: pyaudio.PyAudio,
        device: AudioDevice,
        source: Source,
        clock: SessionClock,
        frames: queue.SimpleQueue[AudioFrame],
    ) -> None:
        self._audio = audio
        self._device = device
        self._source = source
        self._clock = clock
        self._frames = frames
        self._converter = FrameConverter(device.channels, device.sample_rate_hz)
        self._stream: Any = None

    def start(self) -> None:
        """Open the device and start recording."""
        try:
            self._stream = self._audio.open(
                format=pyaudio.paFloat32,
                channels=self._device.channels,
                rate=self._device.sample_rate_hz,
                input=True,
                input_device_index=self._device.index,
                frames_per_buffer=self._device.sample_rate_hz * CALLBACK_MS // 1000,
                stream_callback=self._on_audio,
            )
        except OSError as error:
            raise CaptureError(
                f"Cannot record {self._source.value} audio from "
                f"'{self._device.name}': {error}"
            ) from error

    def stop(self) -> None:
        """Stop recording and close the device."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _on_audio(
        self, in_data: bytes, frame_count: int, time_info: dict, status: int
    ) -> tuple[None, int]:
        # Runs on the audio thread: convert, enqueue and return at once.
        if status & pyaudio.paInputOverflow:
            logger.warning(
                "%s audio overflowed; some audio was lost", self._source.value
            )
        samples = self._converter.convert(in_data)
        if samples.size:
            # The newest sample was recorded just now, so the frame ends now.
            # Loopback sends nothing while the call is silent, so sample counts
            # alone would drift from the session clock.
            start_s = self._clock.elapsed_s() - samples.size / SAMPLE_RATE_HZ
            self._frames.put(AudioFrame(self._source, start_s, samples))
        return None, pyaudio.paContinue

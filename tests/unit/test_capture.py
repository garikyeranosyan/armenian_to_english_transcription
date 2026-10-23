"""Tests for audio capture (PLAN.md §5.1: two sources captured at once)."""

import logging
import queue

import numpy as np
import pyaudiowpatch as pyaudio
import pytest

from live_hy_en.capture import (
    CALLBACK_MS,
    SAMPLE_RATE_HZ,
    AudioCapture,
    AudioFrame,
    CaptureError,
    FrameConverter,
    SessionClock,
    Source,
)
from live_hy_en.config import AudioDevice

DEVICE_RATE_HZ = 48_000
MIC = AudioDevice(
    index=1,
    name="Microphone Array (Realtek)",
    channels=2,
    sample_rate_hz=DEVICE_RATE_HZ,
)
LOOPBACK = AudioDevice(
    index=5,
    name="Speakers (Realtek) [Loopback]",
    channels=2,
    sample_rate_hz=DEVICE_RATE_HZ,
)
# The resampler holds back a few ms, so totals are checked to this tolerance.
RESAMPLER_SLACK = 0.05


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, now_s: float = 100.0) -> None:
        self.now_s = now_s

    def __call__(self) -> float:
        return self.now_s


class FakeStream:
    """Stands in for a PyAudio stream; the test calls its callback."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.is_closed = False

    def deliver(self, chunk: np.ndarray, status: int = 0) -> tuple:
        """Send one interleaved float32 chunk to the capture callback."""
        frames = chunk.size // self.kwargs["channels"]
        return self.kwargs["stream_callback"](chunk.tobytes(), frames, {}, status)

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.is_closed = True


class FakeAudio:
    """Stands in for pyaudiowpatch.PyAudio with the method capture uses."""

    def __init__(self, *, fails: bool = False) -> None:
        self.streams: dict[int, FakeStream] = {}
        self._fails = fails

    def open(self, **kwargs) -> FakeStream:
        if self._fails:
            raise OSError("Device unavailable")
        stream = FakeStream(**kwargs)
        self.streams[kwargs["input_device_index"]] = stream
        return stream


def _stereo_chunk(left: float, right: float) -> np.ndarray:
    frames = DEVICE_RATE_HZ * CALLBACK_MS // 1000
    return (
        np.column_stack([np.full(frames, left), np.full(frames, right)])
        .astype(np.float32)
        .ravel()
    )


def _drain(frames: queue.SimpleQueue) -> list[AudioFrame]:
    drained = []
    while not frames.empty():
        drained.append(frames.get())
    return drained


def _start(audio: FakeAudio, device: AudioDevice, source: Source, clock: SessionClock):
    frames: queue.SimpleQueue[AudioFrame] = queue.SimpleQueue()
    capture = AudioCapture(audio, device, source, clock, frames)
    capture.start()
    return capture, frames


def test_converter_outputs_mono_float32_at_common_rate():
    converter = FrameConverter(channels=2, sample_rate_hz=DEVICE_RATE_HZ)
    chunks = 50
    output = np.concatenate(
        [converter.convert(_stereo_chunk(0.2, 0.6).tobytes()) for _ in range(chunks)]
    )
    expected_samples = chunks * SAMPLE_RATE_HZ * CALLBACK_MS // 1000
    assert output.dtype == np.float32
    assert output.ndim == 1
    assert output.size == pytest.approx(expected_samples, rel=RESAMPLER_SLACK)
    # Channels are averaged: (0.2 + 0.6) / 2.
    assert output[-100:] == pytest.approx(0.4, abs=1e-3)


@pytest.mark.parametrize("sample_rate_hz", [44_100, 48_000, 96_000])
def test_converter_handles_common_device_rates(sample_rate_hz):
    converter = FrameConverter(channels=1, sample_rate_hz=sample_rate_hz)
    # Long enough that the resampler's fixed hold-back stays within the slack.
    seconds = 3
    tone = np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, sample_rate_hz * seconds))
    output = converter.convert(tone.astype(np.float32).tobytes())
    assert output.size == pytest.approx(SAMPLE_RATE_HZ * seconds, rel=RESAMPLER_SLACK)


def test_session_clock_counts_from_creation():
    now = FakeClock(now_s=100.0)
    clock = SessionClock(now)
    now.now_s = 102.5
    assert clock.elapsed_s() == pytest.approx(2.5)


def test_start_opens_device_in_its_native_format():
    audio = FakeAudio()
    _start(audio, MIC, Source.MIC, SessionClock(FakeClock()))
    kwargs = audio.streams[MIC.index].kwargs
    assert kwargs["input"] is True
    assert kwargs["format"] == pyaudio.paFloat32
    assert kwargs["channels"] == MIC.channels
    assert kwargs["rate"] == MIC.sample_rate_hz


def test_unavailable_device_gives_clear_error():
    capture = AudioCapture(
        FakeAudio(fails=True), MIC, Source.MIC, SessionClock(), queue.SimpleQueue()
    )
    with pytest.raises(CaptureError, match="Microphone Array"):
        capture.start()


def test_frames_are_tagged_with_their_source():
    audio = FakeAudio()
    clock = SessionClock(FakeClock())
    _, mic_frames = _start(audio, MIC, Source.MIC, clock)
    _, system_frames = _start(audio, LOOPBACK, Source.SYSTEM, clock)
    for _ in range(5):
        audio.streams[MIC.index].deliver(_stereo_chunk(0.1, 0.1))
        audio.streams[LOOPBACK.index].deliver(_stereo_chunk(0.1, 0.1))
    assert {frame.source for frame in _drain(mic_frames)} == {Source.MIC}
    assert {frame.source for frame in _drain(system_frames)} == {Source.SYSTEM}


def test_frame_ends_at_the_moment_it_arrives():
    now = FakeClock(now_s=100.0)
    clock = SessionClock(now)
    audio = FakeAudio()
    _, frames = _start(audio, MIC, Source.MIC, clock)
    now.now_s = 103.0
    for _ in range(3):
        audio.streams[MIC.index].deliver(_stereo_chunk(0.1, 0.1))
    for frame in _drain(frames):
        end_s = frame.start_s + frame.samples.size / SAMPLE_RATE_HZ
        assert end_s == pytest.approx(3.0)


def test_both_sources_share_one_clock_across_silence_gaps():
    now = FakeClock(now_s=100.0)
    clock = SessionClock(now)
    audio = FakeAudio()
    _, mic_frames = _start(audio, MIC, Source.MIC, clock)
    _, system_frames = _start(audio, LOOPBACK, Source.SYSTEM, clock)
    for _ in range(3):
        audio.streams[MIC.index].deliver(_stereo_chunk(0.1, 0.1))
    # Loopback delivers nothing for 10 s of call silence, then audio resumes.
    now.now_s = 110.0
    for _ in range(3):
        audio.streams[LOOPBACK.index].deliver(_stereo_chunk(0.1, 0.1))
    assert _drain(mic_frames)[-1].start_s < 1.0
    assert _drain(system_frames)[-1].start_s == pytest.approx(
        10.0, abs=CALLBACK_MS / 1000
    )


def test_silent_source_does_not_block_the_other():
    audio = FakeAudio()
    clock = SessionClock(FakeClock())
    _, mic_frames = _start(audio, MIC, Source.MIC, clock)
    _, system_frames = _start(audio, LOOPBACK, Source.SYSTEM, clock)
    for _ in range(10):
        audio.streams[MIC.index].deliver(_stereo_chunk(0.1, 0.1))
    assert _drain(mic_frames)
    assert system_frames.empty()


def test_callback_never_waits_on_a_slow_consumer():
    audio = FakeAudio()
    _, frames = _start(audio, MIC, Source.MIC, SessionClock(FakeClock()))
    # Nobody reads the queue; every callback must still return "continue".
    for _ in range(1_000):
        result = audio.streams[MIC.index].deliver(_stereo_chunk(0.1, 0.1))
        assert result == (None, pyaudio.paContinue)
    assert frames.qsize() > 0


def test_overflow_is_logged(caplog):
    audio = FakeAudio()
    _start(audio, MIC, Source.MIC, SessionClock(FakeClock()))
    with caplog.at_level(logging.WARNING):
        audio.streams[MIC.index].deliver(
            _stereo_chunk(0.1, 0.1), status=pyaudio.paInputOverflow
        )
    assert "overflowed" in caplog.text


def test_stop_closes_the_device():
    audio = FakeAudio()
    capture, _ = _start(audio, MIC, Source.MIC, SessionClock(FakeClock()))
    capture.stop()
    assert audio.streams[MIC.index].is_closed

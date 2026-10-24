"""Connect capture, segmentation and the models, and print each phrase live."""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pyaudiowpatch as pyaudio

from live_hy_en.capture import AudioCapture, AudioFrame, SessionClock, Source
from live_hy_en.config import Settings
from live_hy_en.output import TranscriptEntry, print_entry, print_message
from live_hy_en.segmenter import Segmenter, SileroVad, SpeechDetector, SpeechSegment
from live_hy_en.speakers import SpeakerTracker, WeSpeakerEncoder
from live_hy_en.transcriber import FasterWhisperRecognizer, Transcriber
from live_hy_en.translator import CTranslate2Translator, Translator

# Phrases waiting for transcription. When full, segmentation waits for a free
# slot (nothing is dropped) while capture keeps recording into its own queue.
MAX_QUEUED_SEGMENTS = 8
# Transcripts waiting for translation.
MAX_QUEUED_TRANSCRIPTS = 8
# How often the main thread wakes while it waits for Ctrl+C.
WAIT_FOR_CTRL_C_S = 0.2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Models:
    """Loaded models; each source gets its own VAD, since a VAD keeps state."""

    transcriber: Transcriber
    speakers: SpeakerTracker
    translator: Translator
    make_detector: Callable[[], SpeechDetector]


@dataclass(frozen=True)
class _Transcript:
    """A phrase after transcription, waiting for translation."""

    segment: SpeechSegment
    speaker: str
    armenian: str


def load_models() -> Models:
    """Load every model and run each once, so the first phrase isn't slow."""
    print_message("Loading models…")
    transcriber = Transcriber(FasterWhisperRecognizer())
    speakers = SpeakerTracker(WeSpeakerEncoder())
    translator = Translator(CTranslate2Translator())
    for model in (transcriber, speakers, translator):
        model.warm_up()
    return Models(transcriber, speakers, translator, SileroVad)


class Pipeline:
    """Segment both sources' frames, then transcribe, label, translate and show."""

    def __init__(
        self,
        models: Models,
        started_at: datetime,
        *,
        show: Callable[[TranscriptEntry], None] = print_entry,
        max_queued_segments: int = MAX_QUEUED_SEGMENTS,
    ) -> None:
        self._models = models
        self._started_at = started_at
        self._show = show
        # Capture puts frames here; None tells a source's segmenter to finish.
        self.frames: dict[Source, queue.SimpleQueue[AudioFrame | None]] = {
            source: queue.SimpleQueue() for source in Source
        }
        self._segments: queue.Queue[SpeechSegment | None] = queue.Queue(
            max_queued_segments
        )
        self._transcripts: queue.Queue[_Transcript | None] = queue.Queue(
            MAX_QUEUED_TRANSCRIPTS
        )
        self._segmenter_threads = [
            threading.Thread(
                target=self._segment_frames,
                args=(self.frames[source], Segmenter(models.make_detector(), source)),
                name=f"segmenter-{source.value}",
                daemon=True,
            )
            for source in Source
        ]
        # Two stages, so one phrase is translated while the next is transcribed.
        self._transcriber_thread = threading.Thread(
            target=self._transcribe_segments, name="transcriber", daemon=True
        )
        self._translator_thread = threading.Thread(
            target=self._translate_transcripts, name="translator", daemon=True
        )

    def start(self) -> None:
        """Start processing frames as they arrive."""
        for thread in self._threads():
            thread.start()

    def stop(self) -> None:
        """Finish every phrase already heard, then stop all threads."""
        for frames in self.frames.values():
            frames.put(None)
        for thread in self._segmenter_threads:
            thread.join()
        self._segments.put(None)
        self._transcriber_thread.join()
        self._transcripts.put(None)
        self._translator_thread.join()

    def _threads(self) -> list[threading.Thread]:
        return [
            *self._segmenter_threads,
            self._transcriber_thread,
            self._translator_thread,
        ]

    def _segment_frames(
        self, frames: queue.SimpleQueue[AudioFrame | None], segmenter: Segmenter
    ) -> None:
        while (frame := frames.get()) is not None:
            for segment in segmenter.process(frame):
                self._queue_segment(segment)
        for segment in segmenter.flush():
            self._queue_segment(segment)

    def _queue_segment(self, segment: SpeechSegment) -> None:
        try:
            self._segments.put_nowait(segment)
        except queue.Full:
            logger.warning(
                "Transcription is falling behind; %s audio is waiting for it",
                segment.source.value,
            )
            self._segments.put(segment)

    def _transcribe_segments(self) -> None:
        while (segment := self._segments.get()) is not None:
            try:
                transcript = self._transcribe(segment)
            except Exception:
                # One bad phrase must not end the session; the next may be fine.
                logger.exception(
                    "Could not transcribe the %s phrase at %.1f s",
                    segment.source.value,
                    segment.start_s,
                )
                continue
            if transcript is not None:
                self._transcripts.put(transcript)

    def _transcribe(self, segment: SpeechSegment) -> _Transcript | None:
        armenian = self._models.transcriber.transcribe(segment.samples)
        if armenian is None:
            return None
        # Labelled after transcription, so dropped noise never creates a speaker.
        speaker = self._models.speakers.assign_speaker(segment)
        return _Transcript(segment, speaker, armenian)

    def _translate_transcripts(self) -> None:
        while (transcript := self._transcripts.get()) is not None:
            try:
                self._show(self._translate(transcript))
            except Exception:
                logger.exception(
                    "Could not translate or print the phrase at %.1f s",
                    transcript.segment.start_s,
                )

    def _translate(self, transcript: _Transcript) -> TranscriptEntry:
        english = self._models.translator.translate(transcript.armenian)
        spoken_at = self._started_at + timedelta(seconds=transcript.segment.start_s)
        return TranscriptEntry(
            spoken_at, transcript.speaker, transcript.armenian, english
        )


def run(settings: Settings, audio: pyaudio.PyAudio) -> None:
    """Load the models, then capture and print phrases until Ctrl+C."""
    models = load_models()
    clock = SessionClock()
    # Phrase times are session seconds; this turns them into time of day.
    pipeline = Pipeline(models, started_at=datetime.now())
    captures = [
        AudioCapture(audio, device, source, clock, pipeline.frames[source])
        for device, source in (
            (settings.mic, Source.MIC),
            (settings.loopback, Source.SYSTEM),
        )
    ]
    pipeline.start()
    try:
        for capture in captures:
            capture.start()
        print_message(
            f"Listening to '{settings.mic.name}' and '{settings.loopback.name}'. "
            "Press Ctrl+C to stop."
        )
        while True:
            # A short sleep, unlike waiting on an event, lets Ctrl+C through
            # on Windows.
            time.sleep(WAIT_FOR_CTRL_C_S)
    except KeyboardInterrupt:
        print_message("Stopping…")
    finally:
        for capture in captures:
            capture.stop()
        pipeline.stop()

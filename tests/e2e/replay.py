"""Shared helpers for the end-to-end tests: replay recorded speech through the
full pipeline at real-time speed, and check it against PLAN.md's latency targets.
"""

import contextlib
import ctypes
import socket
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from asr_benchmark import DATA_DIR, Clip, error_rates
from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, SessionClock, Source
from live_hy_en.metrics import PhraseTiming, Stage, format_summary, p95
from live_hy_en.output import TranscriptEntry
from live_hy_en.pipeline import Pipeline
from live_hy_en.segmenter import VAD_MODEL_PATH, WINDOW_SAMPLES
from live_hy_en.speakers import MIC_LABEL, SPEAKER_MODEL_PATH
from live_hy_en.translator import MT_MODEL, MT_MODELS_DIR

needs_laptop = pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            VAD_MODEL_PATH,
            SPEAKER_MODEL_PATH,
            MT_MODELS_DIR / MT_MODEL,
            DATA_DIR / "test.tar.gz",
        )
    ),
    reason="needs the downloaded models and FLEURS data on the laptop",
)

GAP_S = 1.5
CHUNK_S = WINDOW_SAMPLES / SAMPLE_RATE_HZ
# A clip's entries are those that start within it, allowing for VAD padding.
CLIP_MARGIN_S = 0.5
# PLAN.md §5.1: p95 latency target, inside the 5 s hard limit.
P95_TARGET_S = 3.5
# PLAN.md §2 latency budget per step; both queue waits share the queue's budget.
STAGE_BUDGETS_S = {
    "pause detection": (0.5, (Stage.DETECTED,)),
    "queue": (0.3, (Stage.TRANSCRIPTION_STARTED, Stage.TRANSLATION_STARTED)),
    "speaker": (0.2, (Stage.LABELLED,)),
    "transcription": (2.0, (Stage.TRANSCRIBED,)),
    "translation": (0.5, (Stage.TRANSLATED,)),
    "printing": (0.05, (Stage.SHOWN,)),
}
# Steps allowed over their budget, because the whole phrase still fits 5 s.
ACCEPTED_OVERRUNS = {
    # END_SILENCE_MS is 700 ms (commit 747817c) so pauses split fewer sentences.
    "pause detection",
    # NLLB-1.3B was chosen for accuracy (DECISIONS.md, translator.py 6).
    "translation",
}


@dataclass(frozen=True)
class PlacedClip:
    """One recording placed on a source's track, in seconds from replay start."""

    source: Source
    start_s: float
    end_s: float
    reference: str


@dataclass(frozen=True)
class ShownEntry:
    entry: TranscriptEntry
    start_s: float
    shown_s: float


def place_alternating(
    clips: Sequence[Clip], start_s: float = GAP_S
) -> tuple[list[PlacedClip], list[np.ndarray]]:
    """Place clips one after another; one in three is mine, the rest the call's."""
    placed, audio, cursor_s = [], [], start_s
    for index, clip in enumerate(clips):
        source = Source.MIC if index % 3 == 0 else Source.SYSTEM
        placed.append(
            PlacedClip(source, cursor_s, cursor_s + clip.duration_s, clip.reference)
        )
        audio.append(clip.samples)
        cursor_s += clip.duration_s + GAP_S
    return placed, audio


def build_tracks(placed: list[PlacedClip], audio: list[np.ndarray]) -> dict:
    """Return one track per source with every clip at its place."""
    length_s = max(clip.end_s for clip in placed) + GAP_S
    tracks = {
        source: np.zeros(round(length_s * SAMPLE_RATE_HZ), dtype=np.float32)
        for source in Source
    }
    for clip, samples in zip(placed, audio, strict=True):
        start = round(clip.start_s * SAMPLE_RATE_HZ)
        tracks[clip.source][start : start + samples.size] = samples
    return tracks


def replay(
    models, tracks: dict, progress_every_s: float | None = None
) -> tuple[list[ShownEntry], Pipeline, float]:
    """Feed the tracks in real time, as capture would; return what was shown.

    With `progress_every_s`, a line every so often shows a long replay is alive.
    """
    clock = SessionClock()
    shown: list[ShownEntry] = []
    replay_start_s = 0.0

    def show(entry: TranscriptEntry) -> None:
        start_s = (entry.spoken_at - started_at).total_seconds() - replay_start_s
        shown.append(ShownEntry(entry, start_s, clock.elapsed_s() - replay_start_s))

    started_at = datetime.now()
    pipeline = Pipeline(models, clock, started_at, show=show)
    pipeline.start()
    chunk = WINDOW_SAMPLES
    replay_start_s = clock.elapsed_s()
    length = len(tracks[Source.MIC])
    total_min = length / SAMPLE_RATE_HZ / 60
    next_progress_s = progress_every_s
    for index, offset in enumerate(range(0, length - chunk + 1, chunk)):
        due_s = replay_start_s + (index + 1) * CHUNK_S
        time.sleep(max(0.0, due_s - clock.elapsed_s()))
        if next_progress_s is not None and index * CHUNK_S >= next_progress_s:
            # flush: output piped to a file is otherwise held back until the end.
            print(
                f"Replayed {index * CHUNK_S / 60:.0f} of {total_min:.0f} min, "
                f"{len(shown)} phrases shown",
                flush=True,
            )
            next_progress_s += progress_every_s
        for source, track in tracks.items():
            samples = track[offset : offset + chunk]
            # Loopback sends nothing while the call is silent; the mic always does.
            if source is Source.SYSTEM and not samples.any():
                continue
            frame_start_s = clock.elapsed_s() - CHUNK_S
            pipeline.frames[source].put(AudioFrame(source, frame_start_s, samples))
    playback_end_s = clock.elapsed_s() - replay_start_s
    pipeline.stop()
    return shown, pipeline, playback_end_s


def entries_of(clip: PlacedClip, shown: list[ShownEntry]) -> list[ShownEntry]:
    """Return the shown entries that belong to one placed clip."""
    return [
        item
        for item in shown
        if clip.start_s - CLIP_MARGIN_S <= item.start_s <= clip.end_s + CLIP_MARGIN_S
        and (item.entry.speaker == MIC_LABEL) == (clip.source is Source.MIC)
    ]


def stage_p95_s(timings: Sequence[PhraseTiming]) -> dict[str, float]:
    """Return each budgeted step's p95 seconds over the phrases."""
    result = {}
    for name, (_, stages) in STAGE_BUDGETS_S.items():
        result[name] = p95(
            [
                sum(timing.step_durations_s().get(stage, 0.0) for stage in stages)
                for timing in timings
            ]
        )
    return result


def stages_over_budget(
    timings: Sequence[PhraseTiming], also_accepted: frozenset[str] = frozenset()
) -> list[str]:
    """Return the steps whose p95 is over budget and not an accepted overrun."""
    accepted = ACCEPTED_OVERRUNS | also_accepted
    return [
        name
        for name, seconds in stage_p95_s(timings).items()
        if seconds > STAGE_BUDGETS_S[name][0] and name not in accepted
    ]


def print_latency(pipeline: Pipeline) -> None:
    """Print the session summary and each step's p95 against its budget."""
    print(format_summary(pipeline.metrics.summary()))
    for name, seconds in stage_p95_s(pipeline.metrics.timings).items():
        budget_s = STAGE_BUDGETS_S[name][0]
        verdict = "ok" if seconds <= budget_s else "OVER"
        if verdict == "OVER" and name in ACCEPTED_OVERRUNS:
            verdict = "over, accepted"
        print(f"  {name:16} p95 {seconds:5.2f} s, budget {budget_s:.2f} s: {verdict}")


def report(title: str, placed, shown, pipeline) -> float:
    """Print each clip's entries, accuracy and latency; return the WER."""
    print(f"\n=== {title} ===")
    pairs = []
    for clip in placed:
        entries = entries_of(clip, shown)
        hypothesis = " ".join(item.entry.armenian for item in entries)
        pairs.append((clip.reference, hypothesis))
        print(
            f"[{clip.start_s:6.1f}-{clip.end_s:6.1f} s] {clip.source.value:6} "
            f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}: "
            + " | ".join(f"{i.entry.speaker}: {i.entry.english}" for i in entries)
        )
    wer, cer = error_rates(pairs)
    print(f"WER {wer:.1%}, CER {cer:.1%} over {len(placed)} clips")
    print_latency(pipeline)
    return wer


def assert_latency_targets(
    pipeline: Pipeline, limit_s: float, also_accepted: frozenset[str] = frozenset()
) -> None:
    """Every phrase within the hard limit, p95 on target, steps within budget."""
    summary = pipeline.metrics.summary()
    assert summary.max_s <= limit_s
    assert summary.p95_s <= P95_TARGET_S
    assert stages_over_budget(pipeline.metrics.timings, also_accepted) == []


def p95_latency_s(timings: Sequence[PhraseTiming]) -> float:
    """Return the p95 latency of some phrases."""
    return p95([timing.latency_s() for timing in timings])


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def process_memory_bytes() -> int | None:
    """Return this process's private memory on Windows, or None elsewhere."""
    if sys.platform != "win32":
        return None
    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    # Without these, 64-bit handles are cut to 32 bits and the call fails.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return counters.PagefileUsage


def block_network(patch: pytest.MonkeyPatch, attempts: list) -> None:
    """Make every network connection fail, recording each attempt."""

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("network blocked by the offline test")

    for owner, name in (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    ):
        patch.setattr(owner, name, refuse)


class _Tee:
    """A text stream that writes to the console and a log file at once."""

    def __init__(self, console, log_file) -> None:
        self._streams = (console, log_file)

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
            # A long run must show its progress as it happens, not at the end.
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def log_to(path: Path) -> Iterator[None]:
    """Show printed output live on the console and also save it, as UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("w", encoding="utf-8") as log_file,
        contextlib.redirect_stdout(_Tee(sys.stdout, log_file)),
    ):
        yield

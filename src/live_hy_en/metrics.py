"""Latency of each phrase through the pipeline, and a summary for the session."""

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from live_hy_en.capture import SessionClock, Source

# The brief's hard limit from the end of a phrase to its EN line.
MAX_LATENCY_S = 5.0
# Warn a little earlier, so a slowdown shows before it breaks the limit.
WARN_LATENCY_S = 4.0
P95_FRACTION = 0.95

logger = logging.getLogger(__name__)


class Stage(Enum):
    """Pipeline milestones, in order; each value names the step that ends there."""

    DETECTED = "pause detection"
    TRANSCRIPTION_STARTED = "queue"
    TRANSCRIBED = "transcription"
    LABELLED = "speaker"
    TRANSLATION_STARTED = "translation queue"
    TRANSLATED = "translation"
    SHOWN = "printing"


@dataclass
class PhraseTiming:
    """When one phrase ended and when it reached each stage, in session seconds."""

    source: Source
    start_s: float
    end_s: float
    marks: dict[Stage, float] = field(default_factory=dict)

    def latency_s(self) -> float:
        """Seconds from the end of the phrase until it was shown."""
        return self.marks[Stage.SHOWN] - self.end_s

    def step_durations_s(self) -> dict[Stage, float]:
        """Seconds spent in each step, keyed by the stage that ends it."""
        durations, previous_s = {}, self.end_s
        for stage in Stage:
            if stage in self.marks:
                durations[stage] = self.marks[stage] - previous_s
                previous_s = self.marks[stage]
        return durations


@dataclass(frozen=True)
class LatencySummary:
    """Latency over a session's shown phrases."""

    phrase_count: int
    p50_s: float
    p95_s: float
    max_s: float
    over_limit_count: int


def p95(values: Sequence[float]) -> float:
    """Return the 95th-percentile value (nearest rank)."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(P95_FRACTION * len(ordered)))]


def format_timing(timing: PhraseTiming) -> str:
    """Return one phrase's latency with the time spent in each step."""
    steps = ", ".join(
        f"{stage.value} {seconds:.2f}"
        for stage, seconds in timing.step_durations_s().items()
    )
    return (
        f"{timing.source.value} phrase at {timing.start_s:.1f} s: "
        f"{timing.latency_s():.2f} s ({steps})"
    )


def format_summary(summary: LatencySummary | None) -> str:
    """Return the session summary as one line for the terminal."""
    if summary is None:
        return "No phrases were shown, so there is no latency to report."
    phrases = "phrase" if summary.phrase_count == 1 else "phrases"
    return (
        f"Latency over {summary.phrase_count} {phrases}: "
        f"p50 {summary.p50_s:.2f} s, p95 {summary.p95_s:.2f} s, "
        f"max {summary.max_s:.2f} s; {summary.over_limit_count} over the "
        f"{MAX_LATENCY_S:g} s limit."
    )


class Metrics:
    """Timestamp phrases at each stage and collect their latency."""

    def __init__(self, clock: SessionClock) -> None:
        self._clock = clock
        self._latencies_s: list[float] = []

    def start(self, source: Source, start_s: float, end_s: float) -> PhraseTiming:
        """Begin timing a phrase the moment its end has been detected."""
        timing = PhraseTiming(source, start_s, end_s)
        self.mark(timing, Stage.DETECTED)
        return timing

    def mark(self, timing: PhraseTiming, stage: Stage) -> None:
        """Record that the phrase reached a stage now."""
        timing.marks[stage] = self._clock.elapsed_s()

    def finish(self, timing: PhraseTiming) -> None:
        """Record that the phrase was shown, and log its latency."""
        self.mark(timing, Stage.SHOWN)
        latency_s = timing.latency_s()
        self._latencies_s.append(latency_s)
        if latency_s > WARN_LATENCY_S:
            logger.warning(
                "Slow phrase, limit %g s: %s", MAX_LATENCY_S, format_timing(timing)
            )
        else:
            logger.debug(format_timing(timing))

    def summary(self) -> LatencySummary | None:
        """Return the latency over all finished phrases, or None if there were none."""
        if not self._latencies_s:
            return None
        return LatencySummary(
            phrase_count=len(self._latencies_s),
            p50_s=statistics.median(self._latencies_s),
            p95_s=p95(self._latencies_s),
            max_s=max(self._latencies_s),
            over_limit_count=sum(
                latency_s > MAX_LATENCY_S for latency_s in self._latencies_s
            ),
        )

"""Tests for latency metrics (PLAN.md §5.1: latency ≤ 5 s, measured by Metrics)."""

import logging

import pytest

from live_hy_en.capture import SessionClock, Source
from live_hy_en.metrics import (
    MAX_LATENCY_S,
    WARN_LATENCY_S,
    Metrics,
    Stage,
    format_summary,
    format_timing,
    p95,
)


class ManualClock:
    """A session clock whose time the test sets."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self.clock = SessionClock(now=lambda: self.now_s)


@pytest.fixture
def manual():
    return ManualClock()


def _phrase(manual: ManualClock, metrics: Metrics, end_s: float, steps_s):
    """Run one phrase through every stage, each step taking the given seconds."""
    manual.now_s = end_s + steps_s[0]
    timing = metrics.start(Source.SYSTEM, end_s - 2.0, end_s)
    stages = [stage for stage in Stage if stage not in (Stage.DETECTED, Stage.SHOWN)]
    for stage, step_s in zip(stages, steps_s[1:], strict=False):
        manual.now_s += step_s
        metrics.mark(timing, stage)
    manual.now_s += steps_s[-1]
    metrics.finish(timing)
    return timing


# Pause detection, queue, transcription, speaker, translation queue,
# translation, printing.
TYPICAL_STEPS_S = (0.5, 0.0, 0.7, 0.1, 0.0, 0.3, 0.0)


def test_latency_runs_from_the_end_of_the_phrase_to_printing(manual):
    metrics = Metrics(manual.clock)
    timing = _phrase(manual, metrics, 10.0, TYPICAL_STEPS_S)
    assert timing.latency_s() == pytest.approx(sum(TYPICAL_STEPS_S))


def test_each_step_is_timed_separately(manual):
    metrics = Metrics(manual.clock)
    timing = _phrase(manual, metrics, 10.0, TYPICAL_STEPS_S)
    durations = timing.step_durations_s()
    assert list(durations) == list(Stage)
    assert list(durations.values()) == pytest.approx(list(TYPICAL_STEPS_S))


def test_summary_reports_p50_p95_max_and_phrases_over_the_limit(manual):
    metrics = Metrics(manual.clock)
    for index, transcription_s in enumerate([0.5, 1.0, 1.5, 2.0, 5.5]):
        steps_s = (0.5, 0.0, transcription_s, 0.0, 0.0, 0.0, 0.0)
        _phrase(manual, metrics, 10.0 * (index + 1), steps_s)
    summary = metrics.summary()
    assert summary.phrase_count == 5
    assert summary.p50_s == pytest.approx(2.0)
    assert summary.p95_s == pytest.approx(6.0)
    assert summary.max_s == pytest.approx(6.0)
    assert summary.over_limit_count == 1


def test_shown_phrases_keep_their_stage_timings_for_tests(manual):
    metrics = Metrics(manual.clock)
    first = _phrase(manual, metrics, 10.0, TYPICAL_STEPS_S)
    second = _phrase(manual, metrics, 20.0, TYPICAL_STEPS_S)
    assert metrics.timings == [first, second]
    metrics.timings.clear()  # a copy: callers can't change the record
    assert len(metrics.timings) == 2


def test_no_phrases_means_no_summary(manual):
    assert Metrics(manual.clock).summary() is None
    assert "No phrases" in format_summary(None)


def test_slow_phrase_is_warned_about_before_the_limit(manual, caplog):
    metrics = Metrics(manual.clock)
    slow_transcription_s = WARN_LATENCY_S  # plus 0.5 s pause detection
    _phrase(manual, metrics, 10.0, (0.5, 0.0, slow_transcription_s, 0, 0, 0, 0))
    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert f"limit {MAX_LATENCY_S:g} s" in record.getMessage()
    assert "transcription 4.00" in record.getMessage()


def test_normal_phrase_is_only_in_the_debug_view(manual, caplog):
    caplog.set_level(logging.DEBUG, logger="live_hy_en.metrics")
    metrics = Metrics(manual.clock)
    _phrase(manual, metrics, 10.0, TYPICAL_STEPS_S)
    [record] = caplog.records
    assert record.levelno == logging.DEBUG
    assert record.getMessage().startswith("system phrase at 8.0 s: 1.60 s (")


def test_timing_line_names_every_step(manual):
    timing = _phrase(manual, Metrics(manual.clock), 10.0, TYPICAL_STEPS_S)
    line = format_timing(timing)
    for stage in Stage:
        assert stage.value in line


def test_summary_line_is_readable(manual):
    metrics = Metrics(manual.clock)
    _phrase(manual, metrics, 10.0, TYPICAL_STEPS_S)
    assert format_summary(metrics.summary()) == (
        "Latency over 1 phrase: p50 1.60 s, p95 1.60 s, max 1.60 s; "
        "0 over the 5 s limit."
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1.0], 1.0), (list(range(1, 21)), 20), (list(range(1, 101)), 96)],
)
def test_p95_is_the_nearest_rank(values, expected):
    assert p95(values) == expected

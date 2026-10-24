"""Tests for terminal output (PLAN.md §5.1: time, speaker, HY and EN lines)."""

import io
import sys
from datetime import datetime

import pytest

from live_hy_en.output import (
    TranscriptEntry,
    configure_terminal,
    format_entry,
    print_entry,
    print_error,
    print_message,
)

ENTRY = TranscriptEntry(
    spoken_at=datetime(2026, 10, 24, 14, 32, 7, 600_000),
    speaker="Speaker 1",
    armenian="Բարև, ինչպե՞ս ես։",
    english="Hi, how are you?",
)


def test_entry_has_time_speaker_armenian_and_english():
    assert format_entry(ENTRY).splitlines() == [
        "[14:32:07] Speaker 1",
        "HY: Բարև, ինչպե՞ս ես։",
        "EN: Hi, how are you?",
    ]


def test_my_entries_are_labelled_me():
    mine = TranscriptEntry(ENTRY.spoken_at, "Me", "Լավ եմ։", "I'm fine.")
    assert format_entry(mine).startswith("[14:32:07] Me\n")


def test_entries_are_separated_by_a_blank_line(capsys):
    print_entry(ENTRY)
    print_entry(ENTRY)
    output = capsys.readouterr().out
    assert output == format_entry(ENTRY) + "\n" + format_entry(ENTRY) + "\n"


def _use_ansi_terminal(monkeypatch) -> io.BytesIO:
    """Point standard streams at Windows' ANSI code page, as when redirected."""
    stdout = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(stdout, encoding="cp1252"))
    monkeypatch.setattr(
        sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    )
    return stdout


def test_armenian_prints_correctly_after_configuring_the_terminal(monkeypatch):
    # Patched inside the test: pytest re-installs its capture after fixtures.
    stdout = _use_ansi_terminal(monkeypatch)
    configure_terminal()
    print_entry(ENTRY)
    assert stdout.getvalue().decode("utf-8") == format_entry(ENTRY) + "\n"


def test_without_configuring_armenian_would_not_print(monkeypatch):
    _use_ansi_terminal(monkeypatch)
    with pytest.raises(UnicodeEncodeError):
        print_entry(ENTRY)


def test_messages_go_to_standard_output(capsys):
    print_message("Listening…")
    assert capsys.readouterr().out == "Listening…\n"


def test_errors_go_to_standard_error(capsys):
    print_error("No microphone matches 'usb'.")
    captured = capsys.readouterr()
    assert captured.err == "Error: No microphone matches 'usb'.\n"
    assert captured.out == ""

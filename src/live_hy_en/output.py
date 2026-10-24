"""Terminal output: transcript entries, status lines, errors and the device list."""

import sys
from dataclasses import dataclass
from datetime import datetime

TIME_FORMAT = "%H:%M:%S"
TERMINAL_ENCODING = "utf-8"


@dataclass(frozen=True)
class TranscriptEntry:
    """One finished phrase: when it was said, who said it, and what it means."""

    spoken_at: datetime
    speaker: str
    armenian: str
    english: str


def configure_terminal() -> None:
    """Write UTF-8 so Armenian prints correctly, also when output is redirected."""
    for stream in (sys.stdout, sys.stderr):
        # A Windows console accepts Unicode already; a pipe or file would
        # otherwise get the ANSI code page, which has no Armenian letters.
        stream.reconfigure(encoding=TERMINAL_ENCODING, errors="replace")


def format_entry(entry: TranscriptEntry) -> str:
    """Return the entry as its time-and-speaker line, HY line and EN line."""
    return (
        f"[{entry.spoken_at.strftime(TIME_FORMAT)}] {entry.speaker}\n"
        f"HY: {entry.armenian}\n"
        f"EN: {entry.english}\n"
    )


def print_entry(entry: TranscriptEntry) -> None:
    """Print one entry at once, followed by a blank line."""
    print(format_entry(entry), flush=True)


def print_message(message: str) -> None:
    """Print a message for the user, e.g. a status line or the device list."""
    print(message, flush=True)


def print_error(message: str) -> None:
    """Print an error for the user on standard error."""
    print(f"Error: {message}", file=sys.stderr, flush=True)

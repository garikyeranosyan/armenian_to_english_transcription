"""Speaker benchmark: how often call-side voices are merged into one label or split.

Calls are simulated from Common Voice hy-AM, which names the speaker of every
recording: a few speakers take turns, each recording goes through the Opus codec
at a voice-call bitrate and the program's own VAD, and SpeakerTracker labels the
phrases as it does live. Download Common Voice Armenian from Mozilla Data
Collective into benchmarks/data/common_voice_hy/, then run from the project folder:
    python benchmarks/speaker_benchmark.py
    python benchmarks/speaker_benchmark.py --thresholds 0.3 0.4 0.5
"""

import argparse
import csv
import hashlib
import io
import random
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile

# Run as a script, so the program's own package isn't on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_benchmark import read_wav  # noqa: E402
from live_hy_en.capture import SAMPLE_RATE_HZ, AudioFrame, Source  # noqa: E402
from live_hy_en.segmenter import (  # noqa: E402
    WINDOW_SAMPLES,
    Segmenter,
    SileroVad,
    SpeechSegment,
)
from live_hy_en.speakers import (  # noqa: E402
    SpeakerTracker,
    VoiceEncoder,
    WeSpeakerEncoder,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data" / "common_voice_hy"
CV_SPLIT_FILE = "test.tsv"
DEFAULT_THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
SPEAKER_COUNT = 40
CLIPS_PER_SPEAKER = 8
CALL_COUNT = 60
# Remote speakers per simulated call, cycled through.
SPEAKERS_PER_CALL = (2, 3, 4)
TURNS_PER_CALL = 16
# Share of turns that are short replies ("yes", "okay"), cut from a phrase.
SHORT_REPLY_SHARE = 0.2
SHORT_REPLY_S = 0.7
# Shorter phrases are left out, so short replies come only from SHORT_REPLY_SHARE.
MIN_PHRASE_S = 1.0
# libsndfile's Opus setting for about 28 kbit/s at 16 kHz, a typical call bitrate.
CALL_CODEC_LEVEL = 0.9
SEED = 0


@dataclass(frozen=True)
class Recording:
    """One Common Voice recording and who spoke it."""

    path: str
    speaker_id: str


@dataclass(frozen=True)
class Turn:
    """One call-side phrase and who really spoke it."""

    speaker_id: str
    segment: SpeechSegment


@dataclass(frozen=True)
class Score:
    """How the labels of some calls compare with the real speakers."""

    phrase_count: int
    merged_count: int
    split_count: int
    speaker_count: int
    label_count: int

    @property
    def merged_rate(self) -> float:
        """Share of phrases under a label that mostly belongs to someone else."""
        return self.merged_count / max(self.phrase_count, 1)

    @property
    def split_rate(self) -> float:
        """Share of phrases away from their speaker's main label."""
        return self.split_count / max(self.phrase_count, 1)


def parse_cv_tsv(text: str) -> list[Recording]:
    """Return the recordings listed in a Common Voice split file."""
    rows = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    return [Recording(row["path"], row["client_id"]) for row in rows]


def choose_recordings(
    recordings: Sequence[Recording], speaker_count: int, clips_per_speaker: int
) -> list[Recording]:
    """Return the first clips of the speakers with the most recordings."""
    by_speaker: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_speaker[recording.speaker_id].append(recording)
    eligible = [
        clips for clips in by_speaker.values() if len(clips) >= clips_per_speaker
    ]
    eligible.sort(key=len, reverse=True)
    return [
        recording
        for clips in eligible[:speaker_count]
        for recording in clips[:clips_per_speaker]
    ]


def load_recordings(
    archive_path: Path, speaker_count: int, clips_per_speaker: int
) -> dict[Recording, np.ndarray]:
    """Read the chosen speakers' audio from the Common Voice archive."""
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if member.name.endswith(f"/{CV_SPLIT_FILE}"):
                text = archive.extractfile(member).read().decode("utf-8")
                break
        else:
            raise FileNotFoundError(f"No {CV_SPLIT_FILE} in {archive_path}")
    chosen = choose_recordings(parse_cv_tsv(text), speaker_count, clips_per_speaker)
    wanted = {recording.path: recording for recording in chosen}
    audio = {}
    with tarfile.open(archive_path) as archive:
        for member in archive:
            recording = wanted.get(Path(member.name).name)
            if recording is not None and "/clips/" in member.name:
                audio[recording] = read_wav(archive.extractfile(member).read())
    return audio


def to_call_quality(samples: np.ndarray) -> np.ndarray:
    """Return the audio as it sounds after a voice call's Opus codec."""
    buffer = io.BytesIO()
    soundfile.write(
        buffer,
        samples,
        SAMPLE_RATE_HZ,
        format="OGG",
        subtype="OPUS",
        compression_level=CALL_CODEC_LEVEL,
    )
    buffer.seek(0)
    decoded, _ = soundfile.read(buffer, dtype="float32")
    return decoded


def segment_recording(samples: np.ndarray, segmenter: Segmenter) -> list[SpeechSegment]:
    """Cut one recording into phrases with the program's own segmenter."""
    segments = []
    for offset in range(0, samples.size - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        frame = AudioFrame(
            Source.SYSTEM,
            offset / SAMPLE_RATE_HZ,
            samples[offset : offset + WINDOW_SAMPLES],
        )
        segments += segmenter.process(frame)
    return segments + segmenter.flush()


def shorten(segment: SpeechSegment, speech_s: float) -> SpeechSegment:
    """Return a short reply: the middle `speech_s` of the phrase's speech."""
    speech = segment.speech_samples
    length = round(speech_s * SAMPLE_RATE_HZ)
    start = max((speech.size - length) // 2, 0)
    return replace(
        segment,
        end_s=segment.start_s + min(length, speech.size) / SAMPLE_RATE_HZ,
        samples=speech[start : start + length],
        lead_in_samples=0,
    )


def simulate_calls(
    phrases: dict[str, list[SpeechSegment]],
    call_count: int,
    turns_per_call: int,
    rng: random.Random,
) -> list[list[Turn]]:
    """Return calls in which a few speakers take turns, never twice in a row."""
    speaker_ids = sorted(phrases)
    calls = []
    for index in range(call_count):
        size = SPEAKERS_PER_CALL[index % len(SPEAKERS_PER_CALL)]
        cast = rng.sample(speaker_ids, size)
        turns, previous = [], None
        for _ in range(turns_per_call):
            speaker_id = rng.choice([s for s in cast if s != previous])
            segment = rng.choice(phrases[speaker_id])
            if rng.random() < SHORT_REPLY_SHARE:
                segment = shorten(segment, SHORT_REPLY_S)
            turns.append(Turn(speaker_id, segment))
            previous = speaker_id
        calls.append(turns)
    return calls


def score_call(speaker_ids: Sequence[str], labels: Sequence[str]) -> Score:
    """Compare one call's labels with who really spoke each phrase."""
    by_label: dict[str, Counter] = defaultdict(Counter)
    by_speaker: dict[str, Counter] = defaultdict(Counter)
    for speaker_id, label in zip(speaker_ids, labels, strict=True):
        by_label[label][speaker_id] += 1
        by_speaker[speaker_id][label] += 1
    merged = sum(sum(c.values()) - max(c.values()) for c in by_label.values())
    split = sum(sum(c.values()) - max(c.values()) for c in by_speaker.values())
    return Score(len(labels), merged, split, len(by_speaker), len(by_label))


def add_scores(scores: Sequence[Score]) -> Score:
    """Return the scores of several calls taken together."""
    return Score(
        *(sum(getattr(s, field) for s in scores) for field in Score.__annotations__)
    )


class CachedEncoder:
    """Computes each phrase's voiceprint once, however many thresholds are tried."""

    def __init__(self, encoder: VoiceEncoder) -> None:
        self._encoder = encoder
        self._voiceprints: dict[bytes, np.ndarray] = {}

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Return the voiceprint, from the cache when this audio was seen before."""
        key = hashlib.sha1(samples.tobytes()).digest()
        if key not in self._voiceprints:
            self._voiceprints[key] = self._encoder.embed(samples)
        return self._voiceprints[key]


def evaluate(
    calls: Sequence[Sequence[Turn]], encoder: VoiceEncoder, threshold: float
) -> Score:
    """Label every call with a fresh tracker, as a new session would."""
    scores = []
    for call in calls:
        tracker = SpeakerTracker(encoder, similarity_threshold=threshold)
        labels = [tracker.assign_speaker(turn.segment) for turn in call]
        scores.append(score_call([turn.speaker_id for turn in call], labels))
    return add_scores(scores)


def format_table(results: Sequence[tuple[float, Score]], call_count: int) -> str:
    """Return the results per threshold as a Markdown table for the README."""
    lines = [
        "| Threshold | Merged | Split | Labels per call | Speakers per call |",
        "|---|---|---|---|---|",
    ]
    for threshold, score in results:
        lines.append(
            f"| {threshold:.2f} | {score.merged_rate:.1%} | {score.split_rate:.1%} "
            f"| {score.label_count / call_count:.2f} "
            f"| {score.speaker_count / call_count:.2f} |"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Simulate the calls once, then label them at every threshold."""
    args = parse_args(argv)
    archives = sorted(DATA_DIR.glob("*.tar.gz"))
    if not archives:
        raise SystemExit(f"Put the Common Voice hy-AM .tar.gz in {DATA_DIR}")
    audio = load_recordings(archives[0], SPEAKER_COUNT, CLIPS_PER_SPEAKER)
    detector = SileroVad()
    phrases: dict[str, list[SpeechSegment]] = defaultdict(list)
    for recording, samples in audio.items():
        detector.reset()
        segments = segment_recording(
            to_call_quality(samples), Segmenter(detector, Source.SYSTEM)
        )
        phrases[recording.speaker_id] += [
            s for s in segments if s.end_s - s.start_s >= MIN_PHRASE_S
        ]
    phrases = {speaker: found for speaker, found in phrases.items() if found}
    print(f"{len(phrases)} speakers, {sum(map(len, phrases.values()))} phrases")
    calls = simulate_calls(phrases, CALL_COUNT, TURNS_PER_CALL, random.Random(SEED))
    encoder = CachedEncoder(WeSpeakerEncoder())
    results = [(t, evaluate(calls, encoder, t)) for t in args.thresholds]
    print("\n" + format_table(results, len(calls)))


if __name__ == "__main__":
    main()

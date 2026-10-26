"""M1 ASR benchmark: Armenian accuracy and speed of Whisper models on this laptop.

Run from the project folder:
    python benchmarks/asr_benchmark.py
    python benchmarks/asr_benchmark.py --models large-v3-turbo --limit 50
"""

import argparse
import csv
import io
import os
import re
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile
import soxr
from dotenv import load_dotenv

# Run as a script, so the program's own package isn't on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live_hy_en.capture import SAMPLE_RATE_HZ  # noqa: E402
from live_hy_en.metrics import p95  # noqa: E402
from live_hy_en.transcriber import (  # noqa: E402
    FasterWhisperRecognizer,
    is_armenian,
)

FLEURS_BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
FLEURS_URL = f"{FLEURS_BASE_URL}/hy_am"
FLEURS_SPLIT = "test"
BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCHMARK_DIR / "data" / "fleurs_hy_am"
RESULTS_DIR = BENCHMARK_DIR / "results"
# The token lives in the project's .env, which git ignores (see .env.example).
ENV_PATH = BENCHMARK_DIR.parent / ".env"
HF_TOKEN_NAME = "HF_TOKEN"
DEFAULT_MODELS = ["small", "medium", "large-v3-turbo", "large-v3"]
DEFAULT_CLIP_LIMIT = 200
# Transcription share of the 5 s latency budget (PLAN.md §2).
ASR_BUDGET_S = 2.0

# Armenian marks written inside a word (e.g. the question mark in "ինչպե՞ս").
IN_WORD_MARKS = re.compile("[՚-՟]")
NON_WORD = re.compile(r"[^\w\s]|_")

Transcribe = Callable[[np.ndarray], str]


@dataclass(frozen=True)
class Clip:
    """One test recording with its reference transcript."""

    clip_id: str
    samples: np.ndarray
    reference: str

    @property
    def duration_s(self) -> float:
        """Length of the recording in seconds."""
        return self.samples.size / SAMPLE_RATE_HZ


@dataclass(frozen=True)
class ClipResult:
    """One model's transcript of one clip and how long it took."""

    clip: Clip
    hypothesis: str
    elapsed_s: float


@dataclass(frozen=True)
class Summary:
    """Accuracy and speed of one model over all clips."""

    model: str
    clip_count: int
    wer: float
    cer: float
    wrong_script_count: int
    mean_s: float
    p95_s: float
    max_s: float
    over_budget_count: int
    real_time_factor: float


def normalize(text: str) -> str:
    """Lower-case Armenian text without punctuation, for fair error counts."""
    text = IN_WORD_MARKS.sub("", text.casefold())
    # "և" is also written as the two letters "եւ" or "եվ" (Whisper's usual
    # spelling); count all three as one, so spelling isn't scored as an error.
    text = text.replace("եւ", "և").replace("եվ", "և")
    return " ".join(NON_WORD.sub(" ", text).split())


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return the Levenshtein distance between two sequences."""
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, 1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_rates(pairs: Sequence[tuple[str, str]]) -> tuple[float, float]:
    """Return corpus WER and CER for (reference, hypothesis) pairs."""
    word_errors = word_total = char_errors = char_total = 0
    for reference, hypothesis in pairs:
        ref, hyp = normalize(reference), normalize(hypothesis)
        word_errors += edit_distance(ref.split(), hyp.split())
        word_total += len(ref.split())
        char_errors += edit_distance(ref, hyp)
        char_total += len(ref)
    return word_errors / max(word_total, 1), char_errors / max(char_total, 1)


def parse_fleurs_tsv(text: str) -> dict[str, tuple[str, str]]:
    """Map each audio file name to its (clip id, raw transcript)."""
    rows = {}
    for line in text.splitlines():
        if line.strip():
            clip_id, file_name, raw_transcript = line.split("\t")[:3]
            rows[file_name] = (clip_id, raw_transcript)
    return rows


def read_wav(data: bytes) -> np.ndarray:
    """Decode WAV (or MP3) bytes to mono float32 at SAMPLE_RATE_HZ."""
    audio, rate_hz = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    samples = audio.mean(axis=1)
    if rate_hz != SAMPLE_RATE_HZ:
        samples = soxr.resample(samples, rate_hz, SAMPLE_RATE_HZ)
    return samples.astype(np.float32)


def load_hf_token(env_path: Path = ENV_PATH) -> str | None:
    """Return the Hugging Face token from .env or the environment, if one is set."""
    # An HF_TOKEN already set in the environment wins over the .env file.
    load_dotenv(env_path)
    return os.environ.get(HF_TOKEN_NAME, "").strip() or None


def download(url: str, target: Path, token: str | None) -> None:
    """Download a file, sending the Hugging Face token when there is one."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as reply:
        with partial.open("wb") as file:
            shutil.copyfileobj(reply, file)
    partial.replace(target)


def download_fleurs(token: str | None, data_dir: Path = DATA_DIR) -> None:
    """Download the FLEURS Armenian test split once."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in (f"{FLEURS_SPLIT}.tsv", f"{FLEURS_SPLIT}.tar.gz"):
        target = data_dir / name
        if not target.is_file():
            remote = name if name.endswith(".tsv") else f"audio/{name}"
            print(f"Downloading {name} ...", flush=True)
            download(f"{FLEURS_URL}/{remote}", target, token)


def load_fleurs(limit: int, data_dir: Path = DATA_DIR) -> list[Clip]:
    """Return up to `limit` clips from the downloaded FLEURS test split."""
    rows = parse_fleurs_tsv(
        (data_dir / f"{FLEURS_SPLIT}.tsv").read_text(encoding="utf-8")
    )
    clips = []
    with tarfile.open(data_dir / f"{FLEURS_SPLIT}.tar.gz") as archive:
        for member in archive:
            file_name = Path(member.name).name
            if member.isfile() and file_name in rows:
                clip_id, reference = rows[file_name]
                samples = read_wav(archive.extractfile(member).read())
                clips.append(Clip(clip_id, samples, reference))
                if len(clips) == limit:
                    break
    return clips


def evaluate(transcribe: Transcribe, clips: Sequence[Clip]) -> list[ClipResult]:
    """Transcribe every clip, timing each one; the first call warms up."""
    transcribe(clips[0].samples)
    results = []
    for clip in clips:
        start_s = time.perf_counter()
        hypothesis = transcribe(clip.samples)
        results.append(ClipResult(clip, hypothesis, time.perf_counter() - start_s))
    return results


def summarize(model: str, results: Sequence[ClipResult]) -> Summary:
    """Combine one model's clip results into accuracy and speed figures."""
    wer, cer = error_rates([(r.clip.reference, r.hypothesis) for r in results])
    times_s = sorted(r.elapsed_s for r in results)
    return Summary(
        model=model,
        clip_count=len(results),
        wer=wer,
        cer=cer,
        wrong_script_count=sum(not is_armenian(r.hypothesis) for r in results),
        mean_s=statistics.fmean(times_s),
        p95_s=p95(times_s),
        max_s=times_s[-1],
        over_budget_count=sum(t > ASR_BUDGET_S for t in times_s),
        real_time_factor=sum(times_s) / sum(r.clip.duration_s for r in results),
    )


def format_table(summaries: Sequence[Summary]) -> str:
    """Return the summaries as a Markdown table for the README."""
    lines = [
        "| Model | Clips | WER | CER | Wrong script | Mean s | p95 s | Max s "
        f"| > {ASR_BUDGET_S:g} s | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {s.clip_count} | {s.wer:.1%} | {s.cer:.1%} "
            f"| {s.wrong_script_count} | {s.mean_s:.2f} | {s.p95_s:.2f} "
            f"| {s.max_s:.2f} | {s.over_budget_count} | {s.real_time_factor:.3f} |"
        )
    return "\n".join(lines)


def model_label(model: str, compute_type: str | None) -> str:
    """Return a short name for results: a local model's folder name, plus the
    compute type when one was chosen, so runs of one model don't overwrite."""
    name = Path(model).name if Path(model).is_dir() else model.replace("/", "_")
    return f"{name}_{compute_type}" if compute_type else name


def write_transcripts(label: str, results: Sequence[ClipResult]) -> Path:
    """Save each clip's reference and transcript, for review and the MT benchmark."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"asr_{label}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["clip_id", "duration_s", "elapsed_s", "reference", "hypothesis"]
        )
        for r in results:
            writer.writerow(
                [
                    r.clip.clip_id,
                    f"{r.clip.duration_s:.2f}",
                    f"{r.elapsed_s:.3f}",
                    r.clip.reference,
                    r.hypothesis,
                ]
            )
    return path


def load_whisper(
    model: str, device: str | None, compute_type: str | None, token: str | None
) -> Transcribe:
    """Load a model through the program's own recognizer, so the same code is timed."""
    return FasterWhisperRecognizer(
        model, device=device, compute_type=compute_type, token=token
    ).recognize


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="faster-whisper model names or local CTranslate2 model folders",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_CLIP_LIMIT)
    parser.add_argument("--device", help="cuda or cpu (default: the GPU if found)")
    parser.add_argument(
        "--compute-type", help="e.g. float16 or int8 (default: by device)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run every model on the same clips and print the comparison."""
    args = parse_args(argv)
    token = load_hf_token()
    if token is None:
        print(f"No {HF_TOKEN_NAME} in {ENV_PATH}; downloading without a token.")
    download_fleurs(token)
    clips = load_fleurs(args.limit)
    total_s = sum(clip.duration_s for clip in clips)
    print(f"{len(clips)} FLEURS hy_am clips, {total_s / 60:.1f} min of audio")
    summaries = []
    for model in args.models:
        label = model_label(model, args.compute_type)
        print(f"\n{label}: loading ...", flush=True)
        transcribe = load_whisper(model, args.device, args.compute_type, token)
        results = evaluate(transcribe, clips)
        summaries.append(summarize(label, results))
        print(f"{label}: transcripts saved to {write_transcripts(label, results)}")
    print("\n" + format_table(summaries))


if __name__ == "__main__":
    main()

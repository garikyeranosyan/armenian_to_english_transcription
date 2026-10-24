"""MT benchmark: Armenian-to-English quality and speed of the candidates on this laptop.

Uses the FLEURS hy_am clips already transcribed by the ASR benchmark, so run that
first. Each model translates the reference Armenian text (clean input) and the
ASR transcripts (real input); FLEURS en_us gives the English references.
Run from the project folder, after benchmarks/convert_mt.py:
    python benchmarks/mt_benchmark.py
    python benchmarks/mt_benchmark.py --models opus-mt-hy-en --asr-model small
"""

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import sacrebleu

# Run as a script, so the program's own package isn't on the path yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_benchmark import (  # noqa: E402
    BENCHMARK_DIR,
    FLEURS_BASE_URL,
    FLEURS_SPLIT,
    HF_TOKEN_NAME,
    RESULTS_DIR,
    download,
    load_hf_token,
    parse_fleurs_tsv,
)
from live_hy_en.metrics import p95  # noqa: E402
from live_hy_en.translator import (  # noqa: E402
    MT_MODELS,
    CTranslate2Translator,
    Translator,
)

ENGLISH_DIR = BENCHMARK_DIR / "data" / "fleurs_en_us"
DEFAULT_ASR_MODEL = "large-v3-turbo"
# Translation share of the 5 s latency budget (PLAN.md §2).
MT_BUDGET_S = 0.5
# chrF++ (character n-grams plus word bigrams), as reported for FLORES-200.
CHRF_WORD_ORDER = 2

Translate = Callable[[str], str]


@dataclass(frozen=True)
class Sentence:
    """One FLEURS sentence: Armenian reference, its ASR transcript, English."""

    clip_id: str
    armenian: str
    armenian_asr: str
    english: str


@dataclass(frozen=True)
class SentenceResult:
    """One model's translations of one sentence, and how long each took."""

    sentence: Sentence
    from_reference: str
    from_asr: str
    elapsed_s: tuple[float, float]


@dataclass(frozen=True)
class Summary:
    """Quality on clean and on ASR input, and speed, of one model."""

    model: str
    sentence_count: int
    chrf_clean: float
    bleu_clean: float
    chrf_asr: float
    bleu_asr: float
    mean_s: float
    p95_s: float
    max_s: float
    over_budget_count: int


def download_english(token: str | None, data_dir: Path = ENGLISH_DIR) -> Path:
    """Download the FLEURS en_us transcripts once (text only, no audio)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{FLEURS_SPLIT}.tsv"
    if not target.is_file():
        print(f"Downloading en_us {target.name} ...", flush=True)
        download(f"{FLEURS_BASE_URL}/en_us/{target.name}", target, token)
    return target


def english_by_id(tsv_text: str) -> dict[str, str]:
    """Map each FLEURS sentence id to its English text.

    FLEURS ids are shared across languages: they number the FLORES-200 sentences.
    """
    english = {}
    for clip_id, raw_transcript in parse_fleurs_tsv(tsv_text).values():
        english.setdefault(clip_id, raw_transcript)
    return english


def load_sentences(asr_csv: Path, english: dict[str, str]) -> list[Sentence]:
    """Pair each transcribed clip with its English reference, where FLEURS has one."""
    sentences = []
    with asr_csv.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            clip_id = row["clip_id"]
            if clip_id in english:
                sentences.append(
                    Sentence(
                        clip_id, row["reference"], row["hypothesis"], english[clip_id]
                    )
                )
    return sentences


def _timed(translate: Translate, text: str) -> tuple[str, float]:
    start_s = time.perf_counter()
    english = translate(text)
    return english, time.perf_counter() - start_s


def evaluate(
    translate: Translate, sentences: Sequence[Sentence]
) -> list[SentenceResult]:
    """Translate both inputs of every sentence, timing each; the first call warms up."""
    translate(sentences[0].armenian)
    results = []
    for sentence in sentences:
        from_reference, reference_s = _timed(translate, sentence.armenian)
        from_asr, asr_s = _timed(translate, sentence.armenian_asr)
        results.append(
            SentenceResult(sentence, from_reference, from_asr, (reference_s, asr_s))
        )
    return results


def corpus_scores(
    hypotheses: Sequence[str], references: Sequence[str]
) -> tuple[float, float]:
    """Return corpus chrF++ and BLEU for the translations."""
    chrf = sacrebleu.CHRF(word_order=CHRF_WORD_ORDER)
    return (
        chrf.corpus_score(hypotheses, [references]).score,
        sacrebleu.BLEU().corpus_score(hypotheses, [references]).score,
    )


def summarize(model: str, results: Sequence[SentenceResult]) -> Summary:
    """Combine one model's results into quality and speed figures."""
    references = [r.sentence.english for r in results]
    chrf_clean, bleu_clean = corpus_scores(
        [r.from_reference for r in results], references
    )
    chrf_asr, bleu_asr = corpus_scores([r.from_asr for r in results], references)
    times_s = [elapsed for r in results for elapsed in r.elapsed_s]
    return Summary(
        model=model,
        sentence_count=len(results),
        chrf_clean=chrf_clean,
        bleu_clean=bleu_clean,
        chrf_asr=chrf_asr,
        bleu_asr=bleu_asr,
        mean_s=statistics.fmean(times_s),
        p95_s=p95(times_s),
        max_s=max(times_s),
        over_budget_count=sum(t > MT_BUDGET_S for t in times_s),
    )


def format_table(summaries: Sequence[Summary]) -> str:
    """Return the summaries as a Markdown table for the README."""
    lines = [
        "| Model | Sentences | chrF++ clean | BLEU clean | chrF++ ASR | BLEU ASR "
        f"| Mean s | p95 s | Max s | > {MT_BUDGET_S:g} s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {s.sentence_count} | {s.chrf_clean:.1f} "
            f"| {s.bleu_clean:.1f} | {s.chrf_asr:.1f} | {s.bleu_asr:.1f} "
            f"| {s.mean_s:.2f} | {s.p95_s:.2f} | {s.max_s:.2f} "
            f"| {s.over_budget_count} |"
        )
    return "\n".join(lines)


def write_translations(model: str, results: Sequence[SentenceResult]) -> Path:
    """Save each sentence's inputs and translations, for manual review."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"mt_{model}.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "clip_id",
                "english_reference",
                "armenian_reference",
                "english_from_reference",
                "armenian_asr",
                "english_from_asr",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.sentence.clip_id,
                    r.sentence.english,
                    r.sentence.armenian,
                    r.from_reference,
                    r.sentence.armenian_asr,
                    r.from_asr,
                ]
            )
    return path


def load_translator(
    model: str, device: str | None, compute_type: str | None
) -> Translate:
    """Load a model through the program's own translator, so the same code is timed."""
    return Translator(
        CTranslate2Translator(model, device=device, compute_type=compute_type)
    ).translate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(MT_MODELS))
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help="whose transcripts in benchmarks/results/ to translate",
    )
    parser.add_argument("--device", help="cuda or cpu (default: the GPU if found)")
    parser.add_argument(
        "--compute-type", help="e.g. float16 or int8_float16 (default: by device)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run every model on the same sentences and print the comparison."""
    args = parse_args(argv)
    token = load_hf_token()
    if token is None:
        print(f"No {HF_TOKEN_NAME} found; downloading without a token.")
    english = english_by_id(download_english(token).read_text(encoding="utf-8"))
    sentences = load_sentences(RESULTS_DIR / f"asr_{args.asr_model}.csv", english)
    print(f"{len(sentences)} sentences, ASR input from {args.asr_model}")
    summaries = []
    for model in args.models:
        print(f"\n{model}: loading ...", flush=True)
        translate = load_translator(model, args.device, args.compute_type)
        results = evaluate(translate, sentences)
        summaries.append(summarize(model, results))
        print(f"{model}: translations saved to {write_translations(model, results)}")
    print("\n" + format_table(summaries))


if __name__ == "__main__":
    main()

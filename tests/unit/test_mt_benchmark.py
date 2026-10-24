"""Tests for the MT benchmark (PLAN.md §5.1: chrF/BLEU on clean text and ASR output)."""

import csv

import pytest

from mt_benchmark import (
    MT_BUDGET_S,
    Sentence,
    SentenceResult,
    corpus_scores,
    english_by_id,
    evaluate,
    format_table,
    load_sentences,
    summarize,
)

EN_TSV = (
    "1889\t10004.wav\tIt gave us the train.\tit gave us the train\tx\t1\tMALE\n"
    "1889\t10005.wav\tIt gave us the train!\tit gave us the train\tx\t1\tFEMALE\n"
    "1921\t10006.wav\tOn the other hand, ice.\ton the other hand ice\tx\t1\tMALE\n"
)


def _sentence(clip_id: str = "1", english: str = "Hello there.") -> Sentence:
    return Sentence(clip_id, "Բարև ձեզ։", "բարև ձես", english)


def test_english_references_are_keyed_by_the_shared_sentence_id():
    assert english_by_id(EN_TSV) == {
        "1889": "It gave us the train.",
        "1921": "On the other hand, ice.",
    }


def test_sentences_pair_asr_output_with_english(tmp_path):
    asr_csv = tmp_path / "asr_model.csv"
    with asr_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["clip_id", "duration_s", "elapsed_s", "reference", "hypothesis"]
        )
        writer.writerow(["1889", "6.66", "0.38", "Այն մեզ տվել է", "այն մեզ տվել է"])
        writer.writerow(["9999", "5.00", "0.30", "Ոչ մի", "ոչ մի"])
    assert load_sentences(asr_csv, english_by_id(EN_TSV)) == [
        Sentence("1889", "Այն մեզ տվել է", "այն մեզ տվել է", "It gave us the train.")
    ]


def test_evaluate_translates_clean_and_asr_input_after_one_warm_up():
    calls = []

    def translate(text: str) -> str:
        calls.append(text)
        return f"EN({text})"

    sentences = [_sentence("1"), Sentence("2", "Լավ եմ։", "լավեմ", "I'm fine.")]
    results = evaluate(translate, sentences)
    assert calls == ["Բարև ձեզ։", "Բարև ձեզ։", "բարև ձես", "Լավ եմ։", "լավեմ"]
    assert results[1].from_reference == "EN(Լավ եմ։)"
    assert results[1].from_asr == "EN(լավեմ)"
    assert all(elapsed >= 0 for r in results for elapsed in r.elapsed_s)


def test_perfect_translations_score_100_and_wrong_ones_score_low():
    references = ["It gave us the train.", "On the other hand, ice."]
    assert corpus_scores(references, references) == pytest.approx((100, 100))
    chrf, bleu = corpus_scores(["Bananas are yellow.", "Cats sleep."], references)
    assert chrf < 20
    assert bleu < 10


def test_summarize_scores_both_inputs_and_counts_slow_translations():
    sentence = _sentence(english="Hello there.")
    results = [
        SentenceResult(sentence, "Hello there.", "Goodbye.", (0.1, MT_BUDGET_S + 0.1)),
        SentenceResult(sentence, "Hello there.", "Hello there.", (0.2, 0.3)),
    ]
    summary = summarize("model", results)
    assert summary.sentence_count == 2
    assert summary.chrf_clean == pytest.approx(100)
    assert summary.chrf_asr < summary.chrf_clean
    assert summary.max_s == pytest.approx(MT_BUDGET_S + 0.1)
    assert summary.over_budget_count == 1


def test_format_table_has_one_row_per_model():
    sentence = _sentence()
    results = [SentenceResult(sentence, "Hello there.", "Hello.", (0.1, 0.2))]
    table = format_table([summarize("nllb", results), summarize("opus", results)])
    lines = table.splitlines()
    assert len(lines) == 4
    assert lines[2].startswith("| nllb | 1 | 100.0 |")
    assert f"> {MT_BUDGET_S:g} s" in lines[0]

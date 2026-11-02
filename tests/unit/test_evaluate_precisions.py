import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "evaluate_precisions.py"
spec = importlib.util.spec_from_file_location("evaluate_precisions", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)
error_rates = evaluate.error_rates


def test_perfect_transcripts():
    result = error_rates(["Hello, world!", "good morning"], ["hello world", "Good morning."])
    assert result["wer"] == 0 and result["cer"] == 0
    assert result["n"] == 2 and result["empty_references"] == 0


def test_hallucination_on_silence_counts_as_insertions():
    refs = ["hello world", ""]
    hyps = ["hello world", "thank you"]
    result = error_rates(refs, hyps)
    # 2 inserted words over 2 reference words; previously the silent row was dropped -> 0.
    assert result["wer"] == pytest.approx(100.0)
    assert result["cer"] == pytest.approx(100 * len("thank you") / len("hello world"))
    assert result["n"] == 2
    assert result["empty_references"] == 1
    assert result["empty_reference_nonempty_hypotheses"] == 1
    assert result["empty_reference_inserted_words"] == 2


def test_correct_silence_adds_no_errors():
    result = error_rates(["hello world", "", "  ...  "], ["hello world", "", "?!"])
    assert result["wer"] == 0 and result["cer"] == 0
    assert result["n"] == 3
    assert result["empty_references"] == 2  # punctuation-only normalizes to empty
    assert result["empty_reference_nonempty_hypotheses"] == 0
    assert result["empty_reference_inserted_words"] == 0


def test_all_empty_references_do_not_divide_by_zero():
    result = error_rates(["", "..."], ["uh huh", ""])
    assert result["wer"] is None and result["cer"] is None
    assert result["n"] == 2
    assert result["empty_references"] == 2
    assert result["empty_reference_nonempty_hypotheses"] == 1
    assert result["empty_reference_inserted_words"] == 2


def test_all_empty_and_silent_is_still_defined_as_none():
    result = error_rates([""], [""])
    assert result["wer"] is None and result["cer"] is None
    assert result["empty_reference_nonempty_hypotheses"] == 0


def test_corpus_rate_weights_by_reference_length():
    # 1 substitution in a 1-word ref + 0 errors in a 3-word ref -> 1/4, not mean(100%, 0%).
    result = error_rates(["cat", "a b c"], ["dog", "a b c"])
    assert result["wer"] == pytest.approx(25.0)

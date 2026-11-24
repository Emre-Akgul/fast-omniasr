"""tools/hotword_eval.py: what it counts."""

import importlib.util
from pathlib import Path

from fast_omniasr import BeamSearchDecoder

_SPEC = importlib.util.spec_from_file_location(
    "hotword_eval", Path(__file__).parents[2] / "tools" / "hotword_eval.py")
hotword_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hotword_eval)


def test_found_and_inserted_hotwords_are_counted_apart():
    counter = BeamSearchDecoder(hotwords=["Masa", "Bahçe"], hotword_max_suffix=5,
                                lang="tur_Latn")
    references = ["Masa'ya git.", "Bahçeye git.", "Eve dön."]
    hypotheses = ["masaya git", "bahça git", "masada dön"]
    assert hotword_eval.score(references, hypotheses, counter) == {
        "wer": 33.33, "found": 1, "said": 2, "inserted": 1}


def test_wer_ignores_case_and_punctuation():
    counter = BeamSearchDecoder(hotwords=["x"], lang="tur_Latn")
    assert hotword_eval.score(["IŞIK, Masa'ya!"], ["ışık masaya"], counter)["wer"] == 0.0

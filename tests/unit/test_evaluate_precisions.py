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


# --- LLM stop reasons are reported, not discarded (issue #20) ---

def test_stop_reason_summary_counts_and_lists_truncated():
    result = evaluate.stop_reason_summary(
        ["a", "b", "c", "d"], ["eos", "max_new_tokens", "eos", "context_limit"]
    )
    assert result["stop_reasons"] == {"eos": 2, "max_new_tokens": 1, "context_limit": 1}
    assert result["truncated"] == 2
    assert result["truncated_ids"] == ["b", "d"]


def test_stop_reason_summary_all_complete_reports_every_reason():
    result = evaluate.stop_reason_summary(["a"], ["eos"])
    assert result["stop_reasons"] == {"eos": 1, "max_new_tokens": 0, "context_limit": 0}
    assert result["truncated"] == 0 and result["truncated_ids"] == []


def test_stop_reason_summary_rejects_unknown_reasons():
    with pytest.raises(ValueError, match="unexpected stop reasons"):
        evaluate.stop_reason_summary(["a", "b"], ["eos", None])


def test_stop_reasons_come_from_the_runtime():
    from fast_omniasr.llm import STOP_REASONS

    assert evaluate.LLM_STOP_REASONS is STOP_REASONS
    assert STOP_REASONS[0] == "eos"


def _fake_llm_main(monkeypatch, tmp_path, reasons_by_precision):
    import json
    import sys
    import types

    class FakeResult:
        def __init__(self, text, stop_reason):
            self.text, self.stop_reason = text, stop_reason

    class FakeLLM:
        def __init__(self, path, backend, device, precision):
            self.reasons = iter(reasons_by_precision[precision])

        def transcribe(self, wav, sample_rate=16000, lang=None):
            return FakeResult("some text", next(self.reasons))

    monkeypatch.setitem(sys.modules, "fast_omniasr.llm", types.SimpleNamespace(OmniASRLLM=FakeLLM))
    import numpy as np

    rows = [{"id": f"u{i}", "wav": np.zeros(16000, np.float32)} for i in range(3)]
    monkeypatch.setattr(evaluate, "load_rows", lambda args: rows)
    monkeypatch.setattr(evaluate, "gpu_mib", lambda: 0)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_precisions.py", "--llm", str(tmp_path), "--audio", "x.wav",
        "--precision", *reasons_by_precision, "--report", str(report),
    ])
    evaluate.main()
    return json.loads(report.read_text())


def test_main_reports_llm_truncation(monkeypatch, tmp_path, capsys):
    # First reason of each precision is consumed by the warm-up call.
    report = _fake_llm_main(monkeypatch, tmp_path, {
        "fp32": ["eos", "eos", "eos", "eos"],
        "fp16": ["eos", "eos", "max_new_tokens", "context_limit"],
    })
    fp32, fp16 = report["precisions"]["fp32"], report["precisions"]["fp16"]
    assert fp32["truncated"] == 0
    assert fp16["stop_reasons"] == {"eos": 1, "max_new_tokens": 1, "context_limit": 1}
    assert fp16["truncated_ids"] == ["u1", "u2"]
    assert fp16["hypothesis_stop_reasons"] == {"u0": "eos", "u1": "max_new_tokens",
                                               "u2": "context_limit"}
    out, err = capsys.readouterr()
    assert "fp16: 2/3 transcripts did not end at EOS" in err
    assert "fp32" not in err
    # stdout summary lines carry counts but not the per-utterance maps
    assert '"truncated": 2' in out and "hypothesis_stop_reasons" not in out



# --- file inputs are normalized exactly once, like the public API (issue #22) ---

def _wav(tmp_path, rate=16000, channels=1):
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(0)
    data = (0.2 * rng.standard_normal((8000, channels) if channels > 1 else 8000) + 0.05)
    path = tmp_path / "clip.wav"
    sf.write(path, data.astype("float32"), rate, subtype="FLOAT")
    return path, data.astype("float32")


def test_load_rows_returns_raw_samples_and_rate(tmp_path):
    import argparse

    import numpy as np

    path, data = _wav(tmp_path)
    rows = evaluate.load_rows(argparse.Namespace(dataset=None, audio=[f"{path}:eng_Latn"]))
    assert rows[0]["sample_rate"] == 16000 and rows[0]["lang"] == "eng_Latn"
    np.testing.assert_array_equal(rows[0]["wav"], data)  # not pre-normalized


@pytest.mark.parametrize("kind", ["ctc", "llm"])
def test_file_rows_reach_the_model_normalized_once(tmp_path, monkeypatch, kind):
    import argparse
    import sys
    import types

    import numpy as np

    from fast_omniasr.audio import load_audio, prepare_audio

    path, _ = _wav(tmp_path)
    received = []

    class FakeCTC:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe_numpy(self, waveform, sample_rate=16000):
            received.append(prepare_audio(waveform, sample_rate))
            return types.SimpleNamespace(text="x")

    class FakeLLM:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, audio, *, sample_rate=16000, lang=None):
            received.append(prepare_audio(audio, sample_rate))
            return types.SimpleNamespace(text="x", stop_reason="eos")

    monkeypatch.setitem(sys.modules, "fast_omniasr.llm",
                        types.SimpleNamespace(OmniASRLLM=FakeLLM))
    import fast_omniasr

    monkeypatch.setattr(fast_omniasr, "OmniASR", FakeCTC)
    args = argparse.Namespace(dataset=None, audio=[str(path)], device="cpu", lang=None,
                              llm=tmp_path if kind == "llm" else None,
                              ctc=None if kind == "llm" else tmp_path)
    row = evaluate.load_rows(args)[0]
    evaluate.transcriber(args, "fp32")(row)
    # Exactly what the public file API feeds the model (one normalization) ...
    np.testing.assert_array_equal(received[0], load_audio(path))
    # ... which differs from normalizing twice, the old behaviour.
    assert not np.array_equal(received[0], prepare_audio(load_audio(path)[0]))


def test_non_16k_and_stereo_files_still_fail_clearly(tmp_path):
    import argparse

    from fast_omniasr.audio import prepare_audio

    for kwargs, message in (({"rate": 8000}, "16 kHz"), ({"channels": 2}, "mono")):
        path, _ = _wav(tmp_path, **kwargs)
        row = evaluate.load_rows(argparse.Namespace(dataset=None, audio=[str(path)]))[0]
        with pytest.raises(ValueError, match=message):
            prepare_audio(row["wav"], row["sample_rate"])

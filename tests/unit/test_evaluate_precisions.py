import importlib.util
import json
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
    rows = evaluate.load_rows(argparse.Namespace(dataset=None, audio=[f"{path}:eng_Latn"],
                                                 limit=None))
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
    args = argparse.Namespace(dataset=None, audio=[str(path)], device="cpu", lang=None, limit=None,
                              llm=tmp_path if kind == "llm" else None,
                              ctc=None if kind == "llm" else tmp_path)
    row = evaluate.load_rows(args)[0]
    evaluate.transcriber(args, "fp32")(row)
    # Exactly what the public file API feeds the model (one normalization) ...
    np.testing.assert_array_equal(received[0], load_audio(path))
    # ... which differs from normalizing twice, the old behaviour.
    assert not np.array_equal(received[0], prepare_audio(load_audio(path)[0]))


@pytest.mark.parametrize("kwargs, message", [({"rate": 8000}, "16 kHz"),
                                             ({"channels": 2}, "mono")])
def test_bad_files_are_rejected_while_loading_rows(tmp_path, kwargs, message):
    """Before any model is loaded, as load_audio() used to do."""
    import argparse

    path, _ = _wav(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        evaluate.load_rows(argparse.Namespace(dataset=None, audio=[str(path)], limit=None))



# --- inputs are validated before any model loads (issue #23) ---

def _row(row_id, ref="hello", n=1600):
    import numpy as np

    return {"id": row_id, "ref": ref,
            "wav": np.random.default_rng(len(str(row_id))).standard_normal(n).astype("float32")}


def test_error_rates_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="2 references but 1 hypotheses"):
        error_rates(["a", "b"], ["a"])


@pytest.mark.parametrize("rows, message", [
    ([], "no evaluation inputs"),
    ("not a list", "must be a list of rows"),
    ([{"id": [1], "wav": []}], "unhashable id"),
    ([{"wav": []}], "must be a dict with 'id' and 'wav'"),
    ([_row("a"), _row("b"), _row("a")], "duplicate ids would overwrite"),
    ([{**_row("a"), "ref": 3}], "has no 'ref'"),
    ([_row("short", n=100)], "row 'short': At least 400 samples"),
])
def test_validate_rows_rejects_bad_datasets(rows, message):
    with pytest.raises((ValueError, TypeError), match=message):
        evaluate.validate_rows(rows, labelled=True)


def test_validate_rows_accepts_unlabelled_audio_rows():
    row = _row("a")
    del row["ref"]
    assert evaluate.validate_rows([row], labelled=False) == [row]


def _dataset(tmp_path, rows):
    import pickle

    path = tmp_path / "data.pkl"
    path.write_bytes(pickle.dumps(rows))
    return path


def test_load_rows_applies_limit_then_validates(tmp_path):
    import argparse

    rows = [_row("a"), _row("b"), _row("a")]  # the duplicate falls outside --limit 2
    args = argparse.Namespace(dataset=_dataset(tmp_path, rows), audio=None, limit=2,
                              trust_pickle=True)
    assert [r["id"] for r in evaluate.load_rows(args)] == ["a", "b"]
    args.limit = None
    with pytest.raises(ValueError, match="duplicate ids"):
        evaluate.load_rows(args)


def test_validate_rows_keeps_silence_refs_and_accepts_tuples():
    rows = (_row("a", ref=None), _row("b"))
    assert [r["id"] for r in evaluate.validate_rows(rows, labelled=True)] == ["a", "b"]
    missing = _row("c")
    del missing["ref"]
    with pytest.raises(ValueError, match="has no 'ref'"):
        evaluate.validate_rows([missing], labelled=True)


def test_llm_runs_reject_clips_over_30_seconds_up_front():
    long_row = _row("long", n=31 * 16000)
    assert evaluate.validate_rows([long_row], labelled=True)  # fine for CTC
    with pytest.raises(ValueError, match="row 'long' is longer than the LLM runtime's 30 s"):
        evaluate.validate_rows([long_row], labelled=True, max_seconds=30)


def test_limit_applies_before_audio_files_are_read(tmp_path):
    import argparse

    path, _ = _wav(tmp_path)
    args = argparse.Namespace(dataset=None, audio=[str(path), str(tmp_path / "missing.wav")],
                              limit=1)
    assert len(evaluate.load_rows(args)) == 1


def test_cli_reports_bad_inputs_without_a_traceback(tmp_path, monkeypatch):
    import sys

    path = _dataset(tmp_path, [])
    monkeypatch.setattr(sys, "argv", ["evaluate_precisions.py", "--ctc", "x",
                                      "--dataset", str(path), "--trust-pickle"])
    with pytest.raises(SystemExit, match="^error: no evaluation inputs"):
        evaluate.main()


def test_load_rows_rejects_non_list_pickles(tmp_path):
    import argparse

    args = argparse.Namespace(dataset=_dataset(tmp_path, {"a": 1}), audio=None, limit=None,
                              trust_pickle=True)
    with pytest.raises(TypeError, match="must contain a list"):
        evaluate.load_rows(args)


def test_duplicate_audio_specs_are_rejected(tmp_path):
    import argparse

    path, _ = _wav(tmp_path)
    args = argparse.Namespace(dataset=None, audio=[str(path), str(path)], limit=None)
    with pytest.raises(ValueError, match="duplicate ids"):
        evaluate.load_rows(args)


@pytest.mark.parametrize("argv, message", [
    (["--limit", "0"], "must be a positive integer"),
    (["--limit", "-3"], "must be a positive integer"),
    (["--precision", "fp32", "int8"], "invalid choice"),
])
def test_cli_rejects_bad_limits_and_precisions(monkeypatch, capsys, argv, message):
    import sys

    monkeypatch.setattr(sys, "argv", ["evaluate_precisions.py", "--ctc", "x", "--audio", "a.wav",
                                      *argv])
    with pytest.raises(SystemExit):
        evaluate.main()
    assert message in capsys.readouterr().err


def test_cli_rejects_duplicate_precisions(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["evaluate_precisions.py", "--ctc", "x", "--audio", "a.wav",
                                      "--precision", "fp32", "fp16", "fp16"])
    with pytest.raises(SystemExit, match="lists a precision twice"):
        evaluate.main()



# --- JSONL manifests replace pickle datasets (issue #24) ---

def _manifest(tmp_path, entries, name="eval.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) if not isinstance(e, str) else e
                              for e in entries) + "\n", encoding="utf-8")
    return path


def _audio_files(tmp_path):
    import numpy as np
    import soundfile as sf

    (tmp_path / "clips").mkdir()
    rng = np.random.default_rng(1)
    flac = rng.standard_normal(16000).astype("float32") * 0.1
    sf.write(tmp_path / "clips" / "a.flac", flac, 16000)
    npy = rng.standard_normal(8000).astype("float32")
    np.save(tmp_path / "clips" / "b.npy", npy)
    return flac, npy


def test_manifest_rows_resolve_audio_relative_to_the_manifest(tmp_path):
    import argparse

    import numpy as np

    flac, npy = _audio_files(tmp_path)
    path = _manifest(tmp_path, [
        {"id": "a", "audio": "clips/a.flac", "ref": "Hello there.", "lang": "eng_Latn"},
        "",
        {"id": "b", "audio": "clips/b.npy", "ref": None},
    ])
    rows = evaluate.load_rows(argparse.Namespace(dataset=path, audio=None, limit=None))
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["lang"] == "eng_Latn" and rows[0]["sample_rate"] == 16000
    np.testing.assert_allclose(rows[0]["wav"], flac, atol=1e-4)  # FLAC is 16-bit
    np.testing.assert_array_equal(rows[1]["wav"], npy)
    assert rows[1]["ref"] is None  # silence


def test_manifest_limit_stops_reading_early(tmp_path):
    import argparse

    _audio_files(tmp_path)
    path = _manifest(tmp_path, [{"id": "a", "audio": "clips/a.flac", "ref": "x"},
                                {"id": "b", "audio": "missing.flac", "ref": "y"}])
    rows = evaluate.load_rows(argparse.Namespace(dataset=path, audio=None, limit=1))
    assert [r["id"] for r in rows] == ["a"]


@pytest.mark.parametrize("line, message", [
    ("{not json", r"eval.jsonl:1: invalid JSON"),
    ({"id": "a", "ref": "x"}, "needs at least 'id' and 'audio'"),
    ({"id": "a", "audio": "clips/a.flac", "ref": "x", "wav": [1]}, r"unknown keys \['wav'\]"),
    ({"id": "a", "audio": "clips/a.flac", "ref": "x", "sample_rate": 8000}, "disagrees"),
])
def test_manifest_errors_name_the_line(tmp_path, line, message):
    _audio_files(tmp_path)
    with pytest.raises(ValueError, match=message):
        evaluate.read_manifest(_manifest(tmp_path, [line]))


def test_manifest_npy_arrays_are_loaded_without_pickle(tmp_path):
    import numpy as np

    (tmp_path / "evil.npy").write_bytes(b"")
    np.save(tmp_path / "evil.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="allow_pickle"):
        evaluate.read_manifest(_manifest(tmp_path, [{"id": "x", "audio": "evil.npy"}]))


def test_pickle_datasets_need_explicit_trust(tmp_path, monkeypatch):
    import sys

    path = _dataset(tmp_path, [_row("a")])
    monkeypatch.setattr(sys, "argv", ["evaluate_precisions.py", "--ctc", "x",
                                      "--dataset", str(path)])
    with pytest.raises(SystemExit, match="can run arbitrary code.*--trust-pickle"):
        evaluate.main()

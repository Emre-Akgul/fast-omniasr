"""benchmarks/benchmark.py records what is needed to reproduce a number (issue #26)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("benchmark", ROOT / "benchmarks" / "benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_percentiles_match_numpy():
    values = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2]
    for q in (0, 50, 90, 95, 99, 100):
        assert benchmark.percentile(values, q) == pytest.approx(np.percentile(values, q))
    assert benchmark.percentile([0.4], 99) == 0.4


def test_latency_summary_includes_rtf():
    summary = benchmark.latency_summary([0.2, 0.4, 0.6], duration=2.0)
    assert summary["runs"] == 3 and summary["p50"] == pytest.approx(0.4)
    assert summary["rtf_p50"] == pytest.approx(0.2) and summary["rtf_mean"] == pytest.approx(0.2)
    assert benchmark.latency_summary([0.5], 1.0)["stdev"] == 0.0


class FakeModel:
    calls: list = []  # noqa: RUF012 -- reset per test by the fixture

    def __init__(self, *args, **kwargs):
        FakeModel.calls.append(("init", args, kwargs))
        self.snapshot = None

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        FakeModel.calls.append(("from_pretrained", repo, kwargs))
        model = cls.__new__(cls)
        import types
        model.snapshot = types.SimpleNamespace(revision="main", commit="c" * 40)
        return model

    def transcribe(self, path):
        import types
        return types.SimpleNamespace(text="hello")

    def close(self):
        FakeModel.calls.append(("close",))


@pytest.fixture
def clip(tmp_path):
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(32000, np.float32), 16000)
    return path


@pytest.fixture
def fake(monkeypatch):
    import fast_omniasr

    FakeModel.calls = []
    monkeypatch.setattr(fast_omniasr, "OmniASR", FakeModel)
    return FakeModel


def test_local_model_report_is_reproducible(clip, fake, tmp_path):
    model, tokenizer = tmp_path / "model.onnx", tmp_path / "tokenizer.model"
    model.write_bytes(b"graph")
    tokenizer.write_bytes(b"tok")
    out = tmp_path / "report.json"
    benchmark.main([str(clip), "--model", str(model), "--tokenizer", str(tokenizer),
                    "--precision", "fp16", "--threads", "2", "--runs", "3", "--warmup", "1",
                    "--output", str(out)])
    report = json.loads(out.read_text())
    assert report["report_version"] == 1
    assert report["model"]["model_sha256"] == benchmark.sha256(model)
    assert report["model"]["tokenizer_sha256"] == benchmark.sha256(tokenizer)
    assert (report["backend"], report["precision"], report["device"], report["threads"]) == (
        "onnx", "fp16", "cpu", 2)
    assert report["audio"]["seconds"] == 2.0 and report["audio"]["sample_rate"] == 16000
    assert report["audio"]["sha256"] == benchmark.sha256(clip)
    latency = report["latency_seconds"]
    assert latency["runs"] == 3 and {"p50", "p90", "p95", "p99", "rtf_p50"} <= latency.keys()
    assert len(report["samples"]) == 3 and report["warmup_runs"] == 1
    assert report["memory"]["peak_gpu_mib"] is None  # CPU run
    assert report["environment"]["python"] and "packages" in report["environment"]
    init = next(c for c in fake.calls if c[0] == "init")
    assert init[2] == {"threads": 2, "precision": "fp16", "backend": "onnx"}
    assert ("close",) in fake.calls


def test_repo_report_records_the_resolved_commit(clip, fake, tmp_path):
    out = tmp_path / "r.json"
    benchmark.main([str(clip), "--repo", "owner/repo", "--precision", "fp16", "--runs", "1",
                    "--output", str(out)])
    report = json.loads(out.read_text())
    assert report["model"] == {"repo": "owner/repo", "revision": "main", "commit": "c" * 40}
    call = next(c for c in fake.calls if c[0] == "from_pretrained")
    assert call[2]["precision"] == "fp16" and call[2]["backend"] == "onnx"


@pytest.mark.parametrize("argv, message", [
    (["--model", "m.onnx"], "needs --tokenizer"),
    (["--repo", "o/r", "--tokenizer", "t"], "a repo provides its own"),
    (["--model", "m", "--tokenizer", "t", "--revision", "v1"], "applies to --repo"),
    (["--repo", "o/r", "--backend", "tensorrt"], "needs --precision"),
    (["--model", "e.engine", "--tokenizer", "t", "--backend", "tensorrt", "--precision", "fp16"],
     "carries its own precision"),
    (["--repo", "o/r", "--runs", "0"], "--runs must be positive"),
    ([], "one of the arguments --repo --model is required"),
])
def test_argument_validation(argv, message, capsys):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["clip.wav", *argv])
    assert message in capsys.readouterr().err

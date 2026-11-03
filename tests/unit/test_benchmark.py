"""benchmarks/benchmark.py records what is needed to reproduce a number (issue #26)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx_bytes
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
    model.write_bytes(onnx_bytes.model_with_external())  # inline weights: one file
    tokenizer.write_bytes(b"tok")
    out = tmp_path / "report.json"
    benchmark.main([str(clip), "--model", str(model), "--tokenizer", str(tokenizer),
                    "--precision", "fp16", "--threads", "2", "--runs", "3", "--warmup", "1",
                    "--output", str(out)])
    report = json.loads(out.read_text())
    assert report["report_version"] == 1
    assert report["model"]["files_sha256"] == {"model.onnx": benchmark.sha256(model)}
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
    (["--repo", "o/r", "--threads", "0"], "--threads must be positive"),
    (["--repo", "o/r", "--precision", "int8"], "--precision for onnx: one of"),
    (["--repo", "o/r", "--precision", "bf16"], "needs --device cuda"),
    (["--repo", "o/r", "--backend", "tensorrt", "--precision", "fp16", "--threads", "2"],
     "drop --threads/--device"),
    (["--repo", "o/r", "--backend", "tensorrt", "--precision", "bf16"], "--precision for tensorrt"),
    ([], "one of the arguments --repo --model is required"),
])
def test_argument_validation(argv, message, capsys, clip):
    with pytest.raises(SystemExit):
        benchmark.parse_args([str(clip), *argv])
    assert message in capsys.readouterr().err


def test_missing_audio_is_a_clean_error(capsys, tmp_path):
    with pytest.raises(SystemExit):
        benchmark.parse_args([str(tmp_path / "nope.wav"), "--repo", "o/r"])
    assert "audio file not found" in capsys.readouterr().err


def test_model_identity_covers_external_data(tmp_path):
    (tmp_path / "fp16").mkdir()
    graph = tmp_path / "fp16" / "model.onnx"
    graph.write_bytes(onnx_bytes.model_with_external("model.onnx.data"))
    (tmp_path / "fp16" / "model.onnx.data").write_bytes(b"weights v1")
    first = benchmark.model_files(graph)
    assert set(first) == {"model.onnx", "model.onnx.data"}
    (tmp_path / "fp16" / "model.onnx.data").write_bytes(b"weights v2")  # same graph, new weights
    assert benchmark.model_files(graph)["model.onnx.data"] != first["model.onnx.data"]


def test_hashing_is_not_timed_as_load(clip, fake, tmp_path, monkeypatch):
    model, tokenizer = tmp_path / "model.onnx", tmp_path / "tokenizer.model"
    model.write_bytes(onnx_bytes.model_with_external())  # inline weights: one file
    tokenizer.write_bytes(b"tok")
    import time as time_module

    real = benchmark.sha256

    def slow_sha256(path):
        time_module.sleep(0.2)
        return real(path)

    monkeypatch.setattr(benchmark, "sha256", slow_sha256)
    out = tmp_path / "r.json"
    benchmark.main([str(clip), "--model", str(model), "--tokenizer", str(tokenizer), "--runs",
                    "1", "--output", str(out)])
    assert json.loads(out.read_text())["load_seconds"] < 0.2


def test_local_engine_precision_comes_from_its_cache_metadata(tmp_path):
    engine = tmp_path / "model.engine"
    (tmp_path / "metadata.json").write_text(json.dumps({"key": {"precision": "mixed"}}))
    assert benchmark.engine_precision(engine) == "mixed"
    assert benchmark.engine_precision(tmp_path / "elsewhere" / "x.engine") is None


def test_runtime_errors_exit_cleanly(clip, monkeypatch, tmp_path):
    import fast_omniasr

    def failing(*args, **kwargs):
        raise ValueError("bad model")

    monkeypatch.setattr(fast_omniasr, "OmniASR", failing)
    (tmp_path / "m.onnx").write_bytes(b"x")
    with pytest.raises(SystemExit, match="error: bad model"):
        benchmark.main([str(clip), "--model", str(tmp_path / "m.onnx"), "--tokenizer",
                        str(tmp_path / "m.onnx")])

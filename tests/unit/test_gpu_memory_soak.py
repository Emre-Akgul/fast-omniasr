"""tools/gpu_memory_soak.py, the per-request memory growth check (#96)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gpu_memory_soak", ROOT / "tools" / "gpu_memory_soak.py")
soak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(soak)


def test_meminfo_reads_nvmap_in_mib(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:  8000000 kB\nNvMapMemUsed:     204800 kB\n")
    assert soak.meminfo_mb("NvMapMemUsed", str(path)) == 200
    assert soak.meminfo_mb("NvMapMemFree", str(path)) is None
    assert soak.meminfo_mb("NvMapMemUsed", str(tmp_path / "missing")) is None


def test_cgroup_kernel_counter(tmp_path):
    path = tmp_path / "memory.stat"
    path.write_text("anon 1048576\nkernel 3145728\nkernel_stack 16384\n")
    assert soak.cgroup_kernel_mb(str(path)) == 3
    assert soak.cgroup_kernel_mb(str(tmp_path / "missing")) is None


def test_growth_is_measured_after_warmup_and_skips_missing_counters():
    samples = [{"a": 100.0, "b": None, "c": 5.0}, {"a": 180.0, "b": None, "c": 5.0},
               {"a": 182.0, "b": None, "c": None}, {"a": 190.0, "b": None, "c": 5.0}]
    assert soak.growth(samples, warmup=2) == {"a": 10.0, "c": 0.0}
    with pytest.raises(ValueError):
        soak.growth(samples, warmup=4)


def test_lengths_are_fixed_or_a_seeded_mix():
    assert soak.lengths(3, 20, mixed=False) == [320000] * 3
    mixed = soak.lengths(50, 30, mixed=True)
    assert mixed == soak.lengths(50, 30, mixed=True)
    assert all(16000 <= n <= 480000 for n in mixed) and len(set(mixed)) == 50


def test_arena_shrinkage_passes_run_options(monkeypatch):
    added = []

    class RunOptions:
        def add_run_config_entry(self, key, value):
            added.append((key, value))

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(RunOptions=RunOptions))
    calls = []
    session = types.SimpleNamespace(run=lambda *args: calls.append(args) or ["out"],
                                    get_providers=lambda: ["CUDAExecutionProvider"])
    model = types.SimpleNamespace(backend=types.SimpleNamespace(session=session))
    soak.shrink_arena(model)
    assert model.backend.session.run(["logits"], {"audio": 1}) == ["out"]
    assert isinstance(calls[0][2], RunOptions)
    assert added == [("memory.enable_memory_arena_shrinkage", "gpu:0")]
    assert model.backend.session.get_providers() == ["CUDAExecutionProvider"]


def test_arena_shrinkage_needs_an_onnx_session():
    with pytest.raises(ValueError, match="ONNX Runtime CTC"):
        soak.shrink_arena(types.SimpleNamespace(backend=types.SimpleNamespace()))


class Model:
    def __init__(self, leak):
        self.leak, self.used = leak, 1000.0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def transcribe_numpy(self, audio):
        assert audio.dtype == np.float32
        self.used += self.leak


@pytest.mark.parametrize(("leak", "code"), [(0.0, 0), (3.0, 1)])
def test_exit_status_reports_growth(monkeypatch, capsys, tmp_path, leak, code):
    model = Model(leak)
    monkeypatch.setattr(soak, "load", lambda args: model)
    monkeypatch.setattr(soak, "sample", lambda nvml: {"gpu_process_mb": model.used,
                                                      "nvmap_mb": None})
    output = tmp_path / "soak.json"
    assert soak.main(["--model", "m", "--requests", "10", "--seconds", "1",
                      "--output", str(output)]) == code
    summary = json.loads(capsys.readouterr().out)
    assert summary["growth_mb"] == {"gpu_process_mb": leak * 5}
    assert summary["leaking"] == (["gpu_process_mb"] if leak else [])
    assert len(json.loads(output.read_text())["samples"]) == 11


def test_no_counters_is_its_own_exit_status(monkeypatch):
    monkeypatch.setattr(soak, "load", lambda args: Model(0.0))
    monkeypatch.setattr(soak, "sample", lambda nvml: {"rss_mb": None})
    assert soak.main(["--model", "m", "--requests", "7", "--seconds", "1"]) == 2


def test_mixed_lengths_default_to_a_long_warmup(monkeypatch, capsys):
    monkeypatch.setattr(soak, "load", lambda args: Model(0.0))
    monkeypatch.setattr(soak, "sample", lambda nvml: {"rss_mb": 1.0})
    soak.main(["--model", "m", "--requests", "8", "--seconds", "2", "--mixed"])
    assert json.loads(capsys.readouterr().out)["warmup"] == 4

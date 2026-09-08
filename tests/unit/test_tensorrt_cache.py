import builtins
import json
import threading
import time

import pytest

from fast_omniasr import tensorrt_cache
from fast_omniasr.tensorrt_cache import get_or_build_engine

FAKE_ENV = {
    "tensorrt_version": "10.16.1.11",
    "compute_capability": "8.6",
    "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
}


@pytest.fixture
def counting_builder(monkeypatch):
    calls = []

    def fake_build_engine(onnx_path, output_path, *, precision, min_samples, opt_samples,
                           max_samples):
        calls.append(precision)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"FAKE ENGINE {precision}".encode())
        return output_path

    monkeypatch.setattr(tensorrt_cache, "build_engine", fake_build_engine)
    monkeypatch.setattr(tensorrt_cache, "_local_environment", lambda: dict(FAKE_ENV))
    return calls


@pytest.fixture
def onnx_a(tmp_path):
    path = tmp_path / "a.onnx"
    path.write_bytes(b"onnx content A")
    return path


@pytest.fixture
def onnx_b(tmp_path):
    path = tmp_path / "b.onnx"
    path.write_bytes(b"onnx content B, totally different")
    return path


def test_empty_cache_builds(counting_builder, onnx_a, tmp_path):
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=tmp_path / "cache")
    assert len(counting_builder) == 1
    assert engine_path.exists()


def test_second_call_reuses_cache(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    second = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert first == second
    assert len(counting_builder) == 1


def test_onnx_hash_change_triggers_rebuild(counting_builder, onnx_a, onnx_b, tmp_path):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    second = get_or_build_engine(onnx_b, precision="fp16", cache_dir=cache_dir)
    assert first != second
    assert len(counting_builder) == 2


def test_precision_change_produces_distinct_entries(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    fp16 = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    fp32 = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache_dir)
    assert fp16 != fp32
    assert len(counting_builder) == 2
    assert fp16.exists() and fp32.exists()  # neither call clobbered the other's entry


def test_incompatible_metadata_triggers_rebuild(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    metadata_path = engine_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tensorrt_version"] = "1.0.0.0"  # simulate a stale/incompatible record
    metadata_path.write_text(json.dumps(metadata))

    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert len(counting_builder) == 2


def test_missing_or_corrupted_metadata_triggers_rebuild(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    (engine_path.parent / "metadata.json").write_text("not valid json{")

    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert len(counting_builder) == 2


def test_invalid_precision_raises(onnx_a, tmp_path):
    with pytest.raises(ValueError, match="precision"):
        get_or_build_engine(onnx_a, precision="int8", cache_dir=tmp_path / "cache")


def test_concurrent_builds_of_same_key_build_only_once(counting_builder, onnx_a, tmp_path,
                                                         monkeypatch):
    cache_dir = tmp_path / "cache"
    real_build = tensorrt_cache.build_engine

    def slow_build(*args, **kwargs):
        time.sleep(0.2)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(tensorrt_cache, "build_engine", slow_build)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
            )
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(counting_builder) == 1
    assert len(set(results)) == 1


def test_missing_tensorrt_raises_helpful_import_error(monkeypatch, onnx_a, tmp_path):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("simulated: tensorrt not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[tensorrt\\]"):
        get_or_build_engine(onnx_a, precision="fp16", cache_dir=tmp_path / "cache")

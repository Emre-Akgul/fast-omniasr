"""Shared runtime option validation (issue #9): same rules and messages everywhere, and
options a backend would ignore are rejected instead."""

import sys

import numpy as np
import pytest

from fast_omniasr import OmniASR
from fast_omniasr.llm import OmniASRLLM
from fast_omniasr.options import check_device, check_flag, check_threads, resolve_threads

BAD_THREADS = [0, -1, 2.5, True, False, "4", None, np.int64(0)]


def test_validators_accept_valid_values():
    assert check_device("cpu") == "cpu" and check_device("cuda") == "cuda"
    assert check_threads(1) == 1 and check_threads(16) == 16
    assert check_threads(np.int64(4)) == 4 and type(check_threads(np.int32(2))) is int
    assert resolve_threads(None) == 4 and resolve_threads(8) == 8
    assert check_flag("low_memory", False) is False


@pytest.mark.parametrize("value", BAD_THREADS)
def test_threads_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="threads must be a positive integer"):
        check_threads(value)


@pytest.mark.parametrize("value", ["gpu", "cuda:0", "CPU", None])
def test_device_must_be_cpu_or_cuda(value):
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        check_device(value)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_flags_must_be_bool(value):
    with pytest.raises(TypeError, match="low_memory must be True or False"):
        check_flag("low_memory", value)


# --- CTC ---------------------------------------------------------------------------

@pytest.mark.parametrize("threads", [0, 2.5, True])
def test_ctc_onnx_rejects_bad_threads_before_loading(threads):
    with pytest.raises(ValueError, match="threads must be a positive integer"):
        OmniASR("missing.onnx", "missing.model", threads=threads)


def test_ctc_onnx_rejects_unknown_device_before_loading():
    with pytest.raises(ValueError, match="device must be 'cpu' or 'cuda'"):
        OmniASR("missing.onnx", "missing.model", device="gpu")


@pytest.mark.parametrize("kwargs, message", [
    ({"device": "cpu"}, "runs on the current CUDA device"),
    ({"threads": 4}, "threads only applies to backend='onnx'"),
    ({"precision": "fp16"}, "carries its own precision"),
])
def test_tensorrt_rejects_options_it_would_ignore(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OmniASR("missing.engine", "missing.model", backend="tensorrt", **kwargs)


def test_tensorrt_from_pretrained_rejects_before_any_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # a download would fail loudly
    with pytest.raises(ValueError, match="runs on the current CUDA device"):
        OmniASR.from_pretrained("o/r", backend="tensorrt", precision="fp16", device="cpu")
    with pytest.raises(ValueError, match="threads only applies"):
        OmniASR.from_pretrained("o/r", backend="tensorrt", precision="fp16", threads=2)


def test_onnx_from_pretrained_rejects_bad_threads_before_any_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ValueError, match="threads must be a positive integer"):
        OmniASR.from_pretrained("o/r", threads=0)


# --- LLM ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, error, message", [
    ({"backend": "onnx", "threads": 0}, ValueError, "threads must be a positive integer"),
    ({"backend": "onnx", "device": "gpu"}, ValueError, "device must be 'cpu' or 'cuda'"),
    ({"backend": "onnx", "low_memory": "yes"}, TypeError, "low_memory must be True or False"),
    ({"backend": "torch", "threads": 8}, ValueError, "threads only applies to backend='onnx'"),
    ({"backend": "torch", "low_memory": True}, ValueError, "low_memory only applies"),
    ({"backend": "torch", "device": "tpu"}, ValueError, "device must be 'cpu' or 'cuda'"),
])
def test_llm_rejects_bad_or_inapplicable_options_before_reading_files(tmp_path, kwargs,
                                                                     error, message):
    with pytest.raises(error, match=message):
        OmniASRLLM(tmp_path / "does-not-exist", **kwargs)


@pytest.mark.parametrize("kwargs, error, message", [
    ({"threads": 0}, ValueError, "threads must be a positive integer"),
    ({"low_memory": 1}, TypeError, "low_memory must be True or False"),
])
def test_llm_from_pretrained_rejects_before_any_download(monkeypatch, kwargs, error, message):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(error, match=message):
        OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-300M-v2-ONNX", **kwargs)


def test_llm_from_pretrained_accepts_threads_none_like_init(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    # Validation passes; the (blocked) download is the first thing that fails.
    with pytest.raises(ImportError, match="huggingface_hub"):
        OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-300M-v2-ONNX", threads=None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-300M-v2-ONNX", threads=np.int64(2))


def test_llm_rejects_non_bool_cached(tmp_path):
    with pytest.raises(TypeError, match="cached must be True or False"):
        OmniASRLLM(tmp_path / "missing", backend="onnx", cached="yes")


@pytest.mark.parametrize("backend, expect_threads", [("torch", False), ("onnx", True)])
def test_validate_llm_passes_threads_only_to_onnx(tmp_path, monkeypatch, backend, expect_threads):
    import importlib.util
    import json
    from pathlib import Path

    import fast_omniasr.llm

    if backend == "torch":
        pytest.importorskip("torch")
    (tmp_path / "reference.json").write_text(json.dumps({"audio": "a.wav"}))
    (tmp_path / "cached_reference.json").write_text(json.dumps({"clips": [{}]}))
    seen = {}

    def fake_llm(directory, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)  # stop before loading audio or models

    monkeypatch.setattr(fast_omniasr.llm, "OmniASRLLM", fake_llm)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))  # the script adds a blocker
    monkeypatch.setattr(sys, "argv", ["validate_llm.py", str(tmp_path), "--backend", backend])
    script = Path(__file__).resolve().parents[2] / "tools" / "validate_llm.py"
    spec = importlib.util.spec_from_file_location("validate_llm_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module.main()
    assert ("threads" in seen) is expect_threads

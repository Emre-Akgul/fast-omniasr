"""Public close()/context-manager lifecycle for ONNX CTC and LLM models (issue #4)."""

import sys
import types

import numpy as np
import pytest

from fast_omniasr import OmniASR
from fast_omniasr.backends.onnx import ONNXBackend
from fast_omniasr.llm import OmniASRLLM


@pytest.fixture
def fake_ort(monkeypatch):
    """Minimal onnxruntime stand-in: a CPU session with the CTC graph's I/O signature."""
    sessions = []

    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            self.runs = 0
            sessions.append(self)

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [types.SimpleNamespace(name="audio", type="tensor(float)", shape=[1, "n"])]

        def get_outputs(self):
            return [types.SimpleNamespace(name="logits")]

        def run(self, names, feeds):
            self.runs += 1
            return [np.zeros((1, feeds["audio"].shape[1] // 320, 4), np.float32)]

    class SessionOptions:
        def add_session_config_entry(self, key, value):
            pass

    ort = types.SimpleNamespace(
        InferenceSession=Session,
        SessionOptions=SessionOptions,
        get_available_providers=lambda: ["CPUExecutionProvider"],
        preload_dlls=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return sessions


def test_onnx_backend_close_releases_session(fake_ort):
    backend = ONNXBackend("model.onnx")
    assert backend.infer(np.zeros((1, 16000), np.float32)).shape == (1, 50, 4)
    backend.close()
    backend.close()  # idempotent
    assert backend.closed and backend.session is None
    with pytest.raises(RuntimeError, match="ONNXBackend is closed"):
        backend.infer(np.zeros((1, 16000), np.float32))


def test_onnx_backend_context_manager(fake_ort):
    with ONNXBackend("model.onnx") as backend:
        backend.infer(np.zeros((1, 16000), np.float32))
    assert backend.closed


class _Tokenizer:
    def __init__(self, path):
        pass

    def decode(self, ids):
        return ""


@pytest.mark.parametrize("use_with", [False, True])
def test_omniasr_onnx_close_and_post_close_error(fake_ort, monkeypatch, use_with):
    from fast_omniasr import model as model_module

    monkeypatch.setattr(model_module, "Tokenizer", _Tokenizer)
    model = OmniASR("model.onnx", "tokenizer.model", backend="onnx")
    if use_with:
        with model:
            model.transcribe_numpy(np.zeros(16000, np.float32))
    else:
        model.transcribe_numpy(np.zeros(16000, np.float32))
        model.close()
    model.close()  # idempotent
    assert model.closed and model.backend.session is None
    with pytest.raises(RuntimeError, match="OmniASR model is closed"):
        model.transcribe_numpy(np.zeros(16000, np.float32))


class _FakeLLMBackend:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _llm(onnx=None, **attrs):
    model = object.__new__(OmniASRLLM)
    model.closed = False
    model.backend = "onnx" if onnx is not None else "torch"
    if onnx is not None:
        model.onnx = onnx
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def test_llm_close_releases_onnx_backend_once():
    backend = _FakeLLMBackend()
    model = _llm(onnx=backend)
    model.close()
    model.close()
    assert model.closed and backend.close_calls == 1
    with pytest.raises(RuntimeError, match="OmniASRLLM model is closed"):
        model.transcribe(np.zeros(16000, np.float32), lang="eng_Latn")


def test_llm_context_manager_closes_on_error():
    backend = _FakeLLMBackend()
    with pytest.raises(KeyError), _llm(onnx=backend) as model:
        raise KeyError("boom")
    assert model.closed and backend.close_calls == 1


def test_llm_torch_close_drops_modules():
    model = _llm(encoder=object(), decoder=object())
    with model:
        pass
    assert model.closed and model.encoder is None and model.decoder is None


def test_llm_close_on_bare_instance_is_safe():
    model = object.__new__(OmniASRLLM)  # e.g. __init__ raised before setting attributes
    model.close()
    assert model.closed


@pytest.fixture
def llm_backend(fake_ort, monkeypatch, tmp_path):
    """A fresh ONNXLLMBackend module bound to the fake onnxruntime, with stub sessions."""
    import importlib

    monkeypatch.delitem(sys.modules, "fast_omniasr.llm.onnx_backend", raising=False)
    sys.modules["onnxruntime"].RunOptions = lambda: types.SimpleNamespace(
        add_run_config_entry=lambda key, value: None
    )
    module = importlib.import_module("fast_omniasr.llm.onnx_backend")
    for name in ["encoder.onnx", "decoder_prefill.onnx", "decoder_step.onnx",
                 "decoder.weights", "decoder_weights.json"]:
        (tmp_path / name).write_bytes(b"")
    loads = []

    class Graph:
        def __init__(self, inputs, outputs):
            self.inputs, self.outputs = inputs, outputs

        def get_inputs(self):
            return [types.SimpleNamespace(name=name) for name in self.inputs]

        def run(self, names, feeds, run_options=None):
            return self.outputs()

    def step_outputs():
        return [np.zeros((1, 10), np.float32), *[np.zeros((1, 2, 1, 1), np.float32)] * 24]

    def session(path, device="cpu", opts=None, arena=None):
        loads.append("encoder")
        return Graph(["audio"], lambda: [np.zeros((1, 4, 8), np.float32)])

    def decoder_sessions(directory, device="cpu", threads=4):
        loads.append("decoder")
        return (Graph(["context", "bos"], step_outputs),
                Graph(module.STEP_INPUTS, step_outputs), [])

    monkeypatch.setattr(module, "session", session)
    monkeypatch.setattr(module, "decoder_sessions", decoder_sessions)
    return module.ONNXLLMBackend, tmp_path, loads


def _run_clip(backend):
    context = backend.encode(np.zeros((1, 1600), np.float32))
    _, cache = backend.prefill(context, np.zeros((1, 1), np.int64))
    backend.decode_step(np.zeros((1, 1), np.int64), cache)


def test_llm_backend_close_is_permanent_and_idempotent(llm_backend):
    backend_class, directory, loads = llm_backend
    backend = backend_class(directory, "cpu")
    _run_clip(backend)
    backend.close()
    backend.close()
    assert backend.closed
    assert backend.encoder_session is None and backend.prefill_session is None
    for call in (lambda: backend.encode(np.zeros((1, 1600), np.float32)),
                 lambda: backend.prefill(np.zeros((1, 4, 8), np.float32),
                                         np.zeros((1, 1), np.int64)),
                 lambda: backend.decode_step(np.zeros((1, 1), np.int64), [None] * 24)):
        with pytest.raises(RuntimeError, match="ONNXLLMBackend is closed"):
            call()
    assert loads == ["encoder", "decoder"]  # nothing was silently reloaded


def test_llm_backend_low_memory_unload_is_not_a_close(llm_backend):
    backend_class, directory, loads = llm_backend
    backend = backend_class(directory, "cpu", low_memory=True)
    _run_clip(backend)
    _run_clip(backend)  # sessions are phased (unloaded and reloaded) between clips
    assert not backend.closed
    assert loads == ["encoder", "decoder"] * 2


def test_close_returns_freed_memory_once(fake_ort, monkeypatch):
    from fast_omniasr.backends import onnx as onnx_module

    calls = []
    monkeypatch.setattr(onnx_module, "release_freed_memory", lambda: calls.append(1))
    backend = ONNXBackend("model.onnx")
    backend.close()
    backend.close()
    assert calls == [1]


def test_release_freed_memory_is_safe_without_glibc(monkeypatch):
    from fast_omniasr import _memory

    def missing(name):
        raise OSError("no libc")

    monkeypatch.setattr(_memory.ctypes, "CDLL", missing)
    _memory.release_freed_memory()  # must not raise
    monkeypatch.setattr(_memory.sys, "platform", "darwin")
    _memory.release_freed_memory()

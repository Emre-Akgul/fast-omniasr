"""CUDA provider options shared by the CTC and LLM ONNX runtimes (#97)."""

import sys
import types

import pytest

from fast_omniasr.options import CUDA_PROVIDER_OPTIONS


def test_cudnn_uses_the_fixed_algorithm():
    # EXHAUSTIVE and HEURISTIC request a ~17.6 GB workspace for short inputs on cuDNN 9.27+.
    assert CUDA_PROVIDER_OPTIONS == {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT"}


def test_ctc_cuda_session_gets_the_shared_options(monkeypatch):
    seen = {}

    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            seen["providers"] = providers

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CUDAExecutionProvider"]

        def get_inputs(self):
            return [types.SimpleNamespace(name="audio", type="tensor(float)", shape=[1, "n"])]

        def get_outputs(self):
            return [types.SimpleNamespace(name="logits", shape=[1, "frames", 4])]

    ort = types.SimpleNamespace(
        InferenceSession=Session,
        SessionOptions=lambda: types.SimpleNamespace(),
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        preload_dlls=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    from fast_omniasr.backends.onnx import ONNXBackend

    ONNXBackend("model.onnx", device="cuda")
    assert seen["providers"] == [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS)]


@pytest.mark.parametrize("arena", ["kSameAsRequested", "kNextPowerOfTwo"])
def test_llm_cuda_sessions_get_the_shared_options(monkeypatch, arena):
    onnx_backend = pytest.importorskip("fast_omniasr.llm.onnx_backend")
    monkeypatch.setattr(onnx_backend.ort, "preload_dlls", lambda **kwargs: None)
    monkeypatch.setattr(onnx_backend.ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    [(name, options)] = onnx_backend.providers("cuda", arena)
    assert name == "CUDAExecutionProvider"
    assert options == {**CUDA_PROVIDER_OPTIONS, "cudnn_conv_use_max_workspace": "0",
                       "arena_extend_strategy": arena}

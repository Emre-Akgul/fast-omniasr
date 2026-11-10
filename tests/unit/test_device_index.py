"""GPU selection with device_index across the CTC, LLM and TensorRT runtimes (#6)."""

import sys
import types

import numpy as np
import pytest

from fast_omniasr import OmniASR
from fast_omniasr.options import (
    CUDA_PROVIDER_OPTIONS,
    check_device_index,
    cuda_provider_options,
)


@pytest.mark.parametrize("value", [0, 3, np.int64(1)])
def test_device_index_accepts_gpu_ordinals_for_cuda(value):
    assert check_device_index(value, "cuda") == int(value)


def test_device_index_defaults_to_none():
    assert check_device_index(None, "cpu") is None


@pytest.mark.parametrize("value", [-1, 1.0, "1", True])
def test_device_index_rejects_non_ordinals(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        check_device_index(value, "cuda")


def test_device_index_needs_cuda():
    with pytest.raises(ValueError, match="only applies to device='cuda'"):
        check_device_index(0, "cpu")


def test_provider_options_name_the_gpu():
    assert cuda_provider_options() == CUDA_PROVIDER_OPTIONS
    assert cuda_provider_options(2) == {**CUDA_PROVIDER_OPTIONS, "device_id": "2"}
    assert "device_id" not in CUDA_PROVIDER_OPTIONS  # the shared dict is never changed


def fake_onnxruntime(monkeypatch):
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
    return seen


def test_ctc_onnx_session_runs_on_the_chosen_gpu(monkeypatch):
    seen = fake_onnxruntime(monkeypatch)
    from fast_omniasr.backends.onnx import ONNXBackend

    ONNXBackend("model.onnx", device="cuda", device_index=1)
    assert seen["providers"] == [("CUDAExecutionProvider",
                                  {**CUDA_PROVIDER_OPTIONS, "device_id": "1"})]
    ONNXBackend("model.onnx", device="cuda")
    assert seen["providers"] == [("CUDAExecutionProvider", CUDA_PROVIDER_OPTIONS)]


def test_ctc_rejects_device_index_before_loading():
    with pytest.raises(ValueError, match="only applies to device='cuda'"):
        OmniASR("missing.onnx", "missing.model", device_index=1)


def test_from_pretrained_rejects_device_index_before_any_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # a download would fail loudly
    with pytest.raises(ValueError, match="only applies to device='cuda'"):
        OmniASR.from_pretrained("o/r", device="cpu", device_index=1)
    with pytest.raises(ValueError, match="non-negative integer"):
        OmniASR.from_pretrained("o/r", backend="tensorrt", precision="fp16", device_index=-1)


@pytest.fixture
def llm_backend():
    return pytest.importorskip("fast_omniasr.llm.onnx_backend")


def test_llm_sessions_and_shrinkage_name_the_gpu(monkeypatch, llm_backend):
    monkeypatch.setattr(llm_backend.ort, "preload_dlls", lambda **kwargs: None)
    monkeypatch.setattr(llm_backend.ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    [(_, options)] = llm_backend.providers("cuda", "kNextPowerOfTwo", 3)
    assert options["device_id"] == "3" and options["arena_extend_strategy"] == "kNextPowerOfTwo"
    entries = []
    monkeypatch.setattr(llm_backend.ort, "RunOptions", lambda: types.SimpleNamespace(
        add_run_config_entry=lambda key, value: entries.append((key, value))))
    llm_backend.run_options("cuda", 3)
    llm_backend.run_options("cuda")
    assert entries == [("memory.enable_memory_arena_shrinkage", "gpu:3"),
                       ("memory.enable_memory_arena_shrinkage", "gpu:0")]


def test_llm_bound_outputs_stay_on_the_chosen_gpu(llm_backend):
    bound = []

    class Binding:
        def bind_cpu_input(self, name, value):
            pass

        def bind_output(self, name, device_type="cpu", device_id=0):
            bound.append((name, device_type, device_id))

        def get_outputs(self):
            return [types.SimpleNamespace(numpy=lambda: np.zeros(1)), "kv"]

    graph = types.SimpleNamespace(
        io_binding=Binding,
        get_outputs=lambda: [types.SimpleNamespace(name="logits"),
                             types.SimpleNamespace(name="present")],
        run_with_iobinding=lambda binding: None,
    )
    backend = llm_backend.ONNXLLMBackend.__new__(llm_backend.ONNXLLMBackend)
    backend.device_index = 2
    backend._run_bound(graph, {"tokens": np.zeros((1, 1), np.int64)})
    assert bound == [("logits", "cpu", 0), ("present", "cuda", 2)]


def test_torchscript_llm_rejects_device_index(tmp_path):
    from fast_omniasr.llm import OmniASRLLM

    with pytest.raises(ValueError, match="only applies to device='cuda'"):
        OmniASRLLM(tmp_path, device_index=0)


class FakeCudart:
    def __init__(self, current=0, count=2):
        self.current, self.count, self.calls = current, count, []

    def cudaGetDevice(self):
        return (0, self.current)

    def cudaGetDeviceCount(self):
        return (0, self.count)

    def cudaSetDevice(self, index):
        self.calls.append(index)
        self.current = index
        return (0,)


@pytest.fixture
def cudart(monkeypatch):
    fake = FakeCudart()
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = fake
    monkeypatch.setitem(sys.modules, "cuda", types.ModuleType("cuda"))
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", fake)
    fake_driver(monkeypatch, active={0: True})  # by default the caller's GPU is in use
    return fake


def test_cuda_device_selects_and_restores(cudart):
    from fast_omniasr.backends.tensorrt import cuda_device

    with cuda_device(1):
        assert cudart.current == 1
    assert cudart.current == 0 and cudart.calls == [1, 0]
    with cuda_device(None):
        assert cudart.calls == [1, 0]  # untouched
    with pytest.raises(RuntimeError), cuda_device(1):
        raise RuntimeError("build failed")
    assert cudart.current == 0


def test_cuda_device_rejects_a_missing_gpu(cudart):
    from fast_omniasr.backends.tensorrt import cuda_device

    with pytest.raises(ValueError, match="only 2 CUDA device"), cuda_device(2):
        pass
    assert cudart.calls == []


def test_tensorrt_engine_loads_on_the_chosen_gpu(cudart, monkeypatch):
    from fast_omniasr import model as model_module
    from fast_omniasr.backends import tensorrt

    loaded_on = []

    class Backend:
        vocabulary_size = None

        def __init__(self, path):
            loaded_on.append(cudart.current)

    monkeypatch.setattr(tensorrt, "TensorRTBackend", Backend)
    monkeypatch.setattr(model_module, "Tokenizer", lambda path: object())
    monkeypatch.setattr(model_module, "check_compatible", lambda *args, **kwargs: None)
    OmniASR("e.engine", "t.model", backend="tensorrt", device_index=1)
    assert loaded_on == [1] and cudart.current == 0


def test_a_missing_gpu_names_the_device_index():
    from fast_omniasr.options import provider_failure

    assert "on device_index=1" in provider_failure("CUDAExecutionProvider", 1)
    assert provider_failure("CUDAExecutionProvider", None).endswith("did not initialize")


def fake_driver(monkeypatch, active):
    driver = types.SimpleNamespace(
        cuInit=lambda flags: (0,),
        cuDevicePrimaryCtxGetState=lambda device: (0, 0, int(active.get(device, False))),
    )
    sys.modules["cuda.bindings"].driver = driver
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)


def test_an_unused_previous_device_is_not_woken_up(cudart, monkeypatch):
    from fast_omniasr.backends.tensorrt import cuda_device

    fake_driver(monkeypatch, active={})  # GPU 0 was never used by this process
    with cuda_device(1):
        assert cudart.current == 1
    assert cudart.calls == [1]  # no cudaSetDevice(0), so no context on GPU 0


def test_a_previous_device_in_use_is_restored(cudart, monkeypatch):
    from fast_omniasr.backends.tensorrt import cuda_device

    fake_driver(monkeypatch, active={0: True})
    with cuda_device(1):
        pass
    assert cudart.calls == [1, 0]


def test_selecting_the_current_device_switches_nothing(cudart, monkeypatch):
    from fast_omniasr.backends.tensorrt import cuda_device

    fake_driver(monkeypatch, active={0: True})
    with cuda_device(0):
        pass
    assert cudart.calls == [0]

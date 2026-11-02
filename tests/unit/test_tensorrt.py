import builtins
import gc
import sys
import types
import weakref

import numpy as np
import pytest

from fast_omniasr.backends.tensorrt import TensorRTBackend


def test_missing_tensorrt_raises_helpful_import_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("simulated: tensorrt not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[tensorrt\\]"):
        TensorRTBackend("nonexistent.engine")


# --- close() resource release, exercised with fake tensorrt / cuda-python modules ---


class _FakeCudart:
    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def __init__(self):
        self.calls = []
        self._next = 100

    def _ptr(self):
        self._next += 1
        return self._next

    def cudaStreamCreate(self):
        self.calls.append("stream_create")
        return 0, self._ptr()

    def cudaMalloc(self, size):
        ptr = self._ptr()
        self.calls.append(("malloc", ptr))
        return 0, ptr

    def cudaFree(self, ptr):
        self.calls.append(("free", ptr))
        return (0,)

    def cudaStreamSynchronize(self, stream):
        self.calls.append("stream_sync")
        return (0,)

    def cudaStreamDestroy(self, stream):
        self.calls.append("stream_destroy")
        return (0,)


class _FakeContext:
    def __init__(self, engine):
        self.engine = engine  # like TensorRT: the context keeps the engine alive

    def set_input_shape(self, name, shape):
        self.shape = shape
        return True

    def get_tensor_shape(self, name):
        return (1, self.shape[1] // 320, 8)

    def set_tensor_address(self, name, ptr):
        return True


class _FakeEngine:
    def __init__(self, fail_context=False):
        self.fail_context = fail_context

    def create_execution_context(self):
        return None if self.fail_context else _FakeContext(self)

    def get_tensor_dtype(self, name):
        return "float32"

    def get_tensor_profile_shape(self, name, profile):
        return (1, 400), (1, 16000), (1, 32000)


class _FakeRuntime:
    engine_factory = _FakeEngine

    def __init__(self, logger):
        pass

    def deserialize_cuda_engine(self, data):
        return self.engine_factory()


@pytest.fixture
def fake_trt(monkeypatch, tmp_path):
    trt = types.ModuleType("tensorrt")
    trt.float32 = "float32"
    trt.Logger = type("Logger", (), {"WARNING": 0, "__init__": lambda self, level: None})
    trt.Runtime = _FakeRuntime
    cudart = _FakeCudart()
    cuda = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = cudart
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"fake")
    return types.SimpleNamespace(trt=trt, cudart=cudart, path=engine_path)


def test_close_releases_engine_context_and_runtime(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    engine_ref = weakref.ref(backend.engine)
    context_ref = weakref.ref(backend.context)
    runtime_ref = weakref.ref(backend.runtime)
    allocated = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "malloc"]

    backend.close()
    gc.collect()

    # The backend is still referenced, yet every TensorRT object is gone.
    assert engine_ref() is None
    assert context_ref() is None
    assert runtime_ref() is None
    freed = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "free"]
    assert sorted(freed) == sorted(allocated)
    calls = fake_trt.cudart.calls
    assert calls.index("stream_sync") < calls.index(("free", allocated[0]))
    assert calls[-1] == "stream_destroy"
    assert backend.input_ptr is None and backend.output_ptr is None and backend.stream is None


def test_close_is_idempotent_and_infer_after_close_raises(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    backend.close()
    calls_after_first_close = list(fake_trt.cudart.calls)
    backend.close()
    assert fake_trt.cudart.calls == calls_after_first_close
    with pytest.raises(RuntimeError, match="closed"):
        backend.infer(np.zeros((1, 16000), dtype=np.float32))


def test_partial_init_failure_releases_engine(fake_trt, monkeypatch):
    engines = []

    def failing_engine():
        engine = _FakeEngine(fail_context=True)
        engines.append(weakref.ref(engine))
        return engine

    monkeypatch.setattr(_FakeRuntime, "engine_factory", staticmethod(failing_engine))
    with pytest.raises(RuntimeError, match="execution context"):
        TensorRTBackend(fake_trt.path)
    gc.collect()
    assert engines and engines[0]() is None
    # Nothing was allocated, so nothing should have been freed.
    assert not any(isinstance(c, tuple) and c[0] == "free" for c in fake_trt.cudart.calls)


def test_omniasr_close_releases_tensorrt_resources(fake_trt, monkeypatch):
    from fast_omniasr import model as model_module

    monkeypatch.setattr(model_module, "Tokenizer", lambda path: object())
    asr = model_module.OmniASR(fake_trt.path, "tokenizer.model", backend="tensorrt")
    engine_ref = weakref.ref(asr.backend.engine)
    with asr:
        pass
    gc.collect()
    assert engine_ref() is None
    assert asr.backend.closed


def test_close_releases_everything_even_if_a_cuda_call_fails(fake_trt, monkeypatch):
    backend = TensorRTBackend(fake_trt.path)
    engine_ref = weakref.ref(backend.engine)
    allocated = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "malloc"]

    def broken_sync(stream):
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(fake_trt.cudart, "cudaStreamSynchronize", broken_sync)
    with pytest.raises(RuntimeError, match="simulated"):
        backend.close()
    gc.collect()
    assert engine_ref() is None
    assert backend.closed
    freed = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "free"]
    assert sorted(freed) == sorted(allocated)
    assert fake_trt.cudart.calls[-1] == "stream_destroy"

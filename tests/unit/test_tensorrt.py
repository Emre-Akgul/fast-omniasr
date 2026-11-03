import builtins
import ctypes
import gc
import sys
import threading
import time
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
        self.memory = {}  # fake device memory: pointer -> bytes
        self.current_device = 0
        self.set_device_calls = []

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

    def cudaMemcpyAsync(self, dst, src, nbytes, kind, stream):
        self.calls.append("memcpy")
        if kind == self.cudaMemcpyKind.cudaMemcpyHostToDevice:
            self.memory[dst] = ctypes.string_at(src, nbytes)
        else:
            data = self.memory.get(src, b"")[:nbytes].ljust(nbytes, b"\0")
            ctypes.memmove(dst, data, nbytes)
        return (0,)

    def cudaGetDevice(self):
        return 0, self.current_device

    def cudaSetDevice(self, device):
        self.set_device_calls.append((threading.get_ident(), device))
        self.current_device = device
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
        return self.engine.logits_shape(self.shape[1])

    def set_tensor_address(self, name, ptr):
        self.addresses = getattr(self, "addresses", {})
        self.addresses[name] = ptr
        return True

    def execute_async_v3(self, stream):
        """Logits filled with the first input sample, written after a delay so that
        unsynchronized concurrent callers would observe each other's data."""
        engine = self.engine
        engine.active += 1
        engine.max_active = max(engine.max_active, engine.active)
        memory = engine.cudart.memory
        first = np.frombuffer(memory[self.addresses["audio"]][:4], dtype=np.float32)[0]
        time.sleep(engine.execute_delay)
        count = int(np.prod(self.get_tensor_shape("logits")))
        memory[self.addresses["logits"]] = np.full(count, first, np.float32).tobytes()
        engine.active -= 1
        return True


class _FakeEngine:
    """A well-formed batch-one CTC engine; tests override attributes to break it."""

    def __init__(self, fail_context=False):
        self.fail_context = fail_context
        self.io = {"audio": ("INPUT", "float32", "DEVICE", (1, -1)),
                   "logits": ("OUTPUT", "float32", "DEVICE", (1, -1, 8))}
        self.num_optimization_profiles = 1
        self.profile = ((1, 400), (1, 16000), (1, 32000))
        self.logits_shape = lambda samples: (1, samples // 320, 8)
        self.cudart = None  # set by the fixture so execute can reach fake device memory
        self.active = self.max_active = 0
        self.execute_delay = 0.0

    @property
    def num_io_tensors(self):
        return len(self.io)

    def get_tensor_name(self, index):
        return list(self.io)[index]

    def get_tensor_mode(self, name):
        return self.io[name][0]

    def get_tensor_dtype(self, name):
        return self.io[name][1]

    def get_tensor_location(self, name):
        return self.io[name][2]

    def get_tensor_shape(self, name):
        return self.io[name][3]

    def create_execution_context(self):
        return None if self.fail_context else _FakeContext(self)

    def get_tensor_profile_shape(self, name, profile):
        return self.profile


class _FakeRuntime:
    engine_factory = _FakeEngine

    def __init__(self, logger):
        pass

    def deserialize_cuda_engine(self, data):
        engine = self.engine_factory()
        engine.cudart = sys.modules["cuda.bindings"].runtime
        return engine


@pytest.fixture
def fake_trt(monkeypatch, tmp_path):
    trt = types.ModuleType("tensorrt")
    trt.float32 = "float32"

    class _Mode(str):
        @property
        def name(self):
            return str(self)

    trt.TensorIOMode = types.SimpleNamespace(INPUT=_Mode("INPUT"), OUTPUT=_Mode("OUTPUT"))
    trt.TensorLocation = types.SimpleNamespace(DEVICE="DEVICE", HOST="HOST")
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


# --- engine I/O contract is validated before any device allocation (issue #5) ---

def _break(monkeypatch, mutate):
    def factory():
        engine = _FakeEngine()
        mutate(engine)
        return engine

    monkeypatch.setattr(_FakeRuntime, "engine_factory", staticmethod(factory))


def _device_allocations(cudart):
    return [c for c in cudart.calls if c == "stream_create" or (isinstance(c, tuple)
                                                                 and c[0] == "malloc")]


def _io(**changes):
    def mutate(engine):
        for name, value in changes.items():
            if value is None:
                del engine.io[name]
            else:
                engine.io[name] = value
    return mutate


@pytest.mark.parametrize("mutate, message", [
    (_io(extra=("INPUT", "float32", "DEVICE", (1, -1))), "Expected I/O tensors"),
    (_io(logits=None), "Expected I/O tensors"),
    (_io(audio=None, wav=("INPUT", "float32", "DEVICE", (1, -1))), "Expected I/O tensors"),
    (_io(audio=("OUTPUT", "float32", "DEVICE", (1, -1))), "'audio' to be an input"),
    (_io(logits=("INPUT", "float32", "DEVICE", (1, -1, 8))), "'logits' to be an output"),
    (_io(logits=("OUTPUT", "float16", "DEVICE", (1, -1, 8))),
     "'logits' to be float32, engine has float16"),
    (lambda e: setattr(e, "logits_shape", lambda n: (1, n // 1000, 8)),
     "zero frames at the profile minimum of 400 samples"),
    (_io(audio=("INPUT", "float32", "HOST", (1, -1))), "device memory"),
    (_io(audio=("INPUT", "float32", "DEVICE", (-1,))), "'audio' to have rank 2"),
    (_io(logits=("OUTPUT", "float32", "DEVICE", (1, -1))), "'logits' to have rank 3"),
    (_io(audio=("INPUT", "float32", "DEVICE", (4, -1))), "batch dimension 1"),
    (lambda e: setattr(e, "num_optimization_profiles", 0), "no optimization profile"),
    (lambda e: setattr(e, "profile", ((2, 400), (2, 16000), (2, 32000))), "batch-one"),
    (lambda e: setattr(e, "profile", ((1, 400, 1),) * 3), "batch-one"),
    (lambda e: setattr(e, "profile", ((1, 32000), (1, 16000), (1, 400))), "sample bounds"),
    (lambda e: setattr(e, "logits_shape", lambda n: (1, -1, 8)), "not a resolved"),
    (lambda e: setattr(e, "logits_shape", lambda n: (2, n // 320, 8)), "not a resolved"),
    (lambda e: setattr(e, "logits_shape", lambda n: (1, n // 320)), "not a resolved"),
    (lambda e: setattr(e, "logits_shape", lambda n: (1, n // 320, 8 if n < 1000 else 9)),
     "Vocabulary size changes"),
])
def test_invalid_engine_contract_rejected_before_allocation(fake_trt, monkeypatch, mutate,
                                                            message):
    _break(monkeypatch, mutate)
    with pytest.raises(ValueError, match=message):
        TensorRTBackend(fake_trt.path)
    assert _device_allocations(fake_trt.cudart) == []


def test_valid_engine_contract_sizes_buffers_from_profile(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    assert (backend.min_samples, backend.max_samples) == (400, 32000)
    mallocs = [c for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "malloc"]
    assert len(mallocs) == 2
    assert backend.max_output_elements == 1 * (32000 // 320) * 8
    backend.close()


def test_infer_refuses_output_larger_than_buffer(fake_trt, monkeypatch):
    backend = TensorRTBackend(fake_trt.path)
    # A misbehaving engine that resolves a bigger output for a shorter input.
    backend.engine.logits_shape = lambda n: (1, 10_000, 8)
    with pytest.raises(RuntimeError, match="does not fit"):
        backend.infer(np.zeros((1, 16000), dtype=np.float32))
    assert not any(c for c in fake_trt.cudart.calls if c == "memcpy")
    backend.close()


# --- concurrent callers are serialized and never mix requests (issue #1) ---

def _run_concurrently(backend, values, calls_per_thread=4):
    results, errors = {}, []
    barrier = threading.Barrier(len(values))

    def worker(value):
        try:
            barrier.wait()
            for i in range(calls_per_thread):
                samples = 16000 + 320 * (int(value) % 7)  # varied lengths per caller
                out = backend.infer(np.full((1, samples), value, np.float32))
                results[(value, i)] = out
        except Exception as exc:  # noqa: BLE001 -- surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(v,)) for v in values]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    return results


def test_concurrent_infer_matches_sequential(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    backend.engine.execute_delay = 0.002
    values = [float(v) for v in range(1, 9)]
    sequential = {v: backend.infer(np.full((1, 16000 + 320 * (int(v) % 7)), v, np.float32))
                  for v in values}
    concurrent = _run_concurrently(backend, values)
    assert backend.engine.max_active == 1  # executions never overlapped
    for (value, _), out in concurrent.items():
        assert out.shape == sequential[value].shape
        assert np.array_equal(out, sequential[value])  # no request saw another's data
    backend.close()


def test_close_waits_for_in_flight_infer(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    backend.engine.execute_delay = 0.2
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(out=backend.infer(np.full((1, 16000), 5.0, np.float32)))
    )
    worker.start()
    deadline = time.monotonic() + 10
    while backend.engine.active == 0:  # wait until execution is in progress
        assert time.monotonic() < deadline, "worker never started executing"
        time.sleep(0.001)
    backend.close()  # must block until the in-flight call has copied its output
    worker.join()
    assert np.all(result["out"] == 5.0)
    calls = fake_trt.cudart.calls
    last_memcpy = max(i for i, c in enumerate(calls) if c == "memcpy")
    first_free = min(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "free")
    assert last_memcpy < first_free
    with pytest.raises(RuntimeError, match="closed"):
        backend.infer(np.zeros((1, 16000), np.float32))


def test_infer_uses_the_load_device_and_restores_the_callers(fake_trt):
    fake_trt.cudart.current_device = 3
    backend = TensorRTBackend(fake_trt.path)
    assert backend.device == 3
    fake_trt.cudart.current_device = 0  # another thread's current device
    worker = threading.Thread(target=backend.infer, args=(np.zeros((1, 16000), np.float32),))
    worker.start()
    worker.join()
    assert fake_trt.cudart.set_device_calls == [(worker.ident, 3), (worker.ident, 0)]
    assert fake_trt.cudart.current_device == 0  # caller's device restored
    fake_trt.cudart.set_device_calls.clear()
    backend.close()  # release also runs on the load device, then restores
    assert [d for _, d in fake_trt.cudart.set_device_calls] == [3, 0]


def test_infer_on_the_load_device_does_not_switch(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    backend.infer(np.zeros((1, 16000), np.float32))
    backend.close()
    assert fake_trt.cudart.set_device_calls == []


def test_finalizer_never_blocks_on_a_busy_lock(fake_trt):
    backend = TensorRTBackend(fake_trt.path)
    backend._lock.acquire()  # e.g. a daemon thread killed mid-infer at shutdown
    try:
        start = time.monotonic()
        backend.__del__()
        assert time.monotonic() - start < 1
        assert not backend.closed  # skipped rather than waiting
    finally:
        backend._lock.release()
    backend.__del__()
    assert backend.closed


def test_close_still_releases_if_device_query_fails(fake_trt, monkeypatch):
    fake_trt.cudart.current_device = 3
    backend = TensorRTBackend(fake_trt.path)
    engine_ref = weakref.ref(backend.engine)
    allocated = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "malloc"]
    monkeypatch.setattr(fake_trt.cudart, "cudaGetDevice", lambda: (1, None))
    backend.close()  # e.g. CUDA partly torn down: release on the current device instead
    gc.collect()
    freed = [c[1] for c in fake_trt.cudart.calls if isinstance(c, tuple) and c[0] == "free"]
    assert sorted(freed) == sorted(allocated)
    assert fake_trt.cudart.calls[-1] == "stream_destroy"
    assert engine_ref() is None


def test_tensorrt_objects_are_dropped_on_the_load_device(fake_trt, monkeypatch):
    fake_trt.cudart.current_device = 3
    backend = TensorRTBackend(fake_trt.path)
    fake_trt.cudart.current_device = 0
    seen = []
    real_drop = backend._drop_tensorrt_objects

    def recording_drop():
        seen.append(fake_trt.cudart.current_device)
        real_drop()

    monkeypatch.setattr(backend, "_drop_tensorrt_objects", recording_drop)
    backend.close()
    assert seen[0] == 3  # engine/context/runtime destroyed while on the load device
    assert fake_trt.cudart.current_device == 0



def test_undeserializable_engine_raises_the_dedicated_error(fake_trt, monkeypatch):
    from fast_omniasr.backends.tensorrt import EngineDeserializationError

    monkeypatch.setattr(_FakeRuntime, "deserialize_cuda_engine", lambda self, data: None)
    with pytest.raises(EngineDeserializationError, match="Failed to deserialize"):
        TensorRTBackend(fake_trt.path)
    assert issubclass(EngineDeserializationError, RuntimeError)  # still a RuntimeError

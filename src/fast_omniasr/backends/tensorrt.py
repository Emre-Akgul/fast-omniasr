"""TensorRT engine inference. No PyTorch, ONNX Runtime, or training-framework imports."""
import contextlib
import sys
import threading
from pathlib import Path

import numpy as np

_INPUT, _OUTPUT = "audio", "logits"


class EngineDeserializationError(RuntimeError):
    """TensorRT could not deserialize the engine file (corrupt, from another TensorRT
    version, or transiently out of GPU memory)."""


def _validate_contract(trt, engine, context):
    """Check the engine matches the batch-one CTC contract this backend assumes:
    exactly one float32 device input `audio` [1, samples] with a usable optimization
    profile, and one float32 device output `logits` [1, frames, vocabulary] whose shape
    fully resolves. Returns (min_samples, max_samples, logits shape at max_samples).
    Raises ValueError naming the first mismatch."""
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    if sorted(names) != [_INPUT, _OUTPUT]:
        raise ValueError(f"Expected I/O tensors {[_INPUT, _OUTPUT]}, engine has {names}")
    for name, mode in ((_INPUT, trt.TensorIOMode.INPUT), (_OUTPUT, trt.TensorIOMode.OUTPUT)):
        if engine.get_tensor_mode(name) != mode:
            raise ValueError(f"Expected {name!r} to be an {mode.name.lower()} tensor")
        dtype = engine.get_tensor_dtype(name)
        if dtype != trt.float32:
            raise ValueError(
                f"Expected {name!r} to be float32, engine has {getattr(dtype, 'name', dtype)}"
            )
        if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
            raise ValueError(f"Expected {name!r} to live in device memory")
    for name, rank in ((_INPUT, 2), (_OUTPUT, 3)):
        shape = tuple(engine.get_tensor_shape(name))
        if len(shape) != rank or shape[0] not in (1, -1):
            raise ValueError(
                f"Expected {name!r} to have rank {rank} with batch dimension 1, got {shape}"
            )
    if engine.num_optimization_profiles < 1:
        raise ValueError("Engine has no optimization profile")
    # Only profile 0 is validated and used: it is the execution context's default, and
    # the project's builder creates exactly one. Additional profiles are ignored.
    min_shape, opt_shape, max_shape = (
        tuple(shape) for shape in engine.get_tensor_profile_shape(_INPUT, 0)
    )
    if any(len(shape) != 2 or shape[0] != 1 for shape in (min_shape, opt_shape, max_shape)):
        raise ValueError(
            f"Expected a batch-one [1, samples] profile, got min {min_shape}, "
            f"opt {opt_shape}, max {max_shape}"
        )
    if not 1 <= min_shape[1] <= opt_shape[1] <= max_shape[1]:
        raise ValueError(f"Invalid profile sample bounds {min_shape}/{opt_shape}/{max_shape}")
    resolved = {}
    for label, shape in (("minimum", min_shape), ("maximum", max_shape)):
        if not context.set_input_shape(_INPUT, shape):
            raise RuntimeError(f"Failed to size the execution context to the profile {label}")
        logits = tuple(context.get_tensor_shape(_OUTPUT))
        if len(logits) == 3 and logits[0] == 1 and logits[1] == 0 and logits[2] >= 1:
            raise ValueError(
                f"Engine produces zero frames at the profile {label} of {shape[1]} samples; "
                "rebuild it with a larger min_samples"
            )
        if len(logits) != 3 or logits[0] != 1 or min(logits) < 1:
            raise ValueError(
                f"Logits shape {logits} at the profile {label} is not a resolved "
                "[1, frames, vocabulary] shape"
            )
        resolved[label] = logits
    if resolved["minimum"][2] != resolved["maximum"][2]:
        raise ValueError(
            f"Vocabulary size changes with input length: {resolved['minimum'][2]} at the "
            f"minimum vs {resolved['maximum'][2]} at the maximum"
        )
    return min_shape[1], max_shape[1], resolved["maximum"]


class TensorRTBackend:
    """Batch-one TensorRT engine runner.

    Thread-safe: one instance owns a single execution context, CUDA stream and pair of
    device buffers, so concurrent `infer()` calls are serialized by a per-instance lock
    (they never overlap or mix requests). For parallel GPU throughput, create one
    instance per worker. `close()` waits for an in-flight `infer()` to finish, so don't
    call it from a signal handler that may interrupt `infer()` on the same thread.
    GPU work runs on the device that was current when the engine was loaded; the
    calling thread's current device is restored afterwards.
    """

    def __init__(self, path: str | Path):
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "Install fast-omniasr[tensorrt] and a compatible TensorRT installation"
            ) from exc
        try:
            from cuda.bindings import runtime as cudart
        except ImportError as exc:
            raise ImportError("Install fast-omniasr[tensorrt] for the cuda-python dependency") from exc
        self._cudart = cudart
        self._lock = threading.Lock()
        self.closed = False  # set early so close()/__del__ can clean up a partial init
        # The engine lives on the device current at load time; the current device is
        # per-thread, so infer() re-selects it for callers on other threads.
        self.device = self._check(cudart.cudaGetDevice())
        logger = trt.Logger(trt.Logger.WARNING)
        # Kept on the instance: the runtime must outlive the engine it deserialized.
        self.runtime = runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if engine is None:
            raise EngineDeserializationError("Failed to deserialize TensorRT engine")
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        # Validate the whole I/O contract before allocating anything on the device.
        self.min_samples, self.max_samples, max_output_shape = _validate_contract(
            trt, engine, self.context
        )
        self.stream = self._check(cudart.cudaStreamCreate())
        self.input_ptr = self._check(cudart.cudaMalloc(self.max_samples * np.dtype(np.float32).itemsize))
        max_output_elements = int(np.prod(max_output_shape))
        self.max_output_elements = max_output_elements
        self.output_ptr = self._check(
            cudart.cudaMalloc(max_output_elements * np.dtype(np.float32).itemsize)
        )
        if not self.context.set_tensor_address("audio", self.input_ptr):
            raise RuntimeError("Failed to bind the audio tensor address")
        if not self.context.set_tensor_address("logits", self.output_ptr):
            raise RuntimeError("Failed to bind the logits tensor address")

    def close(self) -> None:
        """Release all GPU resources: device buffers, the CUDA stream, the execution context,
        the engine, and the runtime. Safe to call more than once, and safe on a
        partially-constructed instance (e.g. if __init__ raised partway through).

        Resources are released even if callers still hold a reference to this backend
        (or to the OmniASR wrapping it); later infer() calls raise RuntimeError."""
        if getattr(self, "closed", True):
            return
        with self._lock:  # wait for an in-flight infer() before releasing its buffers
            if self.closed:
                return
            self.closed = True
            self._release()

    @contextlib.contextmanager
    def _on_device(self, *, best_effort: bool = False):
        """Run on this engine's device, restoring the caller thread's current device.
        With `best_effort` (used when releasing), a failed device query runs the body on
        the current device instead of raising, so cleanup is still attempted."""
        device = getattr(self, "device", None)
        if device is None:  # partial init: nothing was created on a device yet
            yield
            return
        try:
            previous = self._check(self._cudart.cudaGetDevice())
            if previous != device:
                self._check(self._cudart.cudaSetDevice(device))
        except RuntimeError:
            if not best_effort:
                raise
            yield
            return
        try:
            yield
        finally:
            if previous != device:
                self._cudart.cudaSetDevice(previous)

    def _release(self) -> None:
        try:
            with self._on_device(best_effort=True):
                try:
                    self._release_on_device()
                finally:
                    # Inside the device switch so TensorRT's destructors free the
                    # engine's device memory on its own device.
                    self._drop_tensorrt_objects()
        finally:
            self._drop_tensorrt_objects()  # no-op unless the device switch itself failed

    def _drop_tensorrt_objects(self) -> None:
        # Dependency order: context -> engine -> runtime. The Python bindings free the
        # underlying C++ objects (and the engine's device memory) on the last reference.
        self.context = None
        self.engine = None
        self.runtime = None

    def _release_on_device(self) -> None:
        cudart = self._cudart
        stream = getattr(self, "stream", None)
        input_ptr = getattr(self, "input_ptr", None)
        output_ptr = getattr(self, "output_ptr", None)
        self.input_ptr = self.output_ptr = self.stream = None
        # Every step is attempted even if an earlier one raises; the first error is
        # re-raised at the end so a single failing CUDA call cannot leak the rest.
        steps = []
        if stream is not None:
            # Don't free buffers that queued work may still be reading or writing.
            steps.append((cudart.cudaStreamSynchronize, stream))
        steps += [(cudart.cudaFree, ptr) for ptr in (input_ptr, output_ptr) if ptr is not None]
        if stream is not None:
            steps.append((cudart.cudaStreamDestroy, stream))
        first_error = None
        for func, arg in steps:
            try:
                func(arg)
            except Exception as exc:  # noqa: BLE001 -- keep releasing, re-raise below
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __del__(self):
        # Never block in a finalizer: during interpreter shutdown a daemon thread may be
        # killed while holding the lock (waiting would hang or abort the process), and
        # the OS reclaims GPU memory at exit anyway. Skip cleanup if the lock is busy.
        if getattr(self, "closed", True) or sys.is_finalizing():
            return
        lock = getattr(self, "_lock", None)
        if lock is None or not lock.acquire(blocking=False):
            return
        try:
            if not self.closed:
                self.closed = True
                self._release()
        except Exception:  # noqa: S110, BLE001 -- best effort; cuda module may be torn down
            pass
        finally:
            lock.release()

    def _check(self, result):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"CUDA call failed: {error}")
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        if self.closed:
            raise RuntimeError("TensorRTBackend is closed")
        waveform = np.asarray(waveform)
        if waveform.dtype != np.float32 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Expected normalized float32 waveform [1, samples]")
        samples = waveform.shape[1]
        if not (self.min_samples <= samples <= self.max_samples):
            raise ValueError(
                f"Waveform length {samples} is outside the engine profile "
                f"[{self.min_samples}, {self.max_samples}]"
            )
        waveform = np.ascontiguousarray(waveform)
        with self._lock:  # the context, stream and buffers are shared per instance
            if self.closed:
                raise RuntimeError("TensorRTBackend is closed")
            return self._infer_locked(waveform, samples)

    def _infer_locked(self, waveform: np.ndarray, samples: int) -> np.ndarray:
        with self._on_device():
            return self._infer_on_device(waveform, samples)

    def _infer_on_device(self, waveform: np.ndarray, samples: int) -> np.ndarray:
        cudart = self._cudart
        if not self.context.set_input_shape("audio", (1, samples)):
            raise RuntimeError("Failed to set the dynamic input shape")
        output_shape = tuple(self.context.get_tensor_shape("logits"))
        if (len(output_shape) != 3 or output_shape[0] != 1 or min(output_shape) < 1
                or int(np.prod(output_shape)) > self.max_output_elements):
            # Never copy more than the buffer sized at load time.
            raise RuntimeError(
                f"Engine resolved logits shape {output_shape} for {samples} samples, which "
                f"does not fit the {self.max_output_elements}-element output buffer"
            )
        output = np.empty(output_shape, dtype=np.float32)
        self._check(
            cudart.cudaMemcpyAsync(
                self.input_ptr, waveform.ctypes.data, waveform.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            )
        )
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("TensorRT execution failed")
        self._check(
            cudart.cudaMemcpyAsync(
                output.ctypes.data, self.output_ptr, output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )
        )
        self._check(cudart.cudaStreamSynchronize(self.stream))
        return output

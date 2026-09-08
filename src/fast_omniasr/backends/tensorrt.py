"""TensorRT engine inference. No PyTorch, ONNX Runtime, or training-framework imports."""
from pathlib import Path

import numpy as np


class TensorRTBackend:
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
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        if engine.get_tensor_dtype("audio") != trt.float32 or engine.get_tensor_dtype("logits") != trt.float32:
            raise ValueError("Expected a float32 audio input and a float32 logits output")
        min_shape, _, max_shape = engine.get_tensor_profile_shape("audio", 0)
        self.min_samples, self.max_samples = min_shape[1], max_shape[1]
        if not self.context.set_input_shape("audio", tuple(max_shape)):
            raise RuntimeError("Failed to size the execution context to the profile maximum")
        max_output_shape = tuple(self.context.get_tensor_shape("logits"))
        self.stream = self._check(cudart.cudaStreamCreate())
        self.input_ptr = self._check(cudart.cudaMalloc(max_shape[1] * np.dtype(np.float32).itemsize))
        max_output_elements = int(np.prod(max_output_shape))
        self.output_ptr = self._check(
            cudart.cudaMalloc(max_output_elements * np.dtype(np.float32).itemsize)
        )
        if not self.context.set_tensor_address("audio", self.input_ptr):
            raise RuntimeError("Failed to bind the audio tensor address")
        if not self.context.set_tensor_address("logits", self.output_ptr):
            raise RuntimeError("Failed to bind the logits tensor address")

    def _check(self, result):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f"CUDA call failed: {error}")
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        cudart = self._cudart
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
        if not self.context.set_input_shape("audio", (1, samples)):
            raise RuntimeError("Failed to set the dynamic input shape")
        output_shape = tuple(self.context.get_tensor_shape("logits"))
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

from pathlib import Path

import numpy as np

from .._memory import release_freed_memory


class ONNXBackend:
    def __init__(self, path: str | Path, device: str = "cpu", threads: int = 4,
                 precision: str = "fp32", device_index: int | None = None):
        self.closed = False
        from ..options import (
            check_device,
            check_device_index,
            check_threads,
            cuda_provider_options,
            provider_failure,
        )
        from ..precision import check_precision
        check_device(device)
        device_index = check_device_index(device_index, device)
        check_precision(precision, device)
        check_threads(threads)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Install fast-omniasr[onnx] or fast-omniasr[cuda]") from exc
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        if device == "cuda":
            ort.preload_dlls()
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"{provider} is unavailable in this ONNX Runtime installation")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        # With half-precision graphs the memory pattern planner grows the CUDA arena by
        # ~1 GiB once an input length repeats; without it fp16 300M stays at ~1.5 GiB.
        options.enable_mem_pattern = precision == "fp32"
        providers = [(provider, cuda_provider_options(device_index))] if device == "cuda" else [provider]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.session.disable_fallback()
        if self.session.get_providers()[0] != provider:
            raise RuntimeError(provider_failure(provider, device_index))
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "audio" or inputs[0].type != "tensor(float)":
            raise ValueError("Expected one float32 input named audio")
        if len(inputs[0].shape) != 2 or inputs[0].shape[0] != 1:
            raise ValueError("Expected a batch-one waveform model")
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ValueError("Expected one output named logits")
        shape = outputs[0].shape
        # The vocabulary dimension, when the graph declares it statically.
        self.vocabulary_size = shape[-1] if shape and isinstance(shape[-1], int) else None

    def close(self) -> None:
        """Release the ONNX Runtime session (and its CPU/GPU memory arena). Idempotent;
        later infer() calls raise RuntimeError."""
        self.closed = True
        if getattr(self, "session", None) is not None:
            self.session = None
            release_freed_memory()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        session = getattr(self, "session", None)
        if self.closed or session is None:
            raise RuntimeError("ONNXBackend is closed")
        waveform = np.asarray(waveform)
        if waveform.dtype != np.float32 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Expected normalized float32 waveform [1, samples]")
        return session.run(["logits"], {"audio": np.ascontiguousarray(waveform)})[0]

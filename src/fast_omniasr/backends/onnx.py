from pathlib import Path

import numpy as np


class ONNXBackend:
    def __init__(self, path: str | Path, device: str = "cpu", threads: int = 4):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if threads < 1:
            raise ValueError("threads must be positive")
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
        providers = [(provider, {"use_tf32": "0"})] if device == "cuda" else [provider]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.session.disable_fallback()
        if self.session.get_providers()[0] != provider:
            raise RuntimeError(f"Requested provider {provider} did not initialize")
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "audio" or inputs[0].type != "tensor(float)":
            raise ValueError("Expected one float32 input named audio")
        if len(inputs[0].shape) != 2 or inputs[0].shape[0] != 1:
            raise ValueError("Expected a batch-one waveform model")
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ValueError("Expected one output named logits")

    def infer(self, waveform: np.ndarray) -> np.ndarray:
        waveform = np.asarray(waveform)
        if waveform.dtype != np.float32 or waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("Expected normalized float32 waveform [1, samples]")
        return self.session.run(["logits"], {"audio": np.ascontiguousarray(waveform)})[0]

from pathlib import Path

import numpy as np

from .._memory import release_freed_memory


def _check_contract(inputs, outputs) -> bool:
    """Whether the graph is a batch graph (export_onnx.py --batch): float32 audio
    [batch, samples] and int64 audio_lengths [batch] in, logits [batch, frames, vocabulary]
    and int64 logit_lengths [batch] out. Otherwise it must be a batch-one graph: float32
    audio [1, samples] in, logits out. Raises ValueError for anything else."""
    names = [value.name for value in inputs]
    if names == ["audio", "audio_lengths"]:
        audio, lengths = inputs
        if audio.type != "tensor(float)" or len(audio.shape) != 2:
            raise ValueError("Expected a float32 audio input [batch, samples]")
        if lengths.type != "tensor(int64)" or len(lengths.shape) != 1:
            raise ValueError("Expected an int64 audio_lengths input [batch]")
        if [value.name for value in outputs] != ["logits", "logit_lengths"]:
            raise ValueError("Expected a batch graph's outputs to be logits and logit_lengths")
        return True
    if len(inputs) != 1 or inputs[0].name != "audio" or inputs[0].type != "tensor(float)":
        raise ValueError("Expected one float32 input named audio, or audio and "
                         "audio_lengths (a batch graph)")
    if len(inputs[0].shape) != 2 or inputs[0].shape[0] != 1:
        raise ValueError("Expected a batch-one waveform model")
    if len(outputs) != 1 or outputs[0].name != "logits":
        raise ValueError("Expected one output named logits")
    return False


def _check_waveform(waveform) -> np.ndarray:
    waveform = np.asarray(waveform)
    if waveform.dtype != np.float32 or waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("Expected normalized float32 waveform [1, samples]")
    return waveform


class ONNXBackend:
    """Runs a batch-one graph, or a batch graph from `export_onnx.py --batch`. Both take
    one waveform in `infer()`; only a batch graph runs `infer_batch()` as one batch
    (`batched` says which this is)."""

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
        self.batched = _check_contract(self.session.get_inputs(), self.session.get_outputs())
        shape = self.session.get_outputs()[0].shape
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
        waveform = np.ascontiguousarray(_check_waveform(waveform))
        feeds = {"audio": waveform}
        if self.batched:
            feeds["audio_lengths"] = np.array([waveform.shape[1]], dtype=np.int64)
        return session.run(["logits"], feeds)[0]

    def infer_batch(self, waveforms) -> list[np.ndarray]:
        """Normalized float32 [1, samples] waveforms -> one logits [1, frames, vocabulary]
        array each, in order. A batch graph runs them zero-padded in one call, so the
        longest one sets the cost of every row; a batch-one graph runs them one by one."""
        from ..audio import pad_batch

        session = getattr(self, "session", None)
        if self.closed or session is None:
            raise RuntimeError("ONNXBackend is closed")
        waveforms = [_check_waveform(waveform) for waveform in waveforms]
        if not self.batched:
            return [self.infer(waveform) for waveform in waveforms]
        if not waveforms:
            return []
        audio, lengths = pad_batch(waveforms)
        logits, frames = session.run(["logits", "logit_lengths"],
                                     {"audio": audio, "audio_lengths": lengths})
        frames = np.asarray(frames).reshape(-1)
        if (logits.ndim != 3 or logits.shape[0] != len(waveforms)
                or frames.shape != (len(waveforms),)
                or not ((frames >= 1) & (frames <= logits.shape[1])).all()):
            raise RuntimeError(f"Batch graph returned logits {logits.shape} with frame counts "
                               f"{frames.tolist()} for {len(waveforms)} waveforms")
        return [logits[row:row + 1, :count] for row, count in enumerate(frames.tolist())]

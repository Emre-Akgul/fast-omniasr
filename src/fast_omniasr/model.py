from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio, prepare_audio
from .decoder import greedy_token_ids
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Transcription:
    text: str
    token_ids: tuple[int, ...]
    logits_shape: tuple[int, ...]


class OmniASR:
    """Load explicitly supplied ONNX/tokenizer assets; no implicit downloads."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path,
                 *, backend: str = "onnx", device: str = "cpu", threads: int = 4):
        if backend == "onnx":
            from .backends.onnx import ONNXBackend
            self.backend = ONNXBackend(model_path, device=device, threads=threads)
        elif backend == "tensorrt":
            from .backends.tensorrt import TensorRTBackend
            self.backend = TensorRTBackend(model_path)
        else:
            raise ValueError("backend must be 'onnx' or 'tensorrt'")
        self.tokenizer = Tokenizer(tokenizer_path)

    def _transcribe(self, audio: np.ndarray) -> Transcription:
        logits = self.backend.infer(audio)
        ids = greedy_token_ids(logits)
        return Transcription(self.tokenizer.decode(ids), tuple(ids), tuple(logits.shape))

    def transcribe(self, path: str | Path) -> Transcription:
        return self._transcribe(load_audio(path))

    def transcribe_numpy(self, waveform: np.ndarray, sample_rate: int = 16000) -> Transcription:
        return self._transcribe(prepare_audio(waveform, sample_rate))

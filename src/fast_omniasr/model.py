from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio, prepare_audio
from .decoder import greedy_token_ids
from .tensorrt_builder import DEFAULT_MAX_SAMPLES, DEFAULT_MIN_SAMPLES, DEFAULT_OPT_SAMPLES
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

    def close(self) -> None:
        """Release backend resources (e.g. the TensorRT CUDA stream/buffers). Safe to call
        more than once; a no-op for backends (like ONNX) that don't hold explicit GPU state."""
        close = getattr(self.backend, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @classmethod
    def from_pretrained(cls, repo_id: str, *, backend: str = "onnx", precision: str | None = None,
                         revision: str | None = None, cache_dir: str | Path | None = None,
                         engine_cache_dir: str | Path | None = None,
                         min_samples: int | None = None,
                         opt_samples: int | None = None,
                         max_samples: int | None = None,
                         device: str = "cpu", threads: int = 4):
        """Download and verify model/tokenizer assets from a Hugging Face Hub repo.

        The repo must publish `model.onnx`, `tokenizer.model` and a `config.json`
        with a `files` map of expected sha256 checksums. Any ONNX external-data
        files must also be listed in that map (see
        EmreAkgul/omniASR-CTC-300M-v2-ONNX for the reference layout).

        `backend="tensorrt"` requires an explicit `precision` ("fp32" or "fp16" — there is
        no "auto", so the FP16 accuracy caveat is never silently opted into) and builds a
        local engine on first use using the requested min/opt/max sample profile, cached under
        `engine_cache_dir` (default
        `~/.cache/fast-omniasr/tensorrt`) keyed by the ONNX content, precision, profile and
        local TensorRT/GPU identity. A later call with the same key reuses the cached engine.
        """
        if backend not in ("onnx", "tensorrt"):
            raise ValueError("backend must be 'onnx' or 'tensorrt'")
        if backend == "onnx" and precision is not None:
            raise ValueError("precision is only used with backend='tensorrt'")
        if backend == "onnx" and any(
            value is not None for value in (min_samples, opt_samples, max_samples)
        ):
            raise ValueError(
                "min_samples, opt_samples and max_samples are only used with "
                "backend='tensorrt'"
            )
        if backend == "tensorrt" and precision not in ("fp32", "fp16"):
            raise ValueError("backend='tensorrt' requires precision='fp32' or precision='fp16'")
        from .hub import download_verified

        _, verified = download_verified(
            repo_id, {"model.onnx", "tokenizer.model"}, revision=revision, cache_dir=cache_dir
        )
        tokenizer_path = verified["tokenizer.model"]
        model_path = verified["model.onnx"]
        if backend == "onnx":
            return cls(model_path, tokenizer_path, backend="onnx", device=device, threads=threads)

        if precision == "fp16":
            import warnings
            warnings.warn(
                "TensorRT FP16 may produce different transcripts from FP32. "
                "See the project's accuracy documentation.",
                stacklevel=2,
            )
        from .tensorrt_cache import get_or_build_engine
        engine_path = get_or_build_engine(
            model_path,
            precision=precision,
            min_samples=DEFAULT_MIN_SAMPLES if min_samples is None else min_samples,
            opt_samples=DEFAULT_OPT_SAMPLES if opt_samples is None else opt_samples,
            max_samples=DEFAULT_MAX_SAMPLES if max_samples is None else max_samples,
            cache_dir=engine_cache_dir,
        )
        return cls(engine_path, tokenizer_path, backend="tensorrt", device=device, threads=threads)

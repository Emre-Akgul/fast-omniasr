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
                         device: str = "cpu", threads: int = 4):
        """Download and verify model/tokenizer assets from a Hugging Face Hub repo.

        The repo must publish `model.onnx`, `tokenizer.model` and a `config.json`
        with a `files` map of expected sha256 checksums. Any ONNX external-data
        files must also be listed in that map (see
        EmreAkgul/omniASR-CTC-300M-v2-ONNX for the reference layout).

        `backend="tensorrt"` requires an explicit `precision` ("fp32" or "fp16" — there is
        no "auto", so the FP16 accuracy caveat is never silently opted into) and builds a
        local engine on first use, cached under `engine_cache_dir` (default
        `~/.cache/fast-omniasr/tensorrt`) keyed by the ONNX content, precision, profile and
        local TensorRT/GPU identity. A later call with the same key reuses the cached engine.
        """
        if backend not in ("onnx", "tensorrt"):
            raise ValueError("backend must be 'onnx' or 'tensorrt'")
        if backend == "onnx" and precision is not None:
            raise ValueError("precision is only used with backend='tensorrt'")
        if backend == "tensorrt" and precision not in ("fp32", "fp16"):
            raise ValueError("backend='tensorrt' requires precision='fp32' or precision='fp16'")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("Install fast-omniasr[hub] for huggingface_hub") from exc
        import hashlib
        import json

        def fetch(filename: str) -> Path:
            return Path(hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir))

        def fetch_and_verify(filename: str, config: dict) -> Path:
            path = fetch(filename)
            expected = config.get("files", {}).get(filename, {}).get("sha256")
            if not expected:
                raise ValueError(f"config.json is missing a files.{filename}.sha256 entry")
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(
                    f"{filename} checksum mismatch: downloaded asset does not match {repo_id}/config.json"
                )
            return path

        config = json.loads(fetch("config.json").read_text())
        files = config.get("files")
        if not isinstance(files, dict):
            raise TypeError("config.json is missing a files map")
        required = {"model.onnx", "tokenizer.model"}
        if not required.issubset(files):
            missing = ", ".join(sorted(required - files.keys()))
            raise ValueError(f"config.json is missing required file entries: {missing}")
        for filename in files:
            if not isinstance(files[filename], dict):
                raise TypeError(f"config.json has an invalid entry for {filename}")
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"config.json contains an unsafe filename: {filename}")

        # Smallest file first so corruption is caught before larger downloads when possible.
        verified = {}
        for filename in sorted(files, key=lambda name: files[name].get("size", 0)):
            verified[filename] = fetch_and_verify(filename, config)
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
        engine_path = get_or_build_engine(model_path, precision=precision, cache_dir=engine_cache_dir)
        return cls(engine_path, tokenizer_path, backend="tensorrt", device=device, threads=threads)

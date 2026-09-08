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

    @classmethod
    def from_pretrained(cls, repo_id: str, *, backend: str = "onnx", revision: str | None = None,
                         cache_dir: str | Path | None = None, device: str = "cpu", threads: int = 4):
        """Download and verify model/tokenizer assets from a Hugging Face Hub repo.

        The repo must publish `model.onnx`, `tokenizer.model` and a `config.json`
        with a `files` map of expected sha256 checksums (see
        EmreAkgul/omniASR-CTC-300M-v2-ONNX for the reference layout).
        """
        if backend != "onnx":
            raise ValueError("from_pretrained currently supports backend='onnx' only")
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
        # Smallest file first so a corrupted asset is caught before the much larger model download.
        tokenizer_path = fetch_and_verify("tokenizer.model", config)
        model_path = fetch_and_verify("model.onnx", config)
        return cls(model_path, tokenizer_path, backend=backend, device=device, threads=threads)

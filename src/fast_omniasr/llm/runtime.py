"""Greedy decoding with full-prefix or explicit-cache omniASR_LLM_300M_v2 graphs."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import load_audio, prepare_audio
from ..tokenizer import Tokenizer


@dataclass(frozen=True)
class LLMTranscription:
    text: str
    token_ids: list[int]
    stop_reason: str


class OmniASRLLM:
    """Load trusted TorchScript assets. PyTorch is an optional runtime dependency.

    The default baseline recomputes the prefix. Set cached=True to load the
    explicit-cache decoder, which processes the prefix once per transcription.
    It supports one mono 16 kHz clip, up to 30 seconds, without language conditioning.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        device: str = "cpu",
        cached: bool = False,
        backend: str = "torch",
    ):
        if backend not in {"torch", "onnx"}:
            raise ValueError("backend must be torch or onnx")
        self.backend = backend

        directory = Path(directory)
        self.config = json.loads((directory / "config.json").read_text())
        if (self.config.get("format_version"), self.config.get("model")) != (
            1,
            "omniASR_LLM_300M_v2",
        ):
            raise ValueError("Unsupported LLM artifact format or model")
        if backend == "onnx":
            from .onnx_backend import ONNXLLMBackend

            self.onnx = ONNXLLMBackend(directory, device)
            self.device = device
            self.cached = True
            self.tokenizer = Tokenizer(directory / "tokenizer.model")
            return
        import torch

        self.device = torch.device(device)
        if self.device.type != "cpu":
            raise ValueError("This experimental export supports CPU inference only")
        self.encoder = torch.jit.load(str(directory / "encoder.pt"), map_location=self.device)
        self.cached = cached
        decoder_file = "decoder_cached.pt" if cached else "decoder.pt"
        self.decoder = torch.jit.load(str(directory / decoder_file), map_location=self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.tokenizer = Tokenizer(directory / "tokenizer.model")

    def transcribe(
        self, audio: str | Path | np.ndarray, *, sample_rate: int = 16000, max_new_tokens: int = 512
    ) -> LLMTranscription:
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        waveform = (
            load_audio(audio)
            if isinstance(audio, (str, Path))
            else prepare_audio(audio, sample_rate)
        )
        if waveform.shape[1] > 30 * 16000:
            raise ValueError("This runtime supports clips of at most 30 seconds")
        if getattr(self, "backend", "torch") == "onnx":
            return self._transcribe_onnx(waveform, max_new_tokens)
        import torch

        with torch.inference_mode():
            context = self.encoder(torch.from_numpy(waveform).to(self.device))
            tokens = [self.config["bos_idx"]]
            generated = []
            # Upstream stops at absolute position max_generation_length - 4.
            budget = min(max_new_tokens, self.config["max_seq_len"] - context.shape[1] - 3)
            if budget < 1:
                raise ValueError("Audio prefix exceeds decoder context capacity")
            reason = "max_new_tokens" if budget == max_new_tokens else "context_limit"
            cache = None
            for _ in range(budget):
                if getattr(self, "cached", False):
                    ids = torch.tensor([[tokens[-1]]], dtype=torch.int64, device=self.device)
                    if cache is None:
                        logits, cache = self.decoder.prefill(context, ids)
                    else:
                        logits, cache = self.decoder.decode_step(ids, cache)
                else:
                    ids = torch.tensor([tokens], dtype=torch.int64, device=self.device)
                    logits = self.decoder(context, ids)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Decoder produced non-finite logits")
                token = int(logits.argmax(-1).item())
                if token == self.config["eos_idx"]:
                    reason = "eos"
                    break
                generated.append(token)
                tokens.append(token)
        return LLMTranscription(self.tokenizer.decode(generated), generated, reason)

    def _transcribe_onnx(self, waveform, max_new_tokens):
        context = self.onnx.encode(waveform)
        budget = min(max_new_tokens, self.config["max_seq_len"] - context.shape[1] - 3)
        if budget < 1:
            raise ValueError("Audio prefix exceeds decoder context capacity")
        reason = "max_new_tokens" if budget == max_new_tokens else "context_limit"
        token = np.array([[self.config["bos_idx"]]], dtype=np.int64)
        logits, cache = self.onnx.prefill(context, token)
        generated = []
        for index in range(budget):
            if not np.isfinite(logits).all():
                raise RuntimeError("Decoder produced non-finite logits")
            next_token = int(logits.argmax(-1).item())
            if next_token == self.config["eos_idx"]:
                reason = "eos"
                break
            generated.append(next_token)
            if index + 1 < budget:
                token = np.array([[next_token]], dtype=np.int64)
                logits, cache = self.onnx.decode_step(token, cache)
        return LLMTranscription(self.tokenizer.decode(generated), generated, reason)

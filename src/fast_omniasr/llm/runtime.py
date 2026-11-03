"""Greedy decoding with full-prefix or explicit-cache OmniASR LLM-ASR graphs."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio import load_audio, prepare_audio
from ..options import check_device, check_flag, resolve_threads
from ..precision import check_precision
from ..tokenizer import Tokenizer

# Every value LLMTranscription.stop_reason can take: "eos" means the transcript is
# complete; the other two mean generation was cut off.
STOP_REASONS = ("eos", "max_new_tokens", "context_limit")


def _language_files(config: dict) -> set[str]:
    """The language tables config.json declares, shared by every precision."""
    languages = config.get("languages")
    if not isinstance(languages, dict):
        return set()
    names = {languages.get("mapping"), languages.get("embeddings")}
    if not all(isinstance(name, str) for name in names):
        raise ValueError("config.json languages needs 'mapping' and 'embeddings' file names")
    unlisted = names - config.get("files", {}).keys()
    if unlisted:
        raise ValueError(f"config.json declares language tables it does not list: "
                         f"{', '.join(sorted(unlisted))}")
    return names


@dataclass(frozen=True)
class LLMTranscription:
    text: str
    token_ids: list[int]
    stop_reason: str  # one of STOP_REASONS


class OmniASRLLM:
    """Load trusted TorchScript assets. PyTorch is an optional runtime dependency.

    The default baseline recomputes the prefix. Set cached=True to load the
    explicit-cache decoder, which processes the prefix once per transcription.
    It supports one mono 16 kHz clip, up to 30 seconds, without language conditioning.

    Thread-safe: an instance may be shared across threads. Its sessions, lazily created
    graphs, low_memory phasing and device buffers are per-instance state, so concurrent
    `transcribe()` calls are serialized -- each request runs encode and the whole decode
    loop alone, and results never mix. For parallel throughput use one instance per
    worker (each holds its own copy of the weights). `close()` waits for an in-flight
    transcription to finish. The guarantee covers `transcribe()`, `close()` and
    `language_embedding()`; the backend object in `.onnx` is not itself thread-safe, so
    don't call its methods directly while the model is in use.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        device: str = "cpu",
        cached: bool = False,
        backend: str = "torch",
        threads: int | None = None,
        low_memory: bool = False,
        precision: str = "fp32",
    ):
        """`threads` (default 4) and `low_memory` apply to backend="onnx"; the TorchScript
        baseline uses PyTorch's process-wide thread setting and loads one copy, so it
        rejects both rather than silently ignoring them. `cached` selects the TorchScript
        baseline's decoder; the ONNX backend always decodes with an explicit KV cache."""
        self.closed = False
        self._lock = threading.Lock()
        self._language_lock = threading.Lock()
        if backend not in {"torch", "onnx"}:
            raise ValueError("backend must be torch or onnx")
        check_device(device)
        check_flag("low_memory", low_memory)
        check_flag("cached", cached)
        if backend == "onnx":
            threads = resolve_threads(threads)
        else:
            if precision != "fp32":
                raise ValueError("The TorchScript baseline supports precision='fp32' only")
            if threads is not None:
                raise ValueError("threads only applies to backend='onnx'; the TorchScript "
                                 "baseline follows torch.set_num_threads()")
            if low_memory:
                raise ValueError("low_memory only applies to backend='onnx'")
        self.backend = backend

        directory = Path(directory)
        self.directory = directory
        from .validation import (
            InvalidArtifactError,
            validate_config,
            validate_graphs,
            validate_languages,
            validate_tokenizer,
        )

        try:
            self.config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        except ValueError as exc:
            raise InvalidArtifactError(
                f"Invalid LLM artifact {directory / 'config.json'}: not valid JSON ({exc})"
            ) from exc
        # Everything below is checked before any session or module is loaded.
        validate_config(self.config, directory)
        self.tokenizer = Tokenizer(directory / "tokenizer.model")
        validate_tokenizer(self.tokenizer, self.config, directory)
        if backend == "onnx":
            check_precision(precision, device)
            hidden = validate_graphs(directory, precision, self.config)
            validate_languages(self.config, directory, hidden)
            from .onnx_backend import ONNXLLMBackend

            self.onnx = ONNXLLMBackend(directory, device, threads, low_memory, precision)
            self.device = device
            self.cached = True
            return
        validate_languages(self.config, directory, None)
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

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        device: str = "cpu",
        threads: int | None = None,
        low_memory: bool = False,
        precision: str = "fp32",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> "OmniASRLLM":
        """Download, verify and load published ONNX assets (no PyTorch needed).

        The repo's config.json is the runtime config plus a `files` map of
        sha256 checksums covering every graph, weight file and the tokenizer
        (see EmreAkgul/omniASR-LLM-300M-v2-ONNX). Encoder external data, when
        present, is listed in the same map and lands beside encoder.onnx.

        Sessions stay loaded across transcribe() calls; on CUDA the encoder and one decoder
        copy stay resident (~10 GiB for 300M in float32). low_memory=True loads one session
        at a time instead, fitting smaller GPUs but reloading weights for every clip.

        precision="fp16" or "bf16" downloads only that variant (MatMul weights in half
        precision, about half the size) plus the shared tokenizer and language tables.
        bf16 needs device="cuda" and an Ampere or newer GPU.
        """
        check_device(device)
        threads = resolve_threads(threads)
        check_flag("low_memory", low_memory)
        from ..hub import download_verified
        from ..precision import check_precision

        check_precision(precision, device)
        if precision != "fp32":
            required = {f"{precision}/encoder.onnx", f"{precision}/decoder.onnx", "tokenizer.model"}
            _, verified = download_verified(
                repo_id, lambda config: required | _language_files(config), revision=revision,
                cache_dir=cache_dir, dependencies=True,
            )
            directory = verified["tokenizer.model"].parent
            if any(verified[name].parent != directory / precision for name in required - {"tokenizer.model"}):
                raise RuntimeError("Downloaded assets are not co-located in one snapshot")
            return cls(
                directory, backend="onnx", device=device, threads=threads,
                low_memory=low_memory, precision=precision,
            )
        required = {"encoder.onnx", "decoder.weights", "decoder_weights.json", "tokenizer.model"}

        def fp32_files(config):
            # decoder.onnx (one session) is preferred; older repos ship only prefill/step.
            listed = config.get("files", {})
            graphs = {"decoder.onnx"} if "decoder.onnx" in listed else {
                "decoder_prefill.onnx", "decoder_step.onnx"} & listed.keys()
            return required | graphs | _language_files(config)

        _, verified = download_verified(
            repo_id, fp32_files, revision=revision, cache_dir=cache_dir, dependencies=True
        )
        # decoder.onnx (one session) is preferred; older repos ship only prefill/step graphs.
        graphs = {"decoder.onnx"} & verified.keys() or {"decoder_prefill.onnx", "decoder_step.onnx"}
        if not graphs <= verified.keys():
            raise ValueError("config.json lists no decoder graph")
        directory = verified["encoder.onnx"].parent
        if any(verified[name].parent != directory for name in required | graphs):
            raise RuntimeError("Downloaded assets are not co-located in one snapshot")
        return cls(
            directory, backend="onnx", device=device, threads=threads, low_memory=low_memory
        )

    def language_embedding(self, lang: str) -> np.ndarray:
        """Return the learned embedding row for an upstream code such as eng_Latn."""
        languages = self.config.get("languages")
        if not languages:
            raise ValueError("These model assets do not include language conditioning")
        if not hasattr(self, "_lang_embeddings"):
            with self._ensure_lock("_language_lock"):
                if not hasattr(self, "_lang_embeddings"):  # loaded once, even concurrently
                    mapping = json.loads((self.directory / languages["mapping"]).read_text())
                    embeddings = np.load(self.directory / languages["embeddings"],
                                         mmap_mode="r")
                    self._languages = mapping
                    self._lang_embeddings = embeddings  # published last: readers check it
        index = self._languages.get(lang.lower())
        if index is None:
            raise ValueError(f"Unknown language code {lang!r}; expected e.g. 'eng_Latn'")
        return np.array(self._lang_embeddings[index], dtype=np.float32)

    def _ensure_lock(self, name):
        # Instances built without __init__ (tests, subclasses) get their lock lazily;
        # dict.setdefault is atomic under the GIL, so concurrent first use shares one.
        return self.__dict__.setdefault(name, threading.Lock())

    def close(self) -> None:
        """Release the model: the ONNX Runtime sessions (ONNX backend) or the TorchScript
        modules (torch baseline). Idempotent; `transcribe()` afterwards raises
        RuntimeError. Also called on leaving a `with OmniASRLLM(...) as model:` block."""
        if getattr(self, "closed", False):
            return
        with self._ensure_lock("_lock"):  # wait for an in-flight transcription
            if getattr(self, "closed", False):
                return
            self.closed = True
            onnx = getattr(self, "onnx", None)
            if onnx is not None:
                onnx.close()
            self.encoder = self.decoder = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        *,
        sample_rate: int = 16000,
        max_new_tokens: int = 512,
        lang: str | None = None,
    ) -> LLMTranscription:
        """Greedy-decode one clip. `lang` conditions on an upstream language code
        (e.g. "eng_Latn"); without it the model's learned no-language row is used."""
        if getattr(self, "closed", False):
            raise RuntimeError("This OmniASRLLM model is closed")
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
        if lang is None:
            import warnings

            warnings.warn(
                "Transcribing without a `lang` code can degrade quality, as upstream warns; "
                "pass e.g. lang='eng_Latn'",
                stacklevel=2,
            )
        language = None if lang is None else self.language_embedding(lang)
        with self._ensure_lock("_lock"):  # one request at a time per instance
            if getattr(self, "closed", False):
                raise RuntimeError("This OmniASRLLM model is closed")
            if getattr(self, "backend", "torch") == "onnx":
                return self._transcribe_onnx(waveform, max_new_tokens, language)
            return self._transcribe_torch(waveform, max_new_tokens, language)

    def _transcribe_torch(self, waveform, max_new_tokens, language=None):
        import torch

        with torch.inference_mode():
            context = self.encoder(torch.from_numpy(waveform).to(self.device))
            if language is not None:
                # The encoder output ends with [LID marker, language]; replace the language.
                row = torch.from_numpy(language).to(context)
                context = torch.cat([context[:, :-1], row[None, None]], dim=1)
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

    def _transcribe_onnx(self, waveform, max_new_tokens, language=None):
        context = self.onnx.encode(waveform)
        if language is not None:
            context = np.array(context, copy=True)
            context[:, -1] = language
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

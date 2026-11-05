from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_audio, prepare_audio
from .decoder import greedy_token_ids
from .options import check_device, resolve_threads
from .tensorrt_builder import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_OPT_SAMPLES,
    TensorRTBuildOptions,
    check_build_options,
)
from .tokenizer import Tokenizer, check_compatible


@dataclass(frozen=True)
class Transcription:
    text: str
    token_ids: tuple[int, ...]
    logits_shape: tuple[int, ...]


def _check_backend_options(backend: str, device: str | None, threads: int | None):
    """Resolve `device`/`threads` for a backend; reject options it would ignore.

    ONNX: device "cpu" (default) or "cuda", threads >= 1 (default 4). TensorRT runs on the
    current CUDA device (select it with CUDA_VISIBLE_DEVICES or cudaSetDevice) and has no
    thread setting, so device must be omitted or "cuda" and threads must be omitted."""
    if backend == "onnx":
        return (check_device("cpu" if device is None else device),
                resolve_threads(threads))
    if backend == "tensorrt":
        if device not in (None, "cuda"):
            raise ValueError(
                f"backend='tensorrt' runs on the current CUDA device; got device={device!r}. "
                "Omit device (or pass 'cuda') and select the GPU with CUDA_VISIBLE_DEVICES"
            )
        if threads is not None:
            raise ValueError("threads only applies to backend='onnx'; TensorRT has no "
                             "CPU thread setting")
        return "cuda", None
    raise ValueError("backend must be 'onnx' or 'tensorrt'")


class OmniASR:
    """Load explicitly supplied ONNX/tokenizer assets; no implicit downloads."""

    snapshot = None  # set by from_pretrained: where the assets came from (hub.Snapshot)

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path,
                 *, backend: str = "onnx", device: str | None = None,
                 threads: int | None = None, precision: str = "fp32"):
        """`device` ("cpu" by default, or "cuda") and `threads` (default 4) apply to the
        ONNX backend; TensorRT runs on the current CUDA device and rejects them.
        `precision` describes an ONNX graph at `model_path` (an fp16/bf16 variant from
        tools/convert_precision.py); TensorRT engines carry their precision themselves, so
        only the default is accepted there."""
        device, threads = _check_backend_options(backend, device, threads)
        if backend != "onnx" and precision != "fp32":
            raise ValueError("precision describes ONNX graphs; a TensorRT engine carries "
                             "its own precision (choose it when building the engine)")
        # The tokenizer and the graph's recorded tokenizer identity are checked first, so a
        # mismatched pair fails before a large graph or engine is loaded.
        self.tokenizer = Tokenizer(tokenizer_path)
        if backend == "onnx":
            check_compatible(self.tokenizer, graph_path=model_path)
        if backend == "onnx":
            from .backends.onnx import ONNXBackend
            self.backend = ONNXBackend(
                model_path, device=device, threads=threads, precision=precision
            )
        else:
            from .backends.tensorrt import TensorRTBackend
            self.backend = TensorRTBackend(model_path)
        try:
            check_compatible(
                self.tokenizer, vocabulary_size=getattr(self.backend, "vocabulary_size", None)
            )
        except BaseException:
            self.backend.close()  # don't leave a loaded session/engine behind
            raise

    @property
    def closed(self) -> bool:
        return getattr(self.backend, "closed", False)

    def _transcribe(self, audio: np.ndarray) -> Transcription:
        if self.closed:
            raise RuntimeError("This OmniASR model is closed")
        logits = self.backend.infer(audio)
        ids = greedy_token_ids(logits)
        return Transcription(self.tokenizer.decode(ids), tuple(ids), tuple(logits.shape))

    def transcribe(self, path: str | Path, *, convert_audio: bool = False) -> Transcription:
        """Transcribe a mono 16 kHz file. `convert_audio=True` downmixes and resamples other
        formats first (the `resample` extra); by default they are rejected."""
        return self._transcribe(load_audio(path, convert_audio=convert_audio))

    def transcribe_numpy(self, waveform: np.ndarray, sample_rate: int = 16000,
                         *, convert_audio: bool = False) -> Transcription:
        """Transcribe a mono 16 kHz waveform. `convert_audio=True` also takes other rates and
        (samples, channels) arrays (the `resample` extra); by default they are rejected."""
        return self._transcribe(prepare_audio(waveform, sample_rate,
                                              convert_audio=convert_audio))

    def close(self) -> None:
        """Release backend resources: the ONNX Runtime session, or the TensorRT engine,
        context, stream and buffers. Idempotent; `transcribe*()` afterwards raises
        RuntimeError. Also called on leaving a `with OmniASR(...) as model:` block."""
        self.backend.close()

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
                         build_options: TensorRTBuildOptions | None = None,
                         device: str | None = None, threads: int | None = None,
                         local_files_only: bool = False):
        """Download and verify model/tokenizer assets from a Hugging Face Hub repo.

        `local_files_only=True` loads the cached copy without contacting the Hub (raising
        fast_omniasr.hub.LocalBundleNotFoundError if there is none). The returned model's
        `snapshot` records the repo, revision, resolved commit and local directory.

        The repo must publish `model.onnx`, `tokenizer.model` and a `config.json`
        with a `files` map of expected sha256 checksums. Any ONNX external-data
        files must also be listed in that map (see
        EmreAkgul/omniASR-CTC-300M-v2-ONNX for the reference layout).

        `backend="tensorrt"` requires an explicit `precision`: "fp32", "fp16" (fastest; a
        few transcripts differ slightly from FP32) or "mixed" (FP16 with the
        precision-sensitive convolutional front end kept in FP32: closest to FP32 at about
        twice the FP16 latency). There is no "auto", so a precision trade-off is never
        silently opted into and builds a
        local engine on first use using the requested min/opt/max sample profile, cached under
        `engine_cache_dir` (default
        `~/.cache/fast-omniasr/tensorrt`) keyed by the ONNX content, precision, profile and
        local TensorRT/GPU identity. A later call with the same key reuses the cached engine.
        `build_options` (TensorRTBuildOptions) tunes the engine build: workspace size,
        optimization level and a reusable timing cache. Non-default options are part of the key.

        `backend="onnx"` accepts precision "fp32" (default), "fp16" or "bf16": the half
        variants keep MatMul weights in half precision. Only the chosen graph, the
        external-data files it references and the tokenizer are downloaded. bf16 needs
        device="cuda" and an Ampere+ GPU.
        """
        if backend not in ("onnx", "tensorrt"):
            raise ValueError("backend must be 'onnx' or 'tensorrt'")
        # Reject inapplicable device/threads before any download or engine build.
        resolved_device, _ = _check_backend_options(backend, device, threads)
        from .precision import check_precision
        if backend == "onnx":
            precision = check_precision(precision or "fp32", resolved_device)
        if backend == "onnx" and any(
            value is not None for value in (min_samples, opt_samples, max_samples)
        ):
            raise ValueError(
                "min_samples, opt_samples and max_samples are only used with "
                "backend='tensorrt'"
            )
        if backend == "onnx" and build_options is not None:
            raise ValueError("build_options is only used with backend='tensorrt'")
        check_build_options(build_options)
        if backend == "tensorrt" and precision not in ("fp32", "fp16", "mixed"):
            raise ValueError(
                "backend='tensorrt' requires precision='fp32', 'fp16' or 'mixed'"
            )
        from .hub import download_verified
        from .options import check_flag

        check_flag("local_files_only", local_files_only)

        # TensorRT builds its engine from the float32 graph whatever the engine precision.
        graph_precision = precision if backend == "onnx" else "fp32"
        graph = "model.onnx" if graph_precision == "fp32" else f"{graph_precision}/model.onnx"
        _, verified, snapshot = download_verified(
            repo_id, {graph, "tokenizer.model"}, revision=revision, cache_dir=cache_dir,
            dependencies=True,  # the graph, its external data wherever it lives, the tokenizer
            local_files_only=local_files_only, return_snapshot=True,
        )
        tokenizer_path = verified["tokenizer.model"]
        model_path = verified[graph]
        if backend == "tensorrt":
            # The engine has no metadata of its own: check the graph it is built from,
            # before a long engine build.
            check_compatible(Tokenizer(tokenizer_path), graph_path=model_path)
        if backend == "onnx":
            return _attach(snapshot, cls(model_path, tokenizer_path, backend="onnx",
                                         device=device, threads=threads, precision=precision))

        if precision == "fp16":
            import warnings
            warnings.warn(
                "TensorRT FP16 can change a few transcripts slightly compared with FP32; "
                "precision='mixed' keeps the sensitive layers in FP32. See the project's "
                "accuracy documentation.",
                stacklevel=2,
            )
        from .backends.tensorrt import EngineDeserializationError
        from .tensorrt_cache import get_or_build_engine, recorded_engine_sha256

        def engine(**rebuild):
            return get_or_build_engine(
                model_path,
                precision=precision,
                min_samples=DEFAULT_MIN_SAMPLES if min_samples is None else min_samples,
                opt_samples=DEFAULT_OPT_SAMPLES if opt_samples is None else opt_samples,
                max_samples=DEFAULT_MAX_SAMPLES if max_samples is None else max_samples,
                options=build_options,
                cache_dir=engine_cache_dir,
                **rebuild,
            )

        engine_path = engine()
        try:
            return _attach(snapshot, cls(engine_path, tokenizer_path, backend="tensorrt",
                                         device=device, threads=threads))
        except EngineDeserializationError as exc:
            reason = str(exc)  # rebuild outside the handler so the failed backend is freed
        import warnings
        warnings.warn(f"Cached TensorRT engine {engine_path} could not be deserialized "
                      f"({reason}); rebuilding it once. If the GPU was low on memory this "
                      "may be transient.", stacklevel=2)
        failed = recorded_engine_sha256(engine_path)
        rebuilt = engine(discard_sha256=failed) if failed else engine(rebuild=True)
        return _attach(snapshot, cls(rebuilt, tokenizer_path, backend="tensorrt", device=device,
                                     threads=threads))


def _attach(snapshot, model):
    model.snapshot = snapshot
    return model

"""The `fast-omniasr` command. Heavy dependencies load only in the commands that need them."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from dataclasses import asdict

# Failures a user can act on, reported as a message rather than a traceback.
EXPECTED_ERRORS = (ValueError, TypeError, RuntimeError, OSError, ImportError)
PACKAGES = ("numpy", "soundfile", "sentencepiece", "onnxruntime", "onnxruntime-gpu", "torch",
            "huggingface-hub", "soxr", "cuda-python")
CTC_ONLY = {"backend": "--backend", "word_timestamps": "--word-timestamps"}
LLM_ONLY = {"lang": "--lang", "max_new_tokens": "--max-new-tokens",
            "repetition_penalty": "--repetition-penalty", "low_memory": "--low-memory"}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str), flush=True)


def _given(args, option: str) -> bool:
    value = getattr(args, option, None)
    return value is not None and value is not False


def _check_kind(parser, args) -> None:
    """Refuse options the model kind would ignore."""
    options = CTC_ONLY if args.kind == "llm" else LLM_ONLY
    for option, flag in options.items():
        if _given(args, option):
            parser.error(f"{flag} doesn't apply to --kind {args.kind}")


def _load(args):
    options = {"revision": args.revision, "cache_dir": args.cache_dir,
               "local_files_only": args.local_files_only}
    for option in ("device", "threads", "precision"):
        if getattr(args, option) is not None:
            options[option] = getattr(args, option)
    if args.kind == "llm":
        from .llm import OmniASRLLM

        return OmniASRLLM.from_pretrained(args.model, low_memory=args.low_memory, **options)
    from .model import OmniASR

    return OmniASR.from_pretrained(args.model, backend=args.backend or "onnx", **options)


def _request(args) -> tuple[bool, dict]:
    """Whether to use transcribe_long, and its keyword arguments."""
    request = {"convert_audio": args.convert_audio}
    if args.kind == "ctc":
        request["word_timestamps"] = args.word_timestamps
    else:
        for option in ("lang", "max_new_tokens", "repetition_penalty"):
            if getattr(args, option) is not None:
                request[option] = getattr(args, option)
    long = args.long or args.vad or args.chunk_length is not None
    if long:
        request["vad_filter"] = args.vad
        if args.chunk_length is not None:
            request["chunk_length"] = args.chunk_length
    return long, request


def _print_text(path: str, result, many: bool) -> None:
    print(f"{path}: {result.text}" if many else result.text, flush=True)
    for word in getattr(result, "words", ()):
        print(f"{word.start:.2f}\t{word.end:.2f}\t{word.word}", flush=True)


def _transcribe(args) -> int:
    failed = False
    long, request = _request(args)
    model = _load(args)
    with model:
        for path in args.audio:
            try:
                result = (model.transcribe_long if long else model.transcribe)(path, **request)
            except EXPECTED_ERRORS as exc:
                failed = True
                if args.json:
                    _write_json({"path": path, "error": str(exc)})
                print(f"error: {path}: {exc}", file=sys.stderr, flush=True)
                continue
            if args.json:
                _write_json({"path": path, **asdict(result)})
            else:
                _print_text(path, result, len(args.audio) > 1)
    return 1 if failed else 0


def _inspect(args) -> int:
    model = _load(args)
    with model:
        info = {"kind": args.kind, **asdict(model.snapshot)}
        if args.kind == "ctc":
            session = getattr(model.backend, "session", None)
            info["backend"] = args.backend or "onnx"
            info["vocabulary_size"] = model.tokenizer.vocabulary_size
            info["providers"] = session.get_providers() if session is not None else []
        else:
            info["device"] = str(model.device)
            info["precision"] = args.precision or "fp32"
    if args.json:
        _write_json(info)
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
    return 0


def _languages(args) -> int:
    from .llm.runtime import list_languages

    codes = list_languages(args.model, revision=args.revision, cache_dir=args.cache_dir,
                           local_files_only=args.local_files_only)
    if args.json:
        _write_json(codes)
    else:
        print("\n".join(codes))
    return 0


def _onnxruntime_providers() -> list[str] | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None
    try:
        import onnxruntime
    except Exception:  # noqa: BLE001 -- a broken install is what this command reports
        return None
    return onnxruntime.get_available_providers()


def _tensorrt_available() -> bool:
    if importlib.util.find_spec("tensorrt") is None:
        return False
    try:
        import tensorrt  # noqa: F401
    except Exception:  # noqa: BLE001 -- it needs NVIDIA libraries that may be missing
        return False
    return True


def _diagnose(args) -> int:
    from .hub import _cache_root, list_cached

    bundles = list_cached(args.cache_dir)
    info = {
        "fast_omniasr": _version("fast-omniasr"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: _version(package) for package in PACKAGES},
        "onnxruntime_providers": _onnxruntime_providers(),
        "tensorrt_available": _tensorrt_available(),
        "cache_dir": str(_cache_root(args.cache_dir)),
        "cached_models": [asdict(bundle) for bundle in bundles],
    }
    if args.json:
        _write_json(info)
        return 0
    for key in ("fast_omniasr", "python", "platform", "onnxruntime_providers",
                "tensorrt_available", "cache_dir"):
        print(f"{key}: {info[key]}")
    for package, version in info["packages"].items():
        print(f"{package}: {version or 'not installed'}")
    for bundle in bundles:
        print(f"cached: {bundle.repo_id}@{bundle.revision} {bundle.commit} "
              f"{bundle.size_bytes} bytes")
    return 0


def _source_options(parser) -> None:
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--revision", help="branch, tag or commit (default: main)")
    parser.add_argument("--cache-dir", help="model cache (default: ~/.cache/fast-omniasr/hub)")
    parser.add_argument("--local-files-only", action="store_true",
                        help="use the cached copy and never contact the Hub")
    parser.add_argument("--json", action="store_true", help="write JSON to stdout")


def _load_options(parser) -> None:
    _source_options(parser)
    parser.add_argument("--kind", choices=("ctc", "llm"), default="ctc", help="default: ctc")
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), help="CTC only; default: onnx")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", help="fp32, fp16, bf16 or, on TensorRT, mixed")
    parser.add_argument("--threads", type=int, help="CPU threads")
    parser.add_argument("--low-memory", action="store_true",
                        help="LLM only: load one session at a time")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fast-omniasr", description="OmniASR inference")
    parser.add_argument("--version", action="version",
                        version=f"fast-omniasr {_version('fast-omniasr')}")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser("transcribe", help="transcribe audio files")
    transcribe.add_argument("audio", nargs="+", help="audio files")
    _load_options(transcribe)
    transcribe.add_argument("--convert-audio", action="store_true",
                            help="downmix and resample other formats (the resample extra)")
    transcribe.add_argument("--long", action="store_true",
                            help="cut long recordings into chunks (transcribe_long)")
    transcribe.add_argument("--chunk-length", type=float, help="seconds; implies --long")
    transcribe.add_argument("--vad", action="store_true",
                            help="cut at silence with Silero VAD; implies --long")
    transcribe.add_argument("--word-timestamps", action="store_true", help="CTC only")
    transcribe.add_argument("--lang", help="LLM only: language code such as eng_Latn")
    transcribe.add_argument("--max-new-tokens", type=int, help="LLM only")
    transcribe.add_argument("--repetition-penalty", type=float, help="LLM only")
    transcribe.set_defaults(handler=_transcribe)

    inspect = commands.add_parser("inspect", help="load a model and report what was loaded")
    _load_options(inspect)
    inspect.set_defaults(handler=_inspect)

    languages = commands.add_parser("languages", help="list an LLM model's language codes")
    _source_options(languages)
    languages.set_defaults(handler=_languages)

    diagnose = commands.add_parser("diagnose", help="report versions, providers and the cache")
    diagnose.add_argument("--cache-dir", help="model cache (default: ~/.cache/fast-omniasr/hub)")
    diagnose.add_argument("--json", action="store_true", help="write JSON to stdout")
    diagnose.set_defaults(handler=_diagnose)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "kind"):
        _check_kind(parser, args)
    try:
        code = args.handler(args)
        sys.stdout.flush()
        return code
    except EXPECTED_ERRORS as exc:
        if isinstance(exc, BrokenPipeError):
            # The reader left, as in `| head`. Stop Python failing again on exit.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return 0
        print(f"error: {exc}", file=sys.stderr)
        return 1

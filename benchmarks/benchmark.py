"""Warm end-to-end transcription benchmark with a reproducible report.

Measures the public API (file loading, normalization, inference and decoding) on one clip
and records what is needed to reproduce the number: the model's identity (Hub repo,
revision and commit, or local file sha256s), backend, precision, device, threads, clip
duration, software versions and the platform. Reports load time, first-call time,
latency percentiles over the warm runs, the real-time factor (latency / clip duration),
and peak memory (process RSS, and GPU memory per nvidia-smi when available).

  python benchmarks/benchmark.py speech.wav --repo EmreAkgul/omniASR-CTC-300M-v2-ONNX \\
      --precision fp16 --device cuda --runs 20 --output report.json
  python benchmarks/benchmark.py speech.wav --model model.onnx --tokenizer tokenizer.model
  python benchmarks/benchmark.py speech.wav --model fp16.engine --tokenizer tokenizer.model \\
      --backend tensorrt
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPORT_VERSION = 1
PACKAGES = ("fast-omniasr", "numpy", "onnxruntime", "onnxruntime-gpu", "tensorrt",
            "cuda-python", "sentencepiece", "soundfile", "huggingface-hub")


def percentile(values, q):
    """Linear-interpolation percentile (numpy's default method), q in [0, 100]."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("no samples")
    position = (len(ordered) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency_summary(samples, duration):
    summary = {
        "runs": len(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
        **{f"p{q}": percentile(samples, q) for q in (50, 90, 95, 99)},
    }
    # Real-time factor: seconds of compute per second of audio (lower is faster).
    summary["rtf_p50"] = summary["p50"] / duration
    summary["rtf_mean"] = summary["mean"] / duration
    return summary


def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(path):
    """sha256 of a model and every file it needs: an ONNX graph's external-data files
    (e.g. fp16/model.onnx.data) are part of the model's identity."""
    path = Path(path)
    files = {path.name: sha256(path)}
    if path.suffix == ".onnx":
        from fast_omniasr._onnx_deps import graph_dependencies

        for name in sorted(graph_dependencies(path.name, path)):
            files[name] = sha256(path.parent / name)
    return files


def engine_precision(engine):
    """The precision recorded beside a cached TensorRT engine (metadata.json), if any."""
    try:
        metadata = json.loads((Path(engine).parent / "metadata.json").read_text())
    except (OSError, ValueError):
        return None
    key = metadata.get("key", metadata)
    return key.get("precision") if isinstance(key, dict) else None


def environment(device):
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "packages": versions()}
    if device == "cuda":
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"], capture_output=True, text=True, check=True,
                timeout=10).stdout
            info["gpus"] = [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError):
            info["gpus"] = None
    return info


def peak_rss_mib():
    """This process's peak resident memory (MiB), or None where unsupported."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _gpu_sampler():
    """tools/evaluate_precisions.GpuMemorySampler (nvidia-smi polling), loaded by path."""
    path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_precisions.py"
    spec = importlib.util.spec_from_file_location("_evaluate_precisions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GpuMemorySampler


def load_model(args):
    from fast_omniasr import OmniASR

    options = {}
    if args.device:
        options["device"] = args.device
    if args.threads is not None:
        options["threads"] = args.threads
    if args.repo:
        model = OmniASR.from_pretrained(args.repo, backend=args.backend,
                                        precision=args.precision, revision=args.revision,
                                        **options)
        snapshot = getattr(model, "snapshot", None)
        identity = {"repo": args.repo,
                    "revision": snapshot.revision if snapshot else (args.revision or "main"),
                    "commit": snapshot.commit if snapshot else None}
    else:
        if args.precision:
            options["precision"] = args.precision
        model = OmniASR(args.model, args.tokenizer, backend=args.backend, **options)
        identity = None  # hashed by the caller, outside the timed load
    return model, identity


def local_identity(args):
    return {"model_path": str(args.model), "files_sha256": model_files(args.model),
            "tokenizer_path": str(args.tokenizer), "tokenizer_sha256": sha256(args.tokenizer)}


def run(args):
    import soundfile as sf

    info = sf.info(str(args.audio))
    duration = info.frames / info.samplerate
    device = args.device or ("cuda" if args.backend == "tensorrt" else "cpu")
    sampler = _gpu_sampler()(interval=0.25) if device == "cuda" else None
    if sampler is not None:
        sampler.__enter__()
    try:
        start = time.perf_counter()
        model, identity = load_model(args)
        load = time.perf_counter() - start
        if identity is None:
            identity = local_identity(args)
        try:
            start = time.perf_counter()
            result = model.transcribe(args.audio)
            first = time.perf_counter() - start
            for _ in range(args.warmup):
                model.transcribe(args.audio)
            samples = []
            for _ in range(args.runs):
                start = time.perf_counter()
                model.transcribe(args.audio)
                samples.append(time.perf_counter() - start)
        finally:
            model.close()
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)
    if args.backend == "tensorrt":
        threads = None  # TensorRT doesn't use an intra-op thread pool
    else:
        from fast_omniasr.options import resolve_threads

        threads = resolve_threads(args.threads)
    return {
        "report_version": REPORT_VERSION,
        "scope": "audio file loading + normalization + inference + decoding",
        "model": identity,
        "backend": args.backend,
        "precision": args.precision or ("fp32" if args.backend == "onnx" or args.repo
                                        else engine_precision(args.model)),
        "device": device,
        "threads": threads,
        "audio": {"path": str(args.audio), "sha256": sha256(args.audio),
                  "seconds": duration, "sample_rate": info.samplerate,
                  "channels": info.channels},
        "load_seconds": load,
        "first_seconds": first,
        "warmup_runs": args.warmup,
        "latency_seconds": latency_summary(samples, duration),
        "memory": {"peak_rss_mib": peak_rss_mib(),
                   "peak_gpu_mib": sampler.peak if sampler is not None else None},
        "environment": environment(device),
        "transcript": result.text,
        "samples": samples,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("audio", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hugging Face Hub repo id (uses from_pretrained)")
    source.add_argument("--model", type=Path, help="local ONNX graph or TensorRT engine")
    parser.add_argument("--revision", help="Hub revision (with --repo; default main)")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer.model (with --model)")
    parser.add_argument("--backend", choices=["onnx", "tensorrt"], default="onnx")
    parser.add_argument("--precision",
                        help="onnx: fp32 (default), fp16, bf16; tensorrt with --repo: "
                             "fp32, fp16, mixed (a local engine carries its own)")
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        help="ONNX only (default cpu); TensorRT uses the current CUDA device")
    parser.add_argument("--threads", type=int, help="ONNX intra-op threads (default 4)")
    parser.add_argument("--runs", type=int, default=10, help="measured warm runs")
    parser.add_argument("--warmup", type=int, default=2,
                        help="unmeasured runs after the first call (default 2)")
    parser.add_argument("--output", type=Path, help="also write the report here")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be zero or more")
    if args.model and not args.tokenizer:
        parser.error("--model needs --tokenizer")
    if args.repo and args.tokenizer:
        parser.error("--tokenizer applies to --model; a repo provides its own")
    if args.revision and not args.repo:
        parser.error("--revision applies to --repo")
    if args.backend == "tensorrt" and args.repo and not args.precision:
        parser.error("--backend tensorrt with --repo needs --precision (fp32, fp16 or mixed)")
    if args.backend == "tensorrt" and args.model and args.precision:
        parser.error("--precision: a local TensorRT engine carries its own precision")
    allowed = {"onnx": ("fp32", "fp16", "bf16"), "tensorrt": ("fp32", "fp16", "mixed")}
    if args.precision and args.precision not in allowed[args.backend]:
        parser.error(f"--precision for {args.backend}: one of {', '.join(allowed[args.backend])}")
    if args.precision == "bf16" and args.device != "cuda":
        parser.error("--precision bf16 needs --device cuda")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    if args.backend == "tensorrt" and (args.threads is not None or args.device == "cpu"):
        parser.error("--backend tensorrt runs on the current CUDA device; drop --threads/--device")
    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

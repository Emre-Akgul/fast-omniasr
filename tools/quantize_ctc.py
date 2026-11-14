"""Experimental: build INT8 / weight-only variants of a CTC ONNX export and score them.

Nothing here is published or loaded by the package: the variants exist to decide, with
measurements, whether a quantized CTC model is worth shipping. The evaluation set is the
pinned FLEURS sample (benchmarks/fleurs/manifest.json); the quality and efficiency gates
live in benchmarks/quantization/gates.json and are explained in
benchmarks/quantization/README.md.

  # 1. calibration audio for static INT8: FLEURS *dev* utterances (disjoint from the test
  #    sample), a fixed hash-ordered selection per language, recorded with sha256s
  python tools/quantize_ctc.py calibration --out calib/

  # 2. quantized graphs (and, on a CUDA machine with TensorRT 10, INT8 engines)
  python tools/quantize_ctc.py quantize --ctc /path/to/ctc-export --out variants/ \\
      --calibration calib/calibration.json
  python tools/quantize_ctc.py quantize --ctc /path/to/ctc-export --out variants/ \\
      --variants trt-fp16 trt-mixed trt-int8 trt-int8-mixed

  # 3. per-language WER/CER, latency and memory, each variant in a fresh process
  python tools/quantize_ctc.py evaluate --ctc /path/to/ctc-export --variants-dir variants/ \\
      --device cpu --gate --report cpu.json

Variants (ONNX ones run on CPU or, with --device cuda, on ONNX Runtime's CUDA provider; the
report then lists, per variant, the operators that ran on the CPU instead, so quantized
operators without a CUDA kernel are visible rather than silently slow):

  fp32              the export itself (the reference)
  fp16              the export's published fp16/ graph (a shipped baseline)
  int8-dynamic      MatMul weights int8 per channel, activations quantized at run time
                    (ONNX Runtime dynamic quantization)
  int8-static       MatMul weights and inputs int8 in QDQ form, activation ranges calibrated
                    offline (symmetric, so TensorRT can consume it too)
  int8-weight-only  MatMul weights int8 in blocks of 32 (MatMulNBits), activations float32
  int4-weight-only  as above with 4-bit weights
  trt-fp16, trt-mixed            TensorRT engines built with the shipped recipe (baselines)
  trt-int8, trt-int8-mixed       TensorRT engines from the int8-static graph (explicit
                                 quantization) with FP16 elsewhere; -mixed also keeps the
                                 precision-sensitive front end in FP32

Only linear layers (MatMul with a weight) are quantized. Attention's activation-by-activation
MatMuls stay float32, and so do the convolutions (feature extractor and positional
convolution): they are the precision-sensitive layers of this model (see
benchmarks/tensorrt_fp16_sensitivity.json) and hold a small share of the weights.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "benchmarks" / "quantization" / "gates.json"

_spec = importlib.util.spec_from_file_location("fleurs_eval",
                                               Path(__file__).with_name("fleurs_eval.py"))
fleurs_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fleurs_eval)

ONNX_VARIANTS = ("fp32", "fp16", "int8-dynamic", "int8-static", "int8-weight-only",
                 "int4-weight-only")
ENGINE_VARIANTS = ("trt-fp16", "trt-mixed", "trt-int8", "trt-int8-mixed")
VARIANTS = ONNX_VARIANTS + ENGINE_VARIANTS
QUANTIZED_OPS = ["MatMul"]  # never Conv: the sensitive front end stays float32


def variant_path(variant: str, ctc: Path, variants_dir: Path) -> Path:
    """Where a variant's graph or engine lives."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    if variant == "fp32":
        return ctc / "model.onnx"
    if variant == "fp16":
        return ctc / "fp16" / "model.onnx"
    if variant in ENGINE_VARIANTS:
        return variants_dir / variant / "model.engine"
    return variants_dir / variant / "model.onnx"


def check_device_variants(device: str, variants: list[str]) -> None:
    if len(set(variants)) != len(variants):
        raise ValueError("a variant is listed twice; its results would be overwritten")
    if device == "cpu":
        engines = [v for v in variants if v in ENGINE_VARIANTS]
        if engines:
            raise ValueError(f"{engines} are TensorRT engines; evaluate them with --device cuda")


# --- calibration audio ----------------------------------------------------------------

def calibration(out: Path, per_language: int, revision: str, languages: list[str],
                cache_dir: Path | None) -> Path:
    """Extract `per_language` FLEURS dev utterances per language (same selection rule as
    the test sample) into `out` and write calibration.json listing them with sha256s.
    Dev and test hold different sentences, so calibration never sees evaluation audio."""
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for fleurs in languages:
        tsv = hf_hub_download(fleurs_eval.DATASET, f"data/{fleurs}/dev.tsv", repo_type="dataset",
                              revision=revision, cache_dir=cache_dir)
        rows = fleurs_eval.select_utterances(fleurs_eval.read_tsv(Path(tsv)), per_language,
                                             fleurs_eval.MAX_SECONDS * 16000)
        wanted = {f"dev/{r['file_name']}": r for r in rows}
        archive = hf_hub_download(fleurs_eval.DATASET, f"data/{fleurs}/audio/dev.tar.gz",
                                  repo_type="dataset", revision=revision, cache_dir=cache_dir)
        with tarfile.open(archive) as tar:
            for member in tar:
                row = wanted.pop(member.name, None)
                if row is None:
                    continue
                data = tar.extractfile(member).read()
                name = f"{fleurs}_{row['file_name']}"
                (out / name).write_bytes(data)
                entries.append({"file": name, "fleurs_language": fleurs,
                                "sha256": hashlib.sha256(data).hexdigest()})
        if wanted:
            raise RuntimeError(f"{archive} is missing {sorted(wanted)}")
        print(f"calibration: {fleurs} done", flush=True)
    manifest = {"dataset": fleurs_eval.DATASET, "revision": revision, "split": "dev",
                "license": "CC-BY-4.0", "clips": entries}
    path = out / "calibration.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def read_calibration(path: Path) -> list[Path]:
    """Calibration clip paths from calibration.json, each checked against its sha256."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("split") == "test":
        raise ValueError("calibration audio must not come from the evaluated test split")
    clips = []
    for entry in manifest.get("clips", []):
        clip = path.parent / Path(entry["file"]).name
        if not fleurs_eval._verified(clip, entry["sha256"]):
            raise ValueError(f"{clip} is missing or does not match calibration.json")
        clips.append(clip)
    if not clips:
        raise ValueError(f"{path} lists no calibration clips")
    return clips


class _AudioReader:
    """ONNX Runtime CalibrationDataReader over normalized waveforms."""

    def __init__(self, clips):
        self._clips = iter(clips)

    def get_next(self):
        import soundfile as sf

        from fast_omniasr.audio import prepare_audio

        clip = next(self._clips, None)
        if clip is None:
            return None
        waveform, rate = sf.read(clip, dtype="float32")
        return {"audio": prepare_audio(waveform, rate)}


# --- building variants ----------------------------------------------------------------

def activation_matmuls(path: Path) -> list[str]:
    """Names of the MatMul nodes with no constant operand: attention scores and context.
    Only linear layers (a MatMul with a weight initializer) are quantized; leaving these
    out matches the weight-only variants and keeps static calibration from recording the
    [heads, frames, frames] attention tensors of every layer."""
    import onnx

    graph = onnx.load(path).graph
    weights = {tensor.name for tensor in graph.initializer}
    names = [node.name for node in graph.node
             if node.op_type == "MatMul" and not weights.intersection(node.input)]
    if not all(names):
        raise ValueError(f"{path} has unnamed MatMul nodes; they can't be excluded by name")
    return names


def quantize(variant: str, ctc: Path, out: Path, calibration_clips: list[Path] | None):
    """Write one variant under `out/<variant>/`. Returns its path."""
    target = variant_path(variant, ctc, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    source = ctc / "model.onnx"
    if variant == "int8-dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(source, target, op_types_to_quantize=QUANTIZED_OPS, per_channel=True,
                         weight_type=QuantType.QInt8,
                         nodes_to_exclude=activation_matmuls(source))
    elif variant == "int8-static":
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        if not calibration_clips:
            raise ValueError("int8-static needs --calibration")
        quantize_static(source, target, _AudioReader(calibration_clips),
                        quant_format=QuantFormat.QDQ, op_types_to_quantize=QUANTIZED_OPS,
                        nodes_to_exclude=activation_matmuls(source),
                        per_channel=True, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    elif variant in ("int8-weight-only", "int4-weight-only"):
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        bits = 8 if variant.startswith("int8") else 4
        config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=bits,
                                              op_types_to_quantize=tuple(QUANTIZED_OPS))
        quantizer = MatMulNBitsQuantizer(onnx.load(source), algo_config=config)
        quantizer.process()
        onnx.save_model(quantizer.model.model, target)
    elif variant in ENGINE_VARIANTS:
        build_engine_variant(variant, ctc, out)
    else:
        raise ValueError(f"{variant} is not built by this tool")
    return target


def build_engine_variant(variant: str, ctc: Path, out: Path) -> Path:
    """trt-fp16/trt-mixed: the package's own build recipe on the float32 graph.
    trt-int8(-mixed): the int8-static QDQ graph with the INT8 and FP16 builder flags, the
    same profile and workspace as the shipped recipe; -mixed pins the front end to FP32."""
    from fast_omniasr import tensorrt_builder as builder_module

    target = variant_path(variant, ctc, out)
    if variant in ("trt-fp16", "trt-mixed"):
        return builder_module.build_engine(ctc / "model.onnx", target,
                                           precision=variant.removeprefix("trt-"))
    qdq = variant_path("int8-static", ctc, out)
    if not qdq.is_file():
        raise FileNotFoundError(f"{variant} is built from {qdq}; quantize int8-static first")
    import tensorrt as trt

    if builder_module.strongly_typed_only(trt):
        raise RuntimeError("INT8 engines here use TensorRT 10's weak-typing builder flags")
    options = builder_module.check_build_options(None)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(qdq)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, options.workspace_bytes)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    if variant == "trt-int8-mixed":
        builder_module.pin_sensitive_layers(trt, network)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    profile = builder.create_optimization_profile()
    profile.set_shape("audio", (1, builder_module.DEFAULT_MIN_SAMPLES),
                      (1, builder_module.DEFAULT_OPT_SAMPLES),
                      (1, builder_module.DEFAULT_MAX_SAMPLES))
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed; inspect the builder log")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bytes(serialized))
    return target


def file_bytes(path: Path) -> int:
    """Size of a graph or engine, including ONNX external-data files beside it."""
    if path.suffix == ".onnx":
        from fast_omniasr._onnx_deps import external_data_locations

        return path.stat().st_size + sum((path.parent / f).stat().st_size
                                         for f in external_data_locations(path))
    return path.stat().st_size


# --- evaluation -----------------------------------------------------------------------

def make_transcriber(variant: str, path: Path, tokenizer: Path, device: str, threads: int):
    """Return clip waveform -> text. ONNX graphs run through a plain session with the
    package's options (the same preprocessing, greedy decoding and tokenizer as
    OmniASR.transcribe_numpy); on CUDA, falling back to CPU for any node is an error."""
    from fast_omniasr import OmniASR

    if variant in ENGINE_VARIANTS:
        model = OmniASR(path, tokenizer, backend="tensorrt")
        return lambda waveform, rate: model.transcribe_numpy(waveform, rate).text
    import onnxruntime as ort

    from fast_omniasr.audio import prepare_audio
    from fast_omniasr.decoder import greedy_token_ids
    from fast_omniasr.options import cuda_provider_options
    from fast_omniasr.tokenizer import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    # As ONNXBackend does: the memory pattern planner inflates half-precision arenas.
    options.enable_mem_pattern = variant != "fp16"
    if device == "cuda":
        ort.preload_dlls()
        providers = [("CUDAExecutionProvider", cuda_provider_options())]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if session.get_providers()[0] != expected:
        raise RuntimeError(f"{variant}: {expected} did not initialize")
    decode = Tokenizer(tokenizer).decode

    def run(waveform, rate):
        logits = session.run(["logits"], {"audio": prepare_audio(waveform, rate)})[0]
        return decode(greedy_token_ids(logits))

    return run


def cpu_placed_ops(path: str) -> dict[str, int]:
    """Operators ONNX Runtime ran outside the CUDA provider for one second of silence, by
    op type, read from a profile of a throwaway session. Even the float32 graph keeps
    shape arithmetic on the CPU; quantized operators without a CUDA kernel show up here
    too, with a host round trip around every one of them."""
    import collections
    import tempfile

    import numpy as np
    import onnxruntime as ort

    ort.preload_dlls()
    with tempfile.TemporaryDirectory() as scratch:
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(scratch) / "profile")
        session = ort.InferenceSession(path, sess_options=options,
                                       providers=["CUDAExecutionProvider"])
        session.run(["logits"], {"audio": np.zeros((1, 16000), dtype=np.float32)})
        events = json.loads(Path(session.end_profiling()).read_text(encoding="utf-8"))
    return dict(sorted(collections.Counter(
        event["args"]["op_name"] for event in events
        if event.get("cat") == "Node" and event.get("name", "").endswith("_kernel_time")
        and event.get("args", {}).get("provider") not in (None, "CUDAExecutionProvider")
    ).items()))


def peak_rss_mib() -> float:
    """Peak resident memory of this process. VmHWM starts afresh at exec, unlike
    ru_maxrss, which a spawned worker inherits from the parent that forked it."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # kB
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def run_variant(variant: str, path: str, tokenizer: str, device: str, threads: int,
                clips: list[str], repeats: int) -> dict:
    """In a fresh process: load, warm up, transcribe every clip `repeats` times. Texts come
    from the first pass; later passes only time (and must reproduce them)."""
    import soundfile as sf

    audio = [sf.read(clip, dtype="float32") for clip in clips]
    sampler = fleurs_eval._metrics.GpuMemorySampler() if device == "cuda" else None
    if sampler is not None:
        sampler.__enter__()
    try:
        start = time.perf_counter()
        try:
            run = make_transcriber(variant, Path(path), Path(tokenizer), device, threads)
        except Exception as exc:  # noqa: BLE001 -- recorded in the report as unsupported
            return {"error": f"{type(exc).__name__}: {exc}".splitlines()[0][:500]}
        load_seconds = time.perf_counter() - start
        run(*audio[0])  # warm-up
        texts, passes, stable = None, [], True
        for _ in range(repeats):
            times, outputs = [], []
            for waveform, rate in audio:
                start = time.perf_counter()
                outputs.append(run(waveform, rate))
                times.append(time.perf_counter() - start)
            passes.append(times)
            if texts is None:
                texts = outputs
            stable = stable and outputs == texts
    finally:
        if sampler is not None:
            sampler.__exit__(None, None, None)
    return {"texts": texts, "passes": passes, "deterministic": stable,
            "load_seconds": load_seconds, "peak_rss_mib": peak_rss_mib(),
            "peak_gpu_mib": sampler.peak if sampler is not None else None}


def in_fresh_process(function, *args):
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(function, args)


def latency_summary(passes: list[list[float]], audio_seconds: float) -> dict:
    """Total compute seconds per pass (median over passes is the headline), its real-time
    factor, and per-clip p50/p90 from the median pass."""
    totals = [sum(p) for p in passes]
    median = statistics.median(totals)
    middle = sorted(passes, key=sum)[len(passes) // 2]
    clips = sorted(middle)
    return {"total_seconds": round(median, 3), "pass_totals": [round(t, 3) for t in totals],
            "rtf": round(median / audio_seconds, 5),
            "clip_p50_ms": round(1000 * clips[len(clips) // 2], 1),
            "clip_p90_ms": round(1000 * clips[min(len(clips) - 1, int(0.9 * len(clips)))], 1)}


def per_language_scores(manifest: dict, languages: list[str], counts: dict, texts: list[str]):
    scores, offset = {}, 0
    for fleurs in languages:
        entry = manifest["languages"][fleurs]
        hypotheses = texts[offset:offset + counts[fleurs]]
        offset += counts[fleurs]
        references = [u["reference"] for u in entry["utterances"][:len(hypotheses)]]
        scores[fleurs] = fleurs_eval.score(references, hypotheses, entry["metric"])
    return scores


def quality_failures(scores: dict, baselines: dict, gates: dict, manifest: dict) -> list[str]:
    """Languages whose gated metric (WER, plus CER; CER only for unsegmented languages)
    exceeds baseline + max(absolute, relative * baseline) from gates.json["quality"]."""
    failures = []
    for fleurs, measured in scores.items():
        baseline = baselines.get(fleurs)
        if baseline is None:
            failures.append(f"{fleurs}: no fp32 baseline")
            continue
        gated = ["cer"] if manifest["languages"][fleurs]["metric"] == "cer" else ["wer", "cer"]
        for metric in gated:
            rule, reference = gates["quality"][metric], baseline[f"baseline_{metric}"]
            limit = reference + max(rule["absolute"], rule["relative"] * reference)
            if measured[metric] is None or measured[metric] > limit + 1e-9:
                failures.append(f"{fleurs}: {metric.upper()} {measured[metric]:.2f} > "
                                f"{limit:.2f} (fp32 {reference:.2f})")
    return failures


def efficiency_check(result: dict, reference: dict | None, gates: dict, device: str):
    """(passed, why): faster than the device's reference by gates' min_speedup, or using
    max_memory_ratio of its memory (RSS on CPU, GPU memory on CUDA) or less."""
    if reference is None:
        return False, "speed reference variant was not measured in this run"
    rule = gates["efficiency"]
    speedup = reference["latency"]["total_seconds"] / result["latency"]["total_seconds"]
    key = "peak_gpu_mib" if device == "cuda" else "peak_rss_mib"
    ratio = (result[key] / reference[key]) if result.get(key) and reference.get(key) else None
    why = (f"speedup {speedup:.3f}x (needs >= {rule['min_speedup']}), memory ratio "
           + (f"{ratio:.3f}" if ratio else "n/a") + f" (or <= {rule['max_memory_ratio']})")
    passed = speedup >= rule["min_speedup"] or (ratio is not None
                                                and ratio <= rule["max_memory_ratio"])
    return passed, why


def evaluate(args) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fleurs_eval.validate_manifest(manifest)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["models"]["ctc"]
    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    variants = args.variants or list(gates["variants"][args.device])
    check_device_variants(args.device, variants)
    languages = args.languages or list(manifest["languages"])
    per_language = args.per_language or thresholds["per_language"]
    if args.gate and (per_language != thresholds["per_language"]
                      or set(languages) != set(manifest["languages"])):
        raise SystemExit("--gate compares against fp32 baselines over every language with "
                         f"{thresholds['per_language']} utterances; drop --languages and "
                         "--per-language")
    paths = fleurs_eval.fetch_audio(manifest, args.audio_dir, languages, per_language,
                                    args.cache_dir)
    counts = {fleurs: len(paths[fleurs]) for fleurs in languages}
    clips = [str(p) for fleurs in languages for p in paths[fleurs]]
    audio_seconds = sum(u["seconds"] for fleurs in languages
                        for u in manifest["languages"][fleurs]["utterances"][:counts[fleurs]])
    tokenizer = args.ctc / "tokenizer.model"
    results, reference_texts = {}, None
    for variant in variants:
        path = variant_path(variant, args.ctc, args.variants_dir)
        if not path.is_file():
            results[variant] = {"error": f"{path} not found; run `quantize` first"}
            print(json.dumps({"variant": variant, **results[variant]}), flush=True)
            continue
        measured = in_fresh_process(run_variant, variant, str(path), str(tokenizer),
                                    args.device, args.threads, clips, args.repeats)
        if "error" in measured:
            results[variant] = measured
            print(json.dumps({"variant": variant, **measured}), flush=True)
            continue
        texts = measured["texts"]
        reference_texts = texts if variant == "fp32" else reference_texts
        result = {
            "bytes": file_bytes(path),
            "languages": per_language_scores(manifest, languages, counts, texts),
            "latency": latency_summary(measured["passes"], audio_seconds),
            "load_seconds": round(measured["load_seconds"], 2),
            "peak_rss_mib": round(measured["peak_rss_mib"]),
            "peak_gpu_mib": measured["peak_gpu_mib"],
            "deterministic": measured["deterministic"],
            "identical_to_fp32": (sum(a.split() == b.split()
                                      for a, b in zip(reference_texts, texts))
                                  if reference_texts is not None else None),
        }
        if args.device == "cuda" and variant in ONNX_VARIANTS:
            result["cpu_placed_ops"] = in_fresh_process(cpu_placed_ops, str(path))
        result["hypotheses"] = dict(zip((Path(c).name for c in clips), texts))
        results[variant] = result
        summary = {k: v for k, v in result.items() if k != "hypotheses"}
        print(json.dumps({"variant": variant, **summary}), flush=True)
    speed_reference = results.get(gates["speed_reference"][args.device])
    if speed_reference is not None and "error" in speed_reference:
        speed_reference = None
    decisions = {}
    for variant, result in results.items():
        if "error" in result:
            decisions[variant] = {"ship": False, "reasons": [result["error"]]}
            continue
        quality = quality_failures(result["languages"], thresholds["languages"], gates,
                                   manifest)
        if variant in gates["shipped"][args.device]:
            decisions[variant] = {"ship": None, "reasons": ["already shipped (reference)"],
                                  "quality_failures": quality}
            continue
        efficient, why = efficiency_check(result, speed_reference, gates, args.device)
        reasons = quality + ([] if efficient else [f"not efficient enough: {why}"])
        decisions[variant] = {"ship": not reasons, "reasons": reasons, "efficiency": why}
    report = {"model": str(args.ctc), "device": args.device, "threads": args.threads,
              "repeats": args.repeats, "dataset": fleurs_eval.DATASET,
              "dataset_revision": manifest["revision"], "per_language": counts,
              "audio_seconds": round(audio_seconds, 2), "gates": gates,
              "environment": environment(args.device), "variants": results,
              "decisions": decisions}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    for variant, decision in decisions.items():
        verdict = {True: "PASS", False: "FAIL", None: "REFERENCE"}[decision["ship"]]
        print(f"{verdict} {variant}: " + ("; ".join(decision["reasons"]) or "all gates met"),
              file=sys.stderr)
    return 1 if args.gate and not any(d["ship"] for d in decisions.values()) else 0


def environment(device: str) -> dict:
    import platform
    from importlib import metadata

    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    import onnxruntime

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "processor": platform.processor() or None,
           "onnxruntime": onnxruntime.__version__, "onnx": version("onnx")}
    if device == "cuda":
        try:
            import tensorrt

            env["tensorrt"] = tensorrt.__version__
        except ImportError:
            env["tensorrt"] = None
        import subprocess

        try:
            env["gpu"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True, check=True,
                timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            env["gpu"] = None
    return env


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    calib = sub.add_parser("calibration", help="extract FLEURS dev clips for static INT8")
    calib.add_argument("--out", type=Path, required=True)
    calib.add_argument("--per-language", type=_positive_int, default=2)
    calib.add_argument("--languages", nargs="+", default=list(fleurs_eval.LANGUAGES))
    calib.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    quant = sub.add_parser("quantize", help="build quantized graphs or TensorRT engines")
    quant.add_argument("--ctc", type=Path, required=True,
                       help="CTC export directory (model.onnx, tokenizer.model)")
    quant.add_argument("--out", type=Path, required=True, help="variants directory")
    quant.add_argument("--variants", nargs="+",
                       choices=[v for v in VARIANTS if v not in ("fp32", "fp16")],
                       default=["int8-dynamic", "int8-static", "int8-weight-only",
                                "int4-weight-only"])
    quant.add_argument("--calibration", type=Path, help="calibration.json (int8-static)")
    ev = sub.add_parser("evaluate", help="per-language WER/CER, latency, memory, gates")
    ev.add_argument("--ctc", type=Path, required=True)
    ev.add_argument("--variants-dir", type=Path, required=True)
    ev.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ev.add_argument("--variants", nargs="+", choices=VARIANTS,
                    help="default: gates.json's list for the device")
    ev.add_argument("--threads", type=_positive_int, default=4,
                    help="ONNX Runtime intra-op threads (default 4, the package default)")
    ev.add_argument("--repeats", type=_positive_int, default=3,
                    help="timed passes over the whole set; the median is reported")
    ev.add_argument("--languages", nargs="+")
    ev.add_argument("--per-language", type=_positive_int)
    ev.add_argument("--manifest", type=Path, default=fleurs_eval.MANIFEST)
    ev.add_argument("--thresholds", type=Path, default=fleurs_eval.THRESHOLDS)
    ev.add_argument("--gates", type=Path, default=GATES)
    ev.add_argument("--audio-dir", type=Path, default=Path(".cache/fleurs"))
    ev.add_argument("--cache-dir", type=Path, help="dataset download cache")
    ev.add_argument("--gate", action="store_true",
                    help="exit 1 unless at least one candidate passes every gate")
    ev.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "calibration":
        manifest = json.loads(fleurs_eval.MANIFEST.read_text(encoding="utf-8"))
        print(calibration(args.out, args.per_language, manifest["revision"], args.languages,
                          args.cache_dir))
        return 0
    if args.command == "quantize":
        clips = read_calibration(args.calibration) if args.calibration else None
        for variant in args.variants:
            start = time.perf_counter()
            path = quantize(variant, args.ctc, args.out, clips)
            print(json.dumps({"variant": variant, "path": str(path), "bytes": file_bytes(path),
                              "seconds": round(time.perf_counter() - start, 1)}), flush=True)
        return 0
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())

"""Compare one batched bundle to its original decoder on the same clips and weights."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from fast_omniasr.llm import OmniASRLLM


def measure(model, clips, language, limits, batched, repeats):
    def run():
        if batched:
            return model.transcribe_batch(clips, lang=language, max_new_tokens=limits)
        return [model.transcribe(clip, lang=language, max_new_tokens=limit)
                for clip, limit in zip(clips, limits, strict=True)]

    run()  # Load sessions and warm up before measuring.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        results = run()
        samples.append(time.perf_counter() - start)
    return results, statistics.median(samples)


def measure_cache(model, prefixes, repeats):
    """Measure steady-state latency for a single row over identical forced tokens."""
    results = []
    backend = model.onnx
    for prefix in prefixes:
        backend.initialize_decoder()
        hidden = backend.empty_context.shape[2]
        ctx = np.random.default_rng(123).normal(size=(1, prefix, hidden)).astype(np.float32)
        token = np.full((1, 1), model.config["bos_idx"], dtype=np.int64)
        medians = []
        for _ in range(repeats):
            _, cache = backend.prefill(ctx, token)
            times = []
            for _ in range(48):
                start = time.perf_counter()
                _, cache = backend.decode_step(token, cache)
                times.append(time.perf_counter() - start)
            medians.append(statistics.median(times[8:]) * 1000)
        size = sum(value.nbytes if isinstance(value, np.ndarray)
                   else np.prod(value.shape()) * backend.empty_cache[0].dtype.itemsize
                   for value in cache)
        results.append({"prefix_positions": prefix, "cache_bytes_after_48_steps": int(size),
                        "median_step_ms": statistics.median(medians)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limits", nargs="+", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-prefixes", nargs="*", type=int, default=[],
                        help="Also compare single-row cache size and step time at these prefix lengths")
    args = parser.parse_args()
    if args.repeats < 1 or len(args.limits) != len(args.clips):
        parser.error("Provide a positive repeat count and one token limit per clip")

    def load():
        return OmniASRLLM(args.directory, backend="onnx", device=args.device,
                          precision=args.precision)

    with load() as model:
        if not model.onnx.batched:
            parser.error("The bundle needs decoder_batched.onnx")
        if any(p < 1 or p + 49 > model.config["max_seq_len"] for p in args.cache_prefixes):
            parser.error("Cache probe exceeds the decoder context range")
        model.onnx.batched = False
        baseline, serial_s = measure(model, args.clips, args.lang, args.limits, False, args.repeats)
        legacy_cache = measure_cache(model, args.cache_prefixes, args.repeats)
    with load() as model:
        batch, batch_s = measure(model, args.clips, args.lang, args.limits, True, args.repeats)
        new_cache = measure_cache(model, args.cache_prefixes, args.repeats)
    matches = [old.token_ids == new.token_ids and old.stop_reason == new.stop_reason
               for old, new in zip(baseline, batch, strict=True)]
    print(json.dumps({
        "precision": args.precision,
        "clips": [clip.name for clip in args.clips],
        "limits": args.limits,
        "token_and_stop_parity": matches,
        "serial_seconds": serial_s,
        "batch_seconds": batch_s,
        "serial_clips_per_second": len(batch) / serial_s,
        "batch_clips_per_second": len(batch) / batch_s,
        "repeats": args.repeats,
        "original_cache": legacy_cache,
        "batched_decoder_cache": new_cache,
    }, indent=2))
    if not all(matches):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

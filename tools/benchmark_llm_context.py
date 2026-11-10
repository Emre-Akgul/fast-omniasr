"""Measure what keeping the LLM encoder context on the GPU through prefill would save (#46).

The ONNX runtime copies the encoder output (`context`, [1, frames, hidden] float32) to the
host, replaces the last row with the language embedding there, and binds it back as a CPU
input to the decoder's prefill. This times both halves of that round trip against the
device-resident alternative (encoder output bound to CUDA and passed to prefill as an
OrtValue), checks that the prefill logits are identical, and reports the share of a full
transcription that the copies take.

    python tools/benchmark_llm_context.py --model EmreAkgul/omniASR-LLM-300M-v2-ONNX \\
        --precision fp16 --output context.json
"""

import argparse
import gc
import json
import statistics
import sys
import time

import numpy as np


def median_ms(run, repeats: int) -> float:
    run()  # warm-up: allocator growth and kernel selection happen here
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


def measure(model, seconds: float, repeats: int, tokens: int) -> dict:
    backend = model.onnx
    audio = (np.random.default_rng(0).standard_normal((1, int(16000 * seconds))) * 0.05)
    audio = audio.astype(np.float32)
    backend.encode(audio)  # loads the encoder session
    backend._initialize_unified()
    encoder = backend.encoder_session

    def encode_to_host():
        return encoder.run(["context"], {"audio": audio})[0]

    def encode_on_device():
        binding = encoder.io_binding()
        binding.bind_cpu_input("audio", audio)
        binding.bind_output("context", "cuda", backend.device_index or 0)
        encoder.run_with_iobinding(binding)
        return binding.get_outputs()[0]

    token = np.array([[model.config["bos_idx"]]], np.int64)
    caches = dict(zip([value.name for value in backend.decoder.get_inputs()][2:],
                      backend.empty_cache, strict=True))
    context = encode_to_host()
    resident = encode_on_device()

    def prefill_from_host():
        return backend._run_bound(backend.decoder, {"context": context, "tokens": token, **caches})

    def prefill_from_device():
        return backend._run_bound(backend.decoder, {"tokens": token, **caches},
                                  [("context", resident)])

    result = {
        "seconds": seconds,
        "context_shape": list(context.shape),
        "context_mib": round(context.nbytes / 2**20, 2),
        "encode_to_host_ms": median_ms(encode_to_host, repeats),
        "encode_on_device_ms": median_ms(encode_on_device, repeats),
        "prefill_from_host_ms": median_ms(prefill_from_host, repeats),
        "prefill_from_device_ms": median_ms(prefill_from_device, repeats),
        "prefill_logits_identical": bool(np.array_equal(prefill_from_host()[0],
                                                        prefill_from_device()[0])),
    }
    resident = None  # free the extra device copy before timing a full transcription
    gc.collect()
    result["transcribe_ms"] = median_ms(
        lambda: model.transcribe(audio[0], lang="eng_Latn", max_new_tokens=tokens),
        max(1, repeats // 3),
    )
    saved = (result["encode_to_host_ms"] - result["encode_on_device_ms"]
             + result["prefill_from_host_ms"] - result["prefill_from_device_ms"])
    result["saved_ms"] = saved
    result["saved_share"] = saved / result["transcribe_ms"]
    return {key: round(value, 4) if isinstance(value, float) else value
            for key, value in result.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seconds", type=float, nargs="+", default=[5.0, 15.0, 30.0])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--tokens", type=int, default=64, help="decode budget per transcription")
    parser.add_argument("--output", help="write the results as JSON")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.tokens < 1:
        parser.error("--repeats and --tokens must be positive")
    from fast_omniasr.llm import OmniASRLLM

    model = OmniASRLLM.from_pretrained(
        args.model, device="cuda", precision=args.precision, device_index=args.device_index,
        cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    if model.onnx.low_memory or not model.onnx.unified:
        raise SystemExit("needs the resident unified decoder (the default)")
    rows = []
    with model:
        for seconds in args.seconds:
            rows.append(measure(model, seconds, args.repeats, args.tokens))
            print(json.dumps(rows[-1]), flush=True)
    record = {"model": args.model, "commit": model.snapshot.commit,
              "precision": args.precision, "tokens": args.tokens,
              "repeats": args.repeats, "results": rows}
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

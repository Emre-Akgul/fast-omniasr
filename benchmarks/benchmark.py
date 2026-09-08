"""Basic warmed ONNX end-to-end transcription benchmark."""
import argparse
import json
import statistics
import time

from fast_omniasr import OmniASR

parser = argparse.ArgumentParser()
parser.add_argument("audio")
parser.add_argument("--model", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--backend", choices=["onnx", "tensorrt"], default="onnx")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
parser.add_argument("--runs", type=int, default=10)
args = parser.parse_args()
if args.runs < 1:
    parser.error("--runs must be positive")
start = time.perf_counter()
model = OmniASR(args.model, args.tokenizer, backend=args.backend, device=args.device)
load = time.perf_counter() - start
start = time.perf_counter()
result = model.transcribe(args.audio)
first = time.perf_counter() - start
samples = []
for _ in range(args.runs):
    start = time.perf_counter()
    model.transcribe(args.audio)
    samples.append(time.perf_counter() - start)
print(json.dumps({"backend": args.backend, "load_seconds": load, "first_seconds": first,
                  "median_warm_seconds": statistics.median(samples),
                  "samples": samples, "transcript": result.text,
                  "scope": "WAV loading + normalization + inference + decoding"}, indent=2))

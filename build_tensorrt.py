"""Build a local engine from the FP32 dynamic ONNX graph."""
import argparse
from pathlib import Path

from fast_omniasr.tensorrt_builder import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_OPT_SAMPLES,
    build_engine,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--opt-samples", type=int, default=DEFAULT_OPT_SAMPLES)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    args = parser.parse_args()
    build_engine(
        args.model,
        args.output,
        precision=args.precision,
        min_samples=args.min_samples,
        opt_samples=args.opt_samples,
        max_samples=args.max_samples,
    )
    print(args.output)


if __name__ == "__main__":
    main()

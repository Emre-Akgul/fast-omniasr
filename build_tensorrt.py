"""Build a local engine from the FP32 dynamic ONNX graph."""
import argparse
from pathlib import Path

from fast_omniasr.tensorrt_builder import build_engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    build_engine(args.model, args.output, precision=args.precision)
    print(args.output)


if __name__ == "__main__":
    main()

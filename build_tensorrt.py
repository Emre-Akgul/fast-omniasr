"""Build a local engine from the FP32 dynamic ONNX graph."""
import argparse
from pathlib import Path

from fast_omniasr.tensorrt_builder import (
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_OPT_SAMPLES,
    DEFAULT_WORKSPACE_BYTES,
    TensorRTBuildOptions,
    build_engine,
    validate_profile,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "mixed"], default="fp32",
                        help="mixed: FP16 with the precision-sensitive front end in FP32 "
                             "(TensorRT 10 only)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--opt-samples", type=int, default=DEFAULT_OPT_SAMPLES)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--workspace-bytes", type=int, default=DEFAULT_WORKSPACE_BYTES,
                        help="workspace memory limit per layer")
    parser.add_argument("--optimization-level", type=int,
                        help="0 to 5; TensorRT's default when omitted")
    parser.add_argument("--timing-cache",
                        help="file of timing measurements, reused and updated by each build")
    args = parser.parse_args()
    try:
        options = TensorRTBuildOptions(workspace_bytes=args.workspace_bytes,
                                       optimization_level=args.optimization_level,
                                       timing_cache=args.timing_cache)
        validate_profile(args.min_samples, args.opt_samples, args.max_samples)
    except ValueError as exc:
        parser.error(str(exc))
    build_engine(
        args.model,
        args.output,
        precision=args.precision,
        min_samples=args.min_samples,
        opt_samples=args.opt_samples,
        max_samples=args.max_samples,
        options=options,
    )
    print(args.output)


if __name__ == "__main__":
    main()

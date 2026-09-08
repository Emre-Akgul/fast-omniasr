"""Shared TensorRT engine build logic, used by build_tensorrt.py and OmniASR.from_pretrained()."""
import os
from pathlib import Path

# Bump when the build recipe changes (flags, workspace size, profile semantics) so cached
# engines built under the old recipe are never mistaken for compatible with the new one.
BUILDER_CONFIG_VERSION = 1


def build_engine(onnx_path: str | Path, output_path: str | Path, *, precision: str = "fp32",
                  min_samples: int = 16000, opt_samples: int = 80000,
                  max_samples: int = 480000) -> Path:
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be 'fp32' or 'fp16'")
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "Install fast-omniasr[tensorrt] and a compatible TensorRT installation"
        ) from exc
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shapes = [(1, min_samples), (1, opt_samples), (1, max_samples)]
    profile.set_shape("audio", *shapes)
    if [tuple(s) for s in profile.get_shape("audio")] != shapes:
        raise RuntimeError("Unexpected optimization profile")
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed; inspect the builder log")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_bytes(bytes(serialized))
    os.replace(tmp_path, output_path)
    return output_path

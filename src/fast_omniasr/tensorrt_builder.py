"""Shared TensorRT engine build logic, used by build_tensorrt.py and OmniASR.from_pretrained()."""
import operator
import os
from pathlib import Path

from .audio import MIN_AUDIO_SAMPLES

# Bump when the build recipe changes (flags, workspace size, profile semantics) so cached
# engines built under the old recipe are never mistaken for compatible with the new one.
BUILDER_CONFIG_VERSION = 2
DEFAULT_MIN_SAMPLES = MIN_AUDIO_SAMPLES
DEFAULT_OPT_SAMPLES = 80000
DEFAULT_MAX_SAMPLES = 480000

PRECISIONS = ("fp32", "fp16", "mixed")
# The precision-sensitive part of the OmniASR CTC graph under FP16 is its convolutional
# front end: the 7-block waveform feature extractor and the positional convolution.
# Keeping exactly these layers in FP32 brought FP16 engines much closer to FP32 on the
# FLEURS sample (median logit error 2.1 -> 0.5, 141 -> 150/160 identical transcripts) at
# ~2x the FP16 latency; normalization, softmax, attention, residuals, the final projection
# and every encoder block made no measurable difference (benchmarks/
# tensorrt_fp16_sensitivity.json).
MIXED_FP32_LAYER_PREFIXES = ("/model/feature_extractor/",)
MIXED_FP32_LAYER_NAMES = frozenset({"/model/pos_encoder/conv/Conv"})


def _is_precision_sensitive(name: str) -> bool:
    return name in MIXED_FP32_LAYER_NAMES or name.startswith(MIXED_FP32_LAYER_PREFIXES)


def pin_sensitive_layers(trt, network) -> int:
    """Force the precision-sensitive layers' compute and floating-point outputs to FP32.
    Returns how many layers were pinned (constants and shape layers are skipped)."""
    skip = {trt.LayerType.CONSTANT, trt.LayerType.SHAPE}
    floating = (trt.float32, trt.float16)
    pinned = 0
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.type in skip or not _is_precision_sensitive(layer.name):
            continue
        outputs = [i for i in range(layer.num_outputs)
                   if layer.get_output(i).dtype in floating]
        if not outputs:
            continue
        layer.precision = trt.float32
        for i in outputs:
            layer.set_output_type(i, trt.float32)
        pinned += 1
    return pinned


def validate_profile(min_samples: int, opt_samples: int, max_samples: int) -> tuple[int, int, int]:
    try:
        min_samples, opt_samples, max_samples = (
            operator.index(value) for value in (min_samples, opt_samples, max_samples)
        )
    except TypeError as exc:
        raise ValueError("TensorRT profile sample counts must be integers") from exc
    if min_samples < MIN_AUDIO_SAMPLES:
        raise ValueError(
            f"TensorRT min_samples must be at least {MIN_AUDIO_SAMPLES}, the model's "
            "feature-extractor minimum"
        )
    if not min_samples <= opt_samples <= max_samples:
        raise ValueError("TensorRT profile must satisfy min_samples <= opt_samples <= max_samples")
    return min_samples, opt_samples, max_samples


def build_engine(onnx_path: str | Path, output_path: str | Path, *, precision: str = "fp32",
                  min_samples: int = DEFAULT_MIN_SAMPLES,
                  opt_samples: int = DEFAULT_OPT_SAMPLES,
                  max_samples: int = DEFAULT_MAX_SAMPLES) -> Path:
    """`precision`: "fp32"; "fp16" (fastest, small transcript differences from FP32); or
    "mixed" (FP16 with the precision-sensitive convolutional front end kept in FP32:
    closest to FP32, about twice the FP16 latency)."""
    if precision not in PRECISIONS:
        raise ValueError("precision must be 'fp32', 'fp16' or 'mixed'")
    min_samples, opt_samples, max_samples = validate_profile(
        min_samples, opt_samples, max_samples
    )
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
    if precision in ("fp16", "mixed"):
        config.set_flag(trt.BuilderFlag.FP16)
    if precision == "mixed":
        if pin_sensitive_layers(trt, network) == 0:
            raise ValueError(
                "precision='mixed' found none of the expected OmniASR front-end layers "
                "(/model/feature_extractor/..., /model/pos_encoder/conv/Conv) in this graph; "
                "use precision='fp16' or 'fp32'"
            )
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
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

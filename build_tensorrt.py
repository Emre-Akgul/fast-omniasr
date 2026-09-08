"""Build a local engine from the FP32 dynamic ONNX graph."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    reader = trt.OnnxParser(network, logger)
    if not reader.parse_from_file(str(args.model)):
        raise RuntimeError("\n".join(str(reader.get_error(i)) for i in range(reader.num_errors)))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024**3)
    if args.precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("audio", (1, 16000), (1, 80000), (1, 480000))
    if [tuple(s) for s in profile.get_shape("audio")] != [(1,16000), (1,80000), (1,480000)]:
        raise RuntimeError("Unexpected optimization profile")
    config.add_optimization_profile(profile)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT build failed; inspect the builder log")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(engine))
    print(args.output)


if __name__ == "__main__":
    main()

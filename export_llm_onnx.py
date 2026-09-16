"""Export Day-3 ONNX graphs from frozen Day-2 TorchScript oracles."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", choices=["encoder", "decoder"], default="encoder")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Use a separate output directory to preserve Day-2 artifacts")
    if args.stage == "decoder":
        if not (args.output / "encoder_parity.json").exists():
            raise ValueError("Encoder parity must pass before decoder export")
        from tools.llm_onnx_graphs import export_decoder

        if args.validate_only:
            raise ValueError("Decoder raw-tensor validation uses tools/validate_llm_onnx.py")
        if any(
            (args.output / n).exists()
            for n in ["decoder_prefill.onnx", "decoder_step.onnx", "decoder.weights"]
        ):
            raise FileExistsError("Decoder artifacts already exist")
        model = torch.jit.load(str(args.source / "decoder_cached.pt")).eval()
        if len(list(model.layers.children())) != 12 or model.embedding.weight.shape[1] != 4096:
            raise ValueError("Only the validated 300M v2 decoder is supported")
        export_decoder(model, args.output)
        config = json.loads((args.source / "config.json").read_text())
        config["onnx"] = {
            "opset": 18,
            "layers": 12,
            "kv_heads": 8,
            "head_dim": 512,
            "cache_layout": "batch, sequence, kv_heads, head_dim",
            "weight_file": "decoder.weights",
            "initializer_manifest": "decoder_weights.json",
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ["tokenizer.model", "reference.json", "cached_reference.json"]:
            shutil.copyfile(args.source / name, args.output / name)
        suite = json.loads((args.source / "cached_reference.json").read_text())
        for clip in suite["clips"]:
            shutil.copyfile(args.source / clip["audio"], args.output / clip["audio"])
        return
    target = args.output / "encoder.onnx"
    if target.exists() and not args.validate_only:
        raise FileExistsError(target)
    model = torch.jit.load(str(args.source / "encoder.pt")).eval()
    from torch.onnx import symbolic_helper

    @symbolic_helper.parse_args("v", "i", "is")
    def unflatten_last(g, value, dim, sizes):
        rank = symbolic_helper._get_tensor_rank(value)
        if rank is None or dim % rank != rank - 1:
            raise ValueError("Expected last-axis unflatten with known rank")
        shape = g.op("Constant", value_t=torch.tensor([0] * (rank - 1) + sizes))
        result = g.op("Reshape", value, shape)
        return result.setType(
            value.type().with_sizes([None] * (rank - 1) + [None if n == -1 else n for n in sizes])
        )

    @symbolic_helper.parse_args("v", "is", "v", "v", "f", "b")
    def layer_norm(g, x, shape, w, b, eps, cudnn):
        axes = g.op(
            "Constant", value_t=torch.tensor(list(range(-len(shape), 0)), dtype=torch.int64)
        )
        d = g.op("Cast", x, to_i=11)
        mean = g.op("ReduceMean", d, axes, keepdims_i=1)
        centered = g.op("Sub", d, mean)
        var = g.op("ReduceMean", g.op("Mul", centered, centered), axes, keepdims_i=1)
        std = g.op(
            "Sqrt",
            g.op("Add", var, g.op("Constant", value_t=torch.tensor(eps, dtype=torch.float64))),
        )
        norm = g.op("Cast", g.op("Div", centered, std), to_i=1)
        return g.op("Add", g.op("Mul", norm, w), b)

    torch.onnx.register_custom_op_symbolic("aten::layer_norm", layer_norm, 18)
    torch.onnx.register_custom_op_symbolic("aten::unflatten", unflatten_last, 18)
    with torch.inference_mode():
        if not args.validate_only:
            torch.onnx.export(
                model,
                (torch.randn(1, 48000),),
                str(target),
                input_names=["audio"],
                output_names=["context"],
                opset_version=18,
                dynamo=False,
                dynamic_axes={"audio": {1: "samples"}, "context": {1: "audio_positions"}},
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(target), sess_options=options, providers=["CPUExecutionProvider"]
        )
        probes = []
        rng = np.random.default_rng(7)
        for samples in [400, 8000, 16000, 48000, 80000, 480000]:
            audio = rng.normal(size=(1, samples)).astype(np.float32)
            expected = model(torch.from_numpy(audio)).numpy()
            actual = session.run(None, {"audio": audio})[0]
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise AssertionError("Encoder shape mismatch or non-finite output")
            difference = np.abs(actual - expected)
            # A learned high-gain output channel crosses zero. Use a per-channel
            # reference scale for the gate; still report raw elementwise relative errors.
            channel_scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1.0)
            scaled_max = float((difference / channel_scale).max())
            relative_l2 = float(np.linalg.norm(difference) / np.linalg.norm(expected))
            if scaled_max > 1e-3 or relative_l2 > 1e-4:
                raise AssertionError(f"Encoder error exceeds gate: {scaled_max=}, {relative_l2=}")
            report = {
                "samples": samples,
                "shape": list(actual.shape),
                "max_abs_error": float(difference.max()),
                "max_relative_error": float(
                    (difference / np.maximum(np.abs(expected), 1e-8)).max()
                ),
                "relative_denominator_floor": 1e-8,
                "max_channel_scaled_error": scaled_max,
                "relative_l2_error": relative_l2,
            }
            probes.append(report)
            print(json.dumps(report), flush=True)
    (args.output / "encoder_parity.json").write_text(
        json.dumps(
            {
                "max_channel_scaled_error_limit": 1e-3,
                "relative_l2_error_limit": 1e-4,
                "channel_scale": "max over sequence of abs(reference), floored at 1",
                "probes": probes,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

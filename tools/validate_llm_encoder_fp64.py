"""Compare the ONNX and TorchScript float32 encoders with a float64 fairseq2 reference.

ONNX-vs-PyTorch float32 differences cannot separate export error from PyTorch's own
rounding: on some inputs PyTorch float32 is the less accurate of the two. Running the
reference model in float64 gives near-exact outputs, and the gate requires the ONNX
encoder's error on real speech to stay within `--factor` times PyTorch float32's error
(or below an absolute floor). fairseq2 is used only here.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def errors(actual, expected):
    difference = np.abs(actual - expected)
    scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1.0)
    return {
        "max_abs_error": float(difference.max()),
        "max_channel_scaled_error": float((difference / scale).max()),
        "relative_l2_error": float(np.linalg.norm(difference) / np.linalg.norm(expected)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory with encoder.pt and config.json")
    parser.add_argument("directory", type=Path, help="Directory with encoder.onnx")
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--floor", type=float, default=1e-5)
    args = parser.parse_args()
    import omnilingual_asr  # noqa: F401
    import onnxruntime as ort
    import torch
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

    from fast_omniasr.audio import load_audio

    threads = int(os.environ.get("FAST_OMNIASR_THREADS", "4"))
    torch.set_num_threads(threads)
    card = json.loads((args.source / "config.json").read_text())["model"]
    if args.checkpoint:
        hub = get_wav2vec2_llama_model_hub()
        model = hub.load_custom_model(
            args.checkpoint,
            hub.get_model_config(card),
            device=torch.device("cpu"),
            dtype=torch.float64,
            mmap=True,
        )
    else:
        model = load_model(card, device=torch.device("cpu"), dtype=torch.float64)
    model.eval()

    def reference(audio):
        # Same wiring as export_llm.Encoder, in float64.
        x = torch.from_numpy(audio).double()
        layout = BatchLayout(x.shape, seq_lens=None, device=x.device)
        x, layout, _ = model.encoder_frontend.extract_features(x, layout)
        x, _ = model.encoder_frontend.process_features(x, layout, None)
        x = model.encoder_proj(model.encoder(x, layout))
        suffix = torch.cat(
            [
                model.text_frontend(torch.tensor([[model.special_tokens.lid_marker]])),
                model.lang_embeddings(torch.tensor([[0]])),
            ],
            dim=1,
        )
        return torch.cat([x, suffix], dim=1).numpy()

    torch_encoder = torch.jit.load(str(args.source / "encoder.pt")).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(args.directory / "encoder.onnx"), options, providers=["CPUExecutionProvider"]
    )
    reports = []
    with torch.inference_mode():
        for clip in args.audio:
            audio = load_audio(clip)
            expected = reference(audio)
            torch_error = errors(torch_encoder(torch.from_numpy(audio)).numpy(), expected)
            onnx_error = errors(session.run(None, {"audio": audio})[0], expected)
            limit = max(args.floor, args.factor * torch_error["relative_l2_error"])
            report = {
                "audio": clip.name,
                "torch_float32": torch_error,
                "onnx_float32": onnx_error,
                "relative_l2_limit": limit,
                "passed": onnx_error["relative_l2_error"] <= limit,
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    summary = {
        "model": card,
        "reference": "fairseq2 float64 on CPU",
        "gate": f"onnx relative L2 <= max({args.floor}, {args.factor} x torch float32 relative L2)",
        "passed": all(report["passed"] for report in reports),
        "clips": reports,
    }
    (args.directory / "encoder_fp64.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise SystemExit("ONNX encoder is less accurate than the float64 gate allows")


if __name__ == "__main__":
    main()

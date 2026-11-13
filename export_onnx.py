"""One-time direct fairseq2 export; these dependencies are not runtime requirements."""
import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
# The repository's tools, also when this file is loaded by path (as the tests do).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import provenance

CTC_MODEL_CARDS = (
    "omniASR_CTC_300M",
    "omniASR_CTC_1B",
    "omniASR_CTC_3B",
    "omniASR_CTC_7B",
    "omniASR_CTC_300M_v2",
    "omniASR_CTC_1B_v2",
    "omniASR_CTC_3B_v2",
    "omniASR_CTC_7B_v2",
)

# Speech clips give the token check something to compare; the noise probes cover the
# shortest input the feature extractor accepts and the default TensorRT profile maximum.
VALIDATION_AUDIO = tuple(sorted((Path(__file__).parent / "tests/fixtures/fleurs").glob("*.flac")))
VALIDATION_NOISE_SAMPLES = (400, 480000)
LOGIT_RTOL = 1e-4
# Measured on the official omniASR_CTC_300M_v2 export (batch-one and --batch graphs alike,
# ONNX Runtime CPU against fairseq2 CPU, float32): logits span about +-21 and differ by up to
# 0.0021 (English FLEURS clip; 0.0009 on the others) with every greedy token equal. 1e-2
# leaves about 5x headroom over that float32 reordering noise and still fails real
# breakage: scaling only the output projection's weights by 1.001 moves logits by 0.021,
# and a random 0.1% relative change to every weight by about 3. Token flips are caught
# separately, whatever the tolerance.
LOGIT_ATOL = 1e-2
# Frames whose top two source logits are this close may flip under float32 reordering.
TIE_MARGIN = 1e-3
PARITY_REPORT = "export_parity.json"

TOKENIZER_URLS = {
    "v1": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
    "v2": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export any released Omnilingual ASR CTC model to dynamic ONNX."
    )
    parser.add_argument(
        "--model",
        choices=CTC_MODEL_CARDS,
        help=(
            "Base architecture and, without --checkpoint, upstream weights "
            "(official-export default: omniASR_CTC_300M_v2)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Fine-tuned fairseq2-compatible custom model checkpoint file or the model "
            "directory inside a native sharded fairseq2 step checkpoint"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validation-audio",
        type=Path,
        action="append",
        help=(
            "16 kHz mono clip for the parity check (repeatable; default: the committed "
            "FLEURS fixtures). Noise probes of 400 and 480,000 samples always run too."
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Export a batch graph: padded audio [batch, samples] plus audio_lengths [batch], "
            "returning logits [batch, frames, vocabulary] and logit_lengths [batch]"
        ),
    )
    parser.add_argument("--logit-atol", type=float, default=LOGIT_ATOL)
    parser.add_argument("--logit-rtol", type=float, default=LOGIT_RTOL)
    args = parser.parse_args(argv)
    if args.logit_atol < 0 or args.logit_rtol < 0:
        parser.error("--logit-atol and --logit-rtol must not be negative")
    if args.checkpoint is not None and args.model is None:
        parser.error("--model is required when --checkpoint is used")
    if args.model is None:
        args.model = "omniASR_CTC_300M_v2"
    return args


def load_source_model(args, torch, load_model, get_model_hub):
    device = torch.device("cpu")
    if args.checkpoint is None:
        return load_model(args.model, device=device, dtype=torch.float32)
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    hub = get_model_hub()
    config = hub.get_model_config(args.model)
    return hub.load_custom_model(
        args.checkpoint,
        config,
        device=device,
        dtype=torch.float32,
        mmap=True,
    )


def tokenizer_family(model_card: str) -> str:
    return "v2" if model_card.endswith("_v2") else "v1"


def sentencepiece_vocab_size(path: Path) -> int:
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(path)).vocab_size()


def prepare_tokenizer(
    model_card: str,
    destination: Path,
    expected_vocab_size: int,
    *,
    download=urlretrieve,
    get_vocab_size=sentencepiece_vocab_size,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    download(TOKENIZER_URLS[tokenizer_family(model_card)], temporary)

    actual_vocab_size = get_vocab_size(temporary)
    if actual_vocab_size != expected_vocab_size:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"The model outputs {expected_vocab_size} classes, but its supported "
            f"{tokenizer_family(model_card)} tokenizer has {actual_vocab_size} entries. "
            "Modified CTC heads and vocabularies are not supported."
        )
    temporary.replace(destination)
    return destination


def validation_inputs(audio_paths=VALIDATION_AUDIO, noise_samples=VALIDATION_NOISE_SAMPLES):
    """Named [1, samples] float32 inputs, normalized as the runtime normalizes audio."""
    from fast_omniasr.audio import load_audio, prepare_audio

    inputs = {Path(path).name: load_audio(path) for path in audio_paths}
    rng = np.random.default_rng(0)
    for samples in noise_samples:
        inputs[f"noise_{samples}"] = prepare_audio(rng.standard_normal(samples, np.float32))
    if not inputs:
        raise ValueError("No validation inputs")
    return inputs


def compare_logits(expected, actual, *, rtol=LOGIT_RTOL, atol=LOGIT_ATOL,
                   tie_margin=TIE_MARGIN, label="logits"):
    """Check exported logits against the source model's; raise ValueError on a mismatch.

    Logits must match elementwise within rtol/atol. Greedy CTC tokens must match too: a
    frame's argmax may only differ where the source's top two logits are within
    `tie_margin`, since float32 reordering can legitimately flip those.
    """
    from fast_omniasr.decoder import greedy_token_ids

    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: exported shape {actual.shape} != source {expected.shape}")
    if not np.isfinite(actual).all():
        raise ValueError(f"{label}: exported logits contain NaN or infinity")
    difference = np.abs(actual.astype(np.float64) - expected)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"{label}: logits differ by up to {difference.max():.6g} "
                         f"(rtol={rtol}, atol={atol})")
    expected_ids = expected.argmax(-1)
    flipped = expected_ids != actual.argmax(-1)
    top_two = np.sort(expected, axis=-1)[..., -2:]
    near_tie = (top_two[..., 1] - top_two[..., 0]) <= tie_margin
    if (flipped & ~near_tie).any():
        frame = int(np.flatnonzero((flipped & ~near_tie).reshape(-1))[0])
        raise ValueError(f"{label}: greedy token differs at frame {frame}")
    tokens = greedy_token_ids(expected)
    return {
        "input": label,
        "frames": int(expected.shape[1]),
        "tokens": len(tokens),
        "tokens_equal": tokens == greedy_token_ids(actual),
        "near_tie_flips": int(flipped.sum()),
        "max_abs_error": float(difference.max()),
    }


def _session(path, threads):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _feeds(session, audio, lengths):
    """Batch graphs also take each row's length; batch-one graphs take the audio alone."""
    names = [value.name for value in session.get_inputs()]
    if "audio_lengths" in names:
        return {"audio": audio, "audio_lengths": np.asarray(lengths, dtype=np.int64)}
    return {"audio": audio}


def run_onnx(path, inputs, threads=4):
    session = _session(path, threads)
    return {name: session.run(["logits"], _feeds(session, audio, [audio.shape[1]]))[0]
            for name, audio in inputs.items()}


def run_onnx_batch(path, inputs, threads=4):
    """Run every input in one zero-padded batch; each row's logits cut to its frame count."""
    from fast_omniasr.audio import pad_batch

    session = _session(path, threads)
    audio, lengths = pad_batch(list(inputs.values()))
    logits, frames = session.run(["logits", "logit_lengths"], _feeds(session, audio, lengths))
    return {name: logits[row:row + 1, :frames[row]] for row, name in enumerate(inputs)}


def validate_export(path, references, inputs, *, rtol=LOGIT_RTOL, atol=LOGIT_ATOL,
                    run=run_onnx, run_batch=None):
    """Run the exported graph on every input and compare with the source logits. With
    `run_batch` (batch graphs), every input is also run in one padded batch, and each
    row must match the source's logits for that input alone."""
    outputs = run(path, inputs)
    probes = [compare_logits(references[name], outputs[name], rtol=rtol, atol=atol, label=name)
              for name in inputs]
    report = {"rtol": rtol, "atol": atol, "tie_margin": TIE_MARGIN, "probes": probes}
    if run_batch is not None:
        batched = run_batch(path, inputs)
        label = f"in a padded batch of {len(inputs)}"
        report["batch_probes"] = [
            compare_logits(references[name], batched[name], rtol=rtol, atol=atol,
                           label=f"{name} {label}")
            for name in inputs
        ]
    return report


def frame_lengths(lengths, layer_descs):
    """Frames the convolutional feature extractor makes from `lengths` samples: one per
    layer stride, after each layer's kernel. Works on ints, numpy arrays and torch
    tensors (traced into the graph); lengths below the receptive field give <= 0."""
    for desc in layer_descs:
        kernel, stride = desc[1], desc[2]
        lengths = (lengths - kernel) // stride + 1
    return lengths


def batch_forward(model, torch):
    """Wrap a wav2vec2 CTC model to run a zero-padded batch.

    Takes audio [batch, samples] and each row's valid length, and returns logits
    [batch, frames, vocabulary] and each row's frame count. A row's valid frames match
    running that row alone: the feature extractor's convolutions and per-frame
    normalization never look past a frame's own samples, the padded frames are zeroed
    before the positional convolution (as the batch-one graph's own zero padding does),
    and self-attention ignores padded keys. Frames past a row's count hold padding.

    Mutates `model`: each encoder layer's attention is replaced by a masked one and the
    positional encoder gets a hook, so run the batch-one reference before calling this.
    """
    from fairseq2.models.transformer.attention_bias import IdentityBias
    from fairseq2.models.transformer.sdpa.torch import TorchSDPA
    from fairseq2.nn import BatchLayout
    from torch.nn.functional import scaled_dot_product_attention

    frontend = model.encoder_frontend
    extractor = frontend.feature_extractor
    if any(isinstance(module, torch.nn.GroupNorm) for module in extractor.modules()):
        # Group normalization spans the time axis, so padding would change every frame.
        raise ValueError("Batch export needs a feature extractor with per-frame "
                         "normalization; this model uses group normalization")
    if frontend.pos_encoder is None:
        raise ValueError("Batch export expects a convolutional positional encoder")
    state = {}

    class KeyPaddingSDPA(torch.nn.Module):
        """Full self-attention over each row's valid frames only."""

        def forward(self, q, q_layout, k, k_layout, v, bias_cache, *, needs_weights=False):
            q, k, v = (tensor.transpose(-2, -3) for tensor in (q, k, v))
            attns = scaled_dot_product_attention(q, k, v, attn_mask=state["keys"])
            return attns.transpose(-2, -3), None

    for layer in model.encoder.layers:
        sdpa = layer.self_attn.sdpa
        if not isinstance(sdpa, TorchSDPA) or not isinstance(sdpa.bias, IdentityBias):
            raise TypeError(f"Batch export expects unbiased PyTorch attention, got {sdpa}")
        layer.self_attn.sdpa = KeyPaddingSDPA()

    def zero_padding(module, args):
        seqs = args[0]
        valid = torch.arange(seqs.shape[1], device=seqs.device)[None, :] < state["frames"][:, None]
        state["keys"] = valid[:, None, None, :]  # (N, 1, 1, S_kv), broadcast over heads
        return (seqs * valid.unsqueeze(-1).to(seqs.dtype), *args[1:])

    frontend.pos_encoder.register_forward_pre_hook(zero_padding)

    class BatchForward(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, audio, audio_lengths):
            state["frames"] = frame_lengths(audio_lengths, extractor.layer_descs)
            layout = BatchLayout(audio.shape, seq_lens=None, device=audio.device)
            return self.model(audio, layout)[0], state["frames"]

    return BatchForward().eval()


def promote(staging: Path, output: Path):
    """Move a validated export into place, the graph last so it never appears half-written."""
    for path in sorted(staging.iterdir()):
        if path.name != output.name:
            path.replace(output.parent / path.name)
    (staging / output.name).replace(output)


def graph_files(model_path: Path) -> list[Path]:
    """The graph and its ONNX external-data files."""
    from tools.publish_hub import external_data_files

    return [model_path, *external_data_files(model_path)]


def build_provenance(args, *, weights_sha256, model_path, files, validation) -> dict:
    """The provenance.json record for an export staged at `model_path`."""
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = {
            "name": args.checkpoint.name,
            "sha256": provenance.path_sha256(args.checkpoint),
        }
    family = tokenizer_family(args.model)
    root = model_path.parent
    tokenizer_path = root / "tokenizer.model"
    return {
        "exporter": provenance.exporter("export_onnx.py"),
        "environment": provenance.environment(),
        "source": {
            "model_card": args.model,
            "checkpoint": checkpoint,
            "weights_sha256": weights_sha256,
        },
        "tokenizer": {
            "file": tokenizer_path.name,
            "family": family,
            "url": TOKENIZER_URLS[family],
            "sha256": provenance.file_sha256(tokenizer_path),
        },
        "files": {
            path.relative_to(root).as_posix(): {
                "sha256": provenance.file_sha256(path),
                "size": path.stat().st_size,
            }
            for path in files
        },
        "validation": validation,
    }


def staged_export(output: Path, build, validate, describe=None):
    """Build into a staging directory beside `output` and promote it only once validated.

    `build(model_path, tokenizer_path)` writes the graph (and any external data next to
    it) and the tokenizer; `validate(model_path)` returns a report or raises;
    `describe(model_path, report)`, if given, returns the provenance record written as
    provenance.json. On failure nothing at `output` changes and the staging directory is
    removed.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        build(staging / output.name, staging / "tokenizer.model")
        report = validate(staging / output.name)
        (staging / PARITY_REPORT).write_text(json.dumps(report, indent=2) + "\n")
        if describe is not None:
            provenance.write(staging, describe(staging / output.name, report))
        promote(staging, output)
        return report
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main():
    args = parse_args()
    import omnilingual_asr  # noqa: F401 -- Registers model cards via the fairseq2 extension.
    import torch
    from fairseq2.models.hub import load_model
    from fairseq2.models.wav2vec2.asr import get_wav2vec2_asr_model_hub
    from fairseq2.nn import BatchLayout

    class Forward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            layout = BatchLayout(audio.shape, seq_lens=None, device=audio.device)
            return self.model(audio, layout)[0]

    torch.set_num_threads(4)
    model = load_source_model(args, torch, load_model, get_wav2vec2_asr_model_hub)
    wrapper = Forward(model).eval()
    weights_sha256 = provenance.weights_sha256(
        (name, tensor.detach().cpu().numpy()) for name, tensor in model.state_dict().items()
    )
    inputs = validation_inputs(args.validation_audio or VALIDATION_AUDIO)
    # Source logits are computed up front so the PyTorch model can be freed before ONNX
    # Runtime loads the export; large models do not fit in memory twice.
    with torch.no_grad():
        references = {name: wrapper(torch.from_numpy(audio)).numpy()
                      for name, audio in inputs.items()}

    def build(model_path, tokenizer_path):
        prepare_tokenizer(args.model, tokenizer_path, model.final_proj.output_dim)
        with torch.no_grad():
            if args.batch:
                # Two rows of different lengths, so nothing specializes on one length.
                torch.onnx.export(
                    batch_forward(model, torch),
                    (torch.zeros(2, 80000), torch.tensor([80000, 48000])), str(model_path),
                    input_names=["audio", "audio_lengths"],
                    output_names=["logits", "logit_lengths"], opset_version=18, dynamo=False,
                    dynamic_axes={"audio": {0: "batch", 1: "samples"},
                                  "audio_lengths": {0: "batch"},
                                  "logits": {0: "batch", 1: "frames"},
                                  "logit_lengths": {0: "batch"}},
                )
            else:
                torch.onnx.export(wrapper, (torch.zeros(1, 80000),), str(model_path),
                                  input_names=["audio"], output_names=["logits"],
                                  opset_version=18, dynamo=False,
                                  dynamic_axes={"audio": {1: "samples"},
                                                "logits": {1: "frames"}})
        # Compute the positional convolution's weight norm once in float64 instead of on
        # every call in float32 (see tools/onnx_fold.py).
        from tools.onnx_fold import fold_file

        if fold_file(model_path) != 1:
            raise ValueError("Expected exactly one weight-normalized positional convolution")
        # Tie the graph to its tokenizer; OmniASR rejects a different one on load.
        from tools.stamp_tokenizer import stamp

        stamp(model_path, tokenizer_path)

    def validate(model_path):
        nonlocal model, wrapper
        del model, wrapper
        gc.collect()
        return validate_export(model_path, references, inputs,
                               rtol=args.logit_rtol, atol=args.logit_atol,
                               run_batch=run_onnx_batch if args.batch else None)

    def describe(model_path, report):
        return build_provenance(args, weights_sha256=weights_sha256, model_path=model_path,
                                files=graph_files(model_path), validation=report)

    report = staged_export(args.output, build, validate, describe)
    for probe in report["probes"] + report.get("batch_probes", []):
        print(json.dumps(probe))
    print(args.output)
    print(args.output.parent / "tokenizer.model")
    print(args.output.parent / provenance.PROVENANCE_FILE)


if __name__ == "__main__":
    main()

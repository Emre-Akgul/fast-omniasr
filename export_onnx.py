"""One-time direct fairseq2 export; these dependencies are not runtime requirements."""
import argparse
import gc
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Make the repository's tools importable when this file is loaded by path (as the tests do).
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
        help=(
            "Clip for the recorded fairseq2 vs ONNX Runtime parity check "
            "(default: 5 seconds of seeded noise)"
        ),
    )
    args = parser.parse_args(argv)
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


VALIDATION_SEED = 0
VALIDATION_SAMPLES = 80000


def validation_input(path=None):
    """Return the parity-check waveform, shaped (1, samples), and a description of it."""
    import numpy as np

    if path is None:
        rng = np.random.default_rng(VALIDATION_SEED)
        audio = 0.1 * rng.standard_normal((1, VALIDATION_SAMPLES), dtype=np.float32)
        source = {"source": "seeded noise", "seed": VALIDATION_SEED, "samples": audio.shape[1]}
        return audio, source
    from fast_omniasr.audio import load_audio

    audio = np.ascontiguousarray(load_audio(path), dtype=np.float32).reshape(1, -1)
    source = {
        "source": "file",
        "name": path.name,
        "sha256": provenance.file_sha256(path),
        "samples": audio.shape[1],
    }
    return audio, source


def compare_logits(reference, actual) -> dict:
    """Summarize how closely the exported graph's logits match the source model's."""
    import numpy as np

    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(
            f"The exported graph returned logits of shape {actual.shape}, "
            f"but the source model returned {reference.shape}"
        )
    if not (np.isfinite(reference).all() and np.isfinite(actual).all()):
        raise ValueError("Non-finite logits in the export parity check")
    error = np.abs(reference - actual)
    return {
        "frames": int(reference.shape[1]),
        "vocabulary_size": int(reference.shape[-1]),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "argmax_agreement": float((reference.argmax(-1) == actual.argmax(-1)).mean()),
    }


def onnx_runtime_logits(model_path: Path, audio):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(["logits"], {"audio": audio})[0]


def graph_files(model_path: Path) -> list[Path]:
    """The graph and its ONNX external-data files."""
    from tools.publish_hub import external_data_files

    return [model_path, *external_data_files(model_path)]


def build_provenance(args, *, weights_sha256, tokenizer_path, files, validation) -> dict:
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = {
            "name": args.checkpoint.name,
            "sha256": provenance.path_sha256(args.checkpoint),
        }
    family = tokenizer_family(args.model)
    root = args.output.parent
    return {
        "exporter": provenance.exporter("export_onnx.py"),
        "environment": provenance.environment(),
        "source": {
            "model_card": args.model,
            "checkpoint": checkpoint,
            "weights_sha256": weights_sha256,
        },
        "tokenizer": {
            "file": tokenizer_path.relative_to(root).as_posix(),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = prepare_tokenizer(
        args.model,
        args.output.parent / "tokenizer.model",
        model.final_proj.output_dim,
    )
    weights_sha256 = provenance.weights_sha256(
        (name, tensor.detach().cpu().numpy()) for name, tensor in model.state_dict().items()
    )
    audio, validation_source = validation_input(args.validation_audio)
    with torch.no_grad():
        reference_logits = wrapper(torch.from_numpy(audio)).numpy()
        torch.onnx.export(wrapper, (torch.zeros(1, 80000),), str(args.output),
                          input_names=["audio"], output_names=["logits"], opset_version=18,
                          dynamo=False, dynamic_axes={"audio": {1: "samples"}, "logits": {1: "frames"}})
    # Compute the positional convolution's weight norm once in float64 instead of on every
    # call in float32 (see tools/onnx_fold.py).
    from tools.onnx_fold import fold_file

    if fold_file(args.output) != 1:
        raise ValueError("Expected exactly one weight-normalized positional convolution")
    # Tie the graph to its tokenizer; OmniASR rejects a different one on load.
    from tools.stamp_tokenizer import stamp

    stamp(args.output, tokenizer_path)
    # Check the final graph against the source model, then record the result with the
    # environment and hashes it came from. The source model is released first so a large
    # model is not held in memory twice.
    del model, wrapper
    gc.collect()
    validation = {
        "runtime": "onnxruntime-cpu",
        "input": validation_source,
        **compare_logits(reference_logits, onnx_runtime_logits(args.output, audio)),
    }
    record = build_provenance(
        args,
        weights_sha256=weights_sha256,
        tokenizer_path=tokenizer_path,
        files=graph_files(args.output),
        validation=validation,
    )
    provenance_path = provenance.write(args.output.parent, record)
    print(args.output)
    print(tokenizer_path)
    print(provenance_path)
    print(
        f"Parity vs fairseq2: max abs error {validation['max_abs_error']:.3g}, "
        f"argmax agreement {validation['argmax_agreement']:.4f}"
    )


if __name__ == "__main__":
    main()

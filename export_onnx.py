"""One-time direct fairseq2 export; these dependencies are not runtime requirements."""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

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
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.zeros(1, 80000),), str(args.output),
                          input_names=["audio"], output_names=["logits"], opset_version=18,
                          dynamo=False, dynamic_axes={"audio": {1: "samples"}, "logits": {1: "frames"}})
    print(args.output)
    print(tokenizer_path)


if __name__ == "__main__":
    main()

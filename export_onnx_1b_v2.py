"""Export the official omniASR_CTC_1B_v2 base model to ONNX.

This venv's omnilingual-asr (0.1.0, PyPI) only registers the v1 "1b" card
(target_vocab_size=9812, checkpoint omniASR-CTC-1B.pt). The v2 checkpoint uses
the same architecture but a larger (10288-token) vocabulary and is not
registered as a card in this package version, so we build the model from an
overridden config (as with the custom Turkish checkpoints) and download the
v2 checkpoint directly from the same host used for the 300M v2 model.
"""
import argparse
from pathlib import Path
from urllib.request import urlretrieve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import omnilingual_asr  # noqa: F401 -- Registers model cards via the fairseq2 extension.
    import torch
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
    hub = get_wav2vec2_asr_model_hub()
    config = hub.get_arch_config("1b")
    config.target_vocab_size = 10288  # v2 tokenizer vocab size
    model = hub.create_new_model(config, device=torch.device("cpu"), dtype=torch.float32)

    if not args.checkpoint_cache.exists():
        print("Downloading omniASR-CTC-1B-v2.pt (~3.9GB)...")
        args.checkpoint_cache.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve("https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B-v2.pt", args.checkpoint_cache)

    state_dict = torch.load(args.checkpoint_cache, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("missing keys:", missing)
    print("unexpected keys:", unexpected)

    wrapper = Forward(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.zeros(1, 80000),), str(args.output),
                          input_names=["audio"], output_names=["logits"], opset_version=18,
                          dynamo=False, dynamic_axes={"audio": {1: "samples"}, "logits": {1: "frames"}})
    print(args.output)


if __name__ == "__main__":
    main()

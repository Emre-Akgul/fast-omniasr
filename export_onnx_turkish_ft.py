"""Export omniASR-ctc-300m-turkish-step6000 (fine-tuned checkpoint) to ONNX.

Loads the base omniASR_CTC_300M_v2 architecture via fairseq2, then overwrites
its weights with the fine-tuned state dict before exporting -- same export
recipe as fast-omniasr/export_onnx.py.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    config = hub.get_arch_config("300m")
    config.target_vocab_size = 10288  # matches the fine-tuned checkpoint's tokenizer (v2, not the v1 card's 9812)
    model = hub.create_new_model(config, device=torch.device("cpu"), dtype=torch.float32)

    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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

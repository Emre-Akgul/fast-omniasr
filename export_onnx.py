"""One-time direct fairseq2 export; these dependencies are not runtime requirements."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import omnilingual_asr  # noqa: F401 -- Registers model cards via the fairseq2 extension.
    import torch
    from fairseq2.models.hub import load_model
    from fairseq2.nn import BatchLayout

    class Forward(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            layout = BatchLayout(audio.shape, seq_lens=None, device=audio.device)
            return self.model(audio, layout)[0]

    torch.set_num_threads(4)
    model = load_model("omniASR_CTC_300M_v2", device=torch.device("cpu"), dtype=torch.float32)
    wrapper = Forward(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (torch.zeros(1, 80000),), str(args.output),
                          input_names=["audio"], output_names=["logits"], opset_version=18,
                          dynamo=False, dynamic_axes={"audio": {1: "samples"}, "logits": {1: "frames"}})
    print(args.output)


if __name__ == "__main__":
    main()

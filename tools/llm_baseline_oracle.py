"""Materialize full-prefix logits in a reference-free process to bound validation RAM."""

import argparse
import json
import sys
from pathlib import Path

from validate_llm import BlockReferenceImports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("audio", type=Path, nargs="+")
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockReferenceImports())
    import torch

    from fast_omniasr.audio import load_audio

    torch.set_num_threads(4)
    config = json.loads((args.directory / "config.json").read_text())
    encoder = torch.jit.load(str(args.directory / "encoder.pt")).eval()
    with torch.inference_mode():
        contexts = []
        for clip in args.audio:
            waveform = load_audio(clip)
            if waveform.shape[1] > 480000:
                raise ValueError("Clips must be <=30 seconds")
            contexts.append(encoder(torch.from_numpy(waveform)))
        del encoder
        decoder = torch.jit.load(str(args.directory / "decoder.pt")).eval()
        for i, (clip, context) in enumerate(zip(args.audio, contexts)):
            ids = [config["bos_idx"]]
            logits = []
            for step in range(min(512, config["max_seq_len"] - context.shape[1] - 3)):
                out = decoder(context, torch.tensor([ids]))
                assert torch.isfinite(out).all()
                logits.append(out)
                token = int(out.argmax(-1))
                if token == config["eos_idx"]:
                    break
                ids.append(token)
                if (step + 1) % 10 == 0:
                    print(f"Full-prefix oracle: {clip.name}: step {step + 1}", flush=True)
            else:
                raise AssertionError(f"No EOS for {clip}")
            torch.save({"context": context, "logits": torch.stack(logits)}, args.output / f"{i}.pt")
            print(f"Full-prefix oracle: {clip.name}: {len(logits)} steps including EOS", flush=True)


if __name__ == "__main__":
    main()

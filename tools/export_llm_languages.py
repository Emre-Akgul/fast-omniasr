"""Add optional language conditioning assets to an exported LLM directory.

Writes the model's learned language embeddings as `lang_embeddings.npy`
([languages + 1, 4096] float32; row 0 is "no language") and the upstream
code-to-row mapping as `languages.json`, then records both in config.json.
The exported encoder already ends with row 0; the runtime swaps in another row
when transcribe(lang=...) is used. fairseq2 is used only here.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    import omnilingual_asr  # noqa: F401
    import torch
    from fairseq2.models.hub import load_model
    from omnilingual_asr.models.wav2vec2_llama.hub import get_wav2vec2_llama_model_hub

    config = json.loads((args.directory / "config.json").read_text())
    if args.checkpoint:
        hub = get_wav2vec2_llama_model_hub()
        model = hub.load_custom_model(
            args.checkpoint,
            hub.get_model_config(config["model"]),
            device=torch.device("cpu"),
            dtype=torch.float32,
            mmap=True,
        )
    else:
        model = load_model(config["model"], device=torch.device("cpu"), dtype=torch.float32)
    if model.lang_embeddings is None or not model.lang_mapping:
        raise ValueError(f"{config['model']} has no language embeddings")
    table = model.lang_embeddings.weight.detach().numpy().astype(np.float32)
    mapping = dict(sorted(model.lang_mapping.items()))
    if table.shape[0] != len(mapping) + 1 or sorted(mapping.values()) != list(
        range(1, table.shape[0])
    ):
        raise ValueError("Language mapping does not index every embedding row")
    np.save(args.directory / "lang_embeddings.npy", table)
    (args.directory / "languages.json").write_text(json.dumps(mapping, indent=0) + "\n")
    config["languages"] = {
        "embeddings": "lang_embeddings.npy",
        "mapping": "languages.json",
        "count": len(mapping),
    }
    (args.directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {len(mapping)} languages, embeddings {list(table.shape)}")


if __name__ == "__main__":
    main()

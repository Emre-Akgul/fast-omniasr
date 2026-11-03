"""Record which tokenizer an exported CTC graph belongs to.

Writes the tokenizer's sha256 and vocabulary size into the ONNX graph's metadata_props
(`fast_omniasr.tokenizer_sha256`, `fast_omniasr.vocabulary_size`). OmniASR checks them on
load, so a graph paired with a different tokenizer fails instead of decoding garbage.
export_onnx.py stamps new exports, and tools/convert_precision.py keeps the metadata in
fp16/bf16 variants. Existing exports can be stamped in place; external weight files are
left untouched. The graph file changes, so a published repo must update its sha256 in
config.json (tools/publish_hub.py does so when republishing):

  python tools/stamp_tokenizer.py export/model.onnx export/fp16/model.onnx \\
      --tokenizer export/tokenizer.model
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fast_omniasr.tokenizer import (
    TOKENIZER_SHA256_KEY,
    VOCABULARY_SIZE_KEY,
    Tokenizer,
)


def stamp(model_path, tokenizer_path):
    """Write the tokenizer identity into the graph at `model_path` in place."""
    import onnx

    model_path = Path(model_path)
    tokenizer = Tokenizer(tokenizer_path)
    graph = onnx.load(str(model_path), load_external_data=False)
    outputs = graph.graph.output
    dims = outputs[0].type.tensor_type.shape.dim if len(outputs) == 1 else []
    if dims and dims[-1].HasField("dim_value") and dims[-1].dim_value != tokenizer.vocabulary_size:
        raise ValueError(f"{model_path} outputs {dims[-1].dim_value} classes but the tokenizer "
                         f"has {tokenizer.vocabulary_size} entries")
    values = {TOKENIZER_SHA256_KEY: hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
              VOCABULARY_SIZE_KEY: str(tokenizer.vocabulary_size)}
    kept = [entry for entry in graph.metadata_props if entry.key not in values]
    del graph.metadata_props[:]
    graph.metadata_props.extend(kept)
    for key, value in values.items():
        graph.metadata_props.add(key=key, value=value)
    # Only the graph proto is rewritten; external tensors keep pointing at their files.
    temporary = model_path.with_name(f".{model_path.name}.stamp")
    try:
        onnx.save_model(graph, str(temporary))
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("graphs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    for graph in args.graphs:
        values = stamp(graph, args.tokenizer)
        print(f"{graph}: tokenizer sha256 {values[TOKENIZER_SHA256_KEY][:12]}…, "
              f"{values[VOCABULARY_SIZE_KEY]} entries")


if __name__ == "__main__":
    main()

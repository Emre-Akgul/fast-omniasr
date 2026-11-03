import hashlib
from pathlib import Path

import sentencepiece as spm

# ONNX metadata_props keys written by the exporter (tools/stamp_tokenizer.py) to tie a
# graph to the exact tokenizer it was exported with.
TOKENIZER_SHA256_KEY = "fast_omniasr.tokenizer_sha256"
VOCABULARY_SIZE_KEY = "fast_omniasr.vocabulary_size"


class Tokenizer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._processor = spm.SentencePieceProcessor(model_file=str(path))

    @property
    def vocabulary_size(self) -> int:
        return self._processor.get_piece_size()

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def decode(self, ids: list[int]) -> str:
        return self._processor.decode(ids)


def check_compatible(tokenizer: Tokenizer, *, vocabulary_size: int | None = None,
                     graph_path: str | Path | None = None) -> None:
    """Raise ValueError unless `tokenizer` fits the model.

    The model's output vocabulary (when known) must equal the tokenizer's size. A matching
    size alone doesn't prove the pair belongs together, so when the ONNX graph at
    `graph_path` records its tokenizer's sha256 (export_onnx.py, tools/stamp_tokenizer.py), the file
    must match it exactly. Graphs without that metadata get the size check only."""
    if vocabulary_size is not None and vocabulary_size != tokenizer.vocabulary_size:
        raise ValueError(
            f"The model outputs {vocabulary_size} classes but the tokenizer "
            f"{tokenizer.path} has {tokenizer.vocabulary_size} entries; use the "
            "tokenizer.model exported with this graph"
        )
    if graph_path is None:
        return
    from ._onnx_deps import model_metadata

    metadata = model_metadata(graph_path)
    recorded_size = metadata.get(VOCABULARY_SIZE_KEY)
    if recorded_size is not None and recorded_size != str(tokenizer.vocabulary_size):
        raise ValueError(
            f"{graph_path} was exported for a {recorded_size}-entry tokenizer, "
            f"but {tokenizer.path} has {tokenizer.vocabulary_size} entries"
        )
    expected = metadata.get(TOKENIZER_SHA256_KEY)
    if expected is not None and expected != tokenizer.sha256():
        raise ValueError(
            f"{tokenizer.path} is not the tokenizer {graph_path} was exported with: the "
            f"graph records the exact file's sha256 {expected[:12]}…; use the tokenizer.model "
            "exported with this graph"
        )

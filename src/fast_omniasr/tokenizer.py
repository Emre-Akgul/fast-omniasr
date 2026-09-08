from pathlib import Path

import sentencepiece as spm


class Tokenizer:
    def __init__(self, path: str | Path):
        self._processor = spm.SentencePieceProcessor(model_file=str(path))

    def decode(self, ids: list[int]) -> str:
        return self._processor.decode(ids)

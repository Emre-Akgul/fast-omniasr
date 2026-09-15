"""Validate exported LLM generation in a process that forbids fairseq2 imports."""

import argparse
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class BlockReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fairseq2", "fairseq2n", "omnilingual_asr"}:
            raise ImportError(
                f"Reference dependency forbidden during standalone validation: {fullname}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockReferenceImports())
    import torch

    from fast_omniasr.audio import load_audio
    from fast_omniasr.llm import OmniASRLLM

    torch.set_num_threads(4)
    reference = json.loads((args.directory / "reference.json").read_text())
    references = [reference]
    if args.cached:
        suite = json.loads((args.directory / "cached_reference.json").read_text())
        references.extend(suite["clips"][1:])
    model = OmniASRLLM(args.directory, cached=args.cached)
    results = []
    for index, expected in enumerate(references):
        audio = (
            args.audio
            if index == 0 and args.audio is not None
            else args.directory / expected["audio"]
        )
        digest = hashlib.sha256(load_audio(audio).tobytes()).hexdigest()
        if digest != expected["normalized_audio_sha256"]:
            raise AssertionError("Audio differs from the reference parity fixture")
        result = model.transcribe(audio, max_new_tokens=reference["max_new_tokens"])
        for field in ["text", "token_ids", "stop_reason"]:
            if field in expected and getattr(result, field) != expected[field]:
                raise AssertionError(f"Standalone {field} differs: {getattr(result, field)!r}")
        if result.stop_reason != "eos":
            raise AssertionError("Matching prefix only: generation did not reach EOS")
        results.append(
            {
                "audio": str(audio),
                "text": result.text,
                "token_ids": result.token_ids,
                "stop_reason": result.stop_reason,
                "reference_imports_blocked": True,
            }
        )
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()

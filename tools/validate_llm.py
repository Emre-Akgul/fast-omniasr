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
    args = parser.parse_args()
    sys.meta_path.insert(0, BlockReferenceImports())
    import torch

    from fast_omniasr.audio import load_audio
    from fast_omniasr.llm import OmniASRLLM

    torch.set_num_threads(4)
    reference = json.loads((args.directory / "reference.json").read_text())
    audio = args.audio if args.audio is not None else args.directory / reference["audio"]
    digest = hashlib.sha256(load_audio(audio).tobytes()).hexdigest()
    if digest != reference["normalized_audio_sha256"]:
        raise AssertionError("Audio differs from the reference parity fixture")
    model = OmniASRLLM(args.directory)
    result = model.transcribe(audio, max_new_tokens=reference["max_new_tokens"])
    for field in ["text", "token_ids", "stop_reason"]:
        if getattr(result, field) != reference[field]:
            raise AssertionError(f"Standalone {field} differs: {getattr(result, field)!r}")
    if result.stop_reason != "eos":
        raise AssertionError("Matching prefix only: generation did not reach EOS")
    print(
        json.dumps(
            {
                "text": result.text,
                "token_ids": result.token_ids,
                "stop_reason": result.stop_reason,
                "reference_imports_blocked": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

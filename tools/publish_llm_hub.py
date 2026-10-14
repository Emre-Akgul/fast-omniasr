"""Prepare and optionally upload a validated ONNX LLM-ASR export to Hugging Face Hub.

The directory must hold the Day-3 ONNX layout (export_llm_onnx.py) and a passing
fairseq2 parity report (tools/validate_llm_fairseq2.py compare --report). The
published config.json is the runtime config plus a sha256 `files` manifest that
OmniASRLLM.from_pretrained() verifies; parity reports are uploaded under parity/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from publish_hub import external_data_files, sha256

from fast_omniasr.llm.models import LLM_MODEL_CARDS, LLM_MODEL_TOKENIZERS, repo_name

RUNTIME_FILES = [
    "encoder.onnx",
    "decoder_prefill.onnx",
    "decoder_step.onnx",
    "decoder_weights.json",
    "tokenizer.model",
    "lang_embeddings.npy",
    "languages.json",
]
PARITY_FILES = [
    "encoder_parity.json",
    "decoder_probes_cpu.json",
    "decoder_probes_cuda.json",
    "fairseq2_parity_cpu.json",
    "fairseq2_parity_cuda.json",
]


def prepare_bundle(directory: Path) -> tuple[dict, list[Path], list[Path]]:
    config = json.loads((directory / "config.json").read_text())
    if config.get("format_version") != 1 or config.get("model") not in LLM_MODEL_CARDS:
        raise ValueError("Unsupported LLM artifact format or model")
    if "languages" not in config:
        raise ValueError("Run tools/export_llm_languages.py before publishing")
    for name in ["fairseq2_parity_cpu.json", "fairseq2_parity_cuda.json"]:
        report = json.loads((directory / name).read_text())
        if not report["passed"] or report["model"] != config["model"]:
            raise ValueError(f"{name} does not record a passing {config['model']} comparison")
    assets = [directory / name for name in RUNTIME_FILES]
    for graph in ["encoder.onnx", "decoder_prefill.onnx", "decoder_step.onnx"]:
        assets.extend(external_data_files(directory / graph))
    assets = sorted(set(assets))
    for path in assets:
        if not path.is_file():
            raise FileNotFoundError(path)
    config.pop("files", None)
    config["files"] = {
        str(path.relative_to(directory)): {"sha256": sha256(path), "size": path.stat().st_size}
        for path in assets
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    parity = [directory / name for name in PARITY_FILES if (directory / name).is_file()]
    return config, assets, parity


def model_card(config: dict, repo_id: str, directory: Path) -> str:
    model = config["model"]
    tokenizer = LLM_MODEL_TOKENIZERS[model]
    rows = []
    for device in ["cpu", "cuda"]:
        report = json.loads((directory / f"fairseq2_parity_{device}.json").read_text())
        rows.append(
            f"| ONNX Runtime {device.upper()} | {len(report['clips'])} | {report['total_steps']} "
            f"| {report['max_logit_abs_error']:.3g} | yes |"
        )
    sizes = sum(entry["size"] for entry in config["files"].values()) / 2**30
    return f"""---
license: apache-2.0
pipeline_tag: automatic-speech-recognition
tags:
- speech
- asr
- onnx
- llm-asr
- omnilingual-asr
---

# {repo_name(model)}

An ONNX export of Meta's `{model}` LLM-ASR model (wav2vec2 encoder + LLaMA decoder)
for KV-cached greedy decoding with
[fast-omniasr](https://github.com/Emre-Akgul/fast-omniasr). Inference needs only
NumPy, ONNX Runtime and SentencePiece; PyTorch and fairseq2 are not required.

## Usage

LLM `from_pretrained` support is newer than the latest PyPI release; until the next
release, install from GitHub:

```bash
python -m pip install "fast-omniasr[onnx,hub] @ git+https://github.com/Emre-Akgul/fast-omniasr"
# GPU: replace [onnx,hub] with [cuda,hub]
```

```python
from fast_omniasr.llm import OmniASRLLM

model = OmniASRLLM.from_pretrained("{repo_id}", device="cpu")  # or device="cuda"
result = model.transcribe("speech.wav", lang="eng_Latn")
print(result.text)
print(result.stop_reason)  # "eos" means complete
```

`lang` takes an upstream language code (`eng_Latn`, `tur_Latn`, `ckb_Arab`, ...; see
`languages.json` for all {config["languages"]["count"]}). Without it the model's learned
"no language" embedding is used, which upstream warns can degrade transcription quality.

Input must be mono 16 kHz audio of at most 30 seconds. The float32 assets total
about {sizes:.1f} GiB; CPU or GPU memory must hold the encoder and then the decoder.

## Model and conversion

- **Upstream model:** Meta
  [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), `{model}`
- **Conversion:** official fairseq2 checkpoint, float32, no retraining or intentional
  weight modification. The encoder is exported with `torch.onnx.export` (opset 18); the
  decoder is lowered to explicit prefill and single-token step graphs that share
  `decoder.weights`.
- **Tokenizer:** upstream `{tokenizer}`, included unmodified as `tokenizer.model`
- **Graphs:**
  - `encoder.onnx`: `audio [1, samples]` -> `context [1, audio_positions, 4096]`
  - `decoder_prefill.onnx`: `context` + BOS -> logits + 24 K/V tensors
  - `decoder_step.onnx`: one token + 24 K/V tensors -> logits + 24 updated K/V tensors

## Parity with fairseq2

Every decoding step's logits were compared against fairseq2 {config.get("fairseq2_version", "")}
incremental greedy decoding (CPU float32) on English, Turkish and Central Kurdish clips, each
with and without language conditioning, with PyTorch and fairseq2 imports blocked in the ONNX
process. Greedy tokens and text matched at every step through EOS.

| Runtime | Clips | Steps incl. EOS | Max abs logit error | Tokens identical |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

Reports are in `parity/`.

## Limitations

Batch one, clips up to 30 seconds and greedy decoding. Output is not
expected to match upstream's default five-beam search. This is a format conversion: it
does not change the upstream model's capabilities, language coverage or accuracy.
Consult the [upstream repository](https://github.com/facebookresearch/omnilingual-asr)
for model details, training data, evaluation results and responsible-use considerations.

## Repository files

| File | Description |
|---|---|
| `encoder.onnx`, `encoder.onnx.data` | Speech encoder graph and its tensor data |
| `decoder_prefill.onnx`, `decoder_step.onnx` | KV-cache decoder graphs |
| `decoder.weights`, `decoder_weights.json` | Shared decoder weights and their offsets |
| `tokenizer.model` | Matching upstream SentencePiece tokenizer |
| `lang_embeddings.npy`, `languages.json` | Learned language embeddings and code-to-row mapping |
| `config.json` | Runtime config and asset manifest with sizes and SHA-256 hashes |

## License and attribution

The upstream Omnilingual ASR code and model weights are licensed under Apache-2.0. This
repository redistributes a format-converted version under the same license and identifies the
conversion above. It is an independent distribution and is not affiliated with or endorsed by
Meta.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--license", type=Path, required=True, help="Apache-2.0 LICENSE text")
    parser.add_argument("--namespace", default="EmreAkgul")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    config, assets, parity = prepare_bundle(args.directory)
    repo_id = f"{args.namespace}/{repo_name(config['model'])}"
    readme = args.directory / "README.md"
    readme.write_text(model_card(config, repo_id, args.directory))
    print(f"Prepared {repo_id}: {', '.join(path.name for path in assets)}")
    if not args.upload:
        print("Dry run only; pass --upload to create and publish the repository.")
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=str(path.relative_to(args.directory)), path_or_fileobj=path)
        for path in [args.directory / "config.json", *assets]
    ]
    operations += [
        CommitOperationAdd(path_in_repo=f"parity/{path.name}", path_or_fileobj=path)
        for path in parity
    ]
    operations += [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme),
        CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=args.license),
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish {config['model']} ONNX export",
    )
    print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

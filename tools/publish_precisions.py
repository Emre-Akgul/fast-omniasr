"""Publish fp16/bf16 variants (and any refreshed fp32 graph) to an existing Hub repo.

The local directory is the repo's float32 export, as downloaded by from_pretrained, with
`fp16/` and `bf16/` written by tools/convert_precision.py and, for CTC, the positional
convolution folded by tools/onnx_fold.py. The tool rebuilds the config.json `files`
manifest from the graphs on disk, uploads only new or changed files, removes files the
graphs no longer reference, publishes the agreement report under parity/, and adds a
"Precision" section to the model card.

  python tools/publish_precisions.py EmreAkgul/omniASR-LLM-1B-ONNX /path/to/export \\
      --report precision_agreement.json --upload
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from publish_hub import external_data_files, sha256

PRECISIONS = ("fp16", "bf16")
SHARED = ["tokenizer.model", "languages.json", "lang_embeddings.npy", "decoder_weights.json"]


def manifest_files(directory):
    """Every file the repo should list: graphs, their external data, shared assets."""
    directory = Path(directory)
    llm = (directory / "encoder.onnx").is_file()
    graphs = (
        ["encoder.onnx", "decoder.onnx", "decoder_prefill.onnx", "decoder_step.onnx"]
        if llm else ["model.onnx"]
    )
    variant_graphs = ["encoder.onnx", "decoder.onnx"] if llm else ["model.onnx"]
    files = set()
    for graph in graphs + [f"{p}/{g}" for p in PRECISIONS for g in variant_graphs]:
        path = directory / graph
        if path.is_file():
            files.add(path)
            files.update(external_data_files(path))
    files.update(directory / name for name in SHARED if (directory / name).is_file())
    for precision in PRECISIONS:
        if not all((directory / precision / g).is_file() for g in variant_graphs):
            raise FileNotFoundError(f"{directory / precision} is missing {variant_graphs}")
    return llm, sorted(files)


def precision_section(llm, repo_id, report):
    results = report["precisions"]
    runs = results["fp32"]["inputs"]
    gib = {p: results[p]["gpu_mib"] / 1024 for p in results}
    agreement = {p: f"{results[p]['identical_to_fp32']}/{runs} identical" for p in PRECISIONS}
    rows = [
        f"| `fp32` (default) | CPU, CUDA | top level | {gib['fp32']:.1f} GiB | reference |",
        f"| `fp16` | CPU, CUDA | `fp16/` | {gib['fp16']:.1f} GiB | {agreement['fp16']} |",
        f"| `bf16` | CUDA (Ampere or newer) | `bf16/` | {gib['bf16']:.1f} GiB | {agreement['bf16']} |",
    ]
    if llm:
        usage = (
            "from fast_omniasr.llm import OmniASRLLM\n\n"
            f'model = OmniASRLLM.from_pretrained("{repo_id}", device="cuda", precision="fp16")\n'
            'print(model.transcribe("speech.wav", lang="eng_Latn").text)'
        )
        inputs = "English, Turkish and Central Kurdish clips, each with and without a language code"
    else:
        usage = (
            "from fast_omniasr import OmniASR\n\n"
            f'model = OmniASR.from_pretrained("{repo_id}", device="cuda", precision="fp16")\n'
            'print(model.transcribe("speech.wav").text)'
        )
        inputs = "English, Turkish and Central Kurdish clips"
    return f"""## Precision

`from_pretrained(..., precision=...)` downloads only the requested variant (plus the shared
tokenizer) and needs fast-omniasr 0.3.0 or newer.

| `precision` | Devices | Files | GPU memory (resident, CUDA) | Transcripts vs fp32 |
|---|---|---|---:|---:|
{chr(10).join(rows)}

```python
{usage}
```

fp16 and bf16 store the MatMul weights in half precision, roughly halving the download and
the weight memory. Convolutions, normalization, softmax and all inputs and outputs stay
float32. On CPU fp16 saves memory but is not faster; ONNX Runtime has no CPU bf16 kernels.
Agreement was measured on {inputs}
(`parity/precision_agreement.json`); half precision can change individual words where fp32
itself is close to a tie, as upstream's bf16 default does.
"""


def update_model_card(text, section):
    if "## Precision" in text:
        return re.sub(r"## Precision\n.*?(?=\n## |\Z)", section.rstrip() + "\n", text, flags=re.DOTALL)
    anchor = text.find("## Repository files")
    if anchor == -1:
        anchor = text.find("## License")
    return text + "\n" + section if anchor == -1 else text[:anchor] + section + "\n" + text[anchor:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("repo_id")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path, required=True, help="tools/evaluate_precisions.py report")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    config_path = args.directory / "config.json"
    config = json.loads(config_path.read_text())
    previous = config.get("files", {})
    llm, files = manifest_files(args.directory)
    manifest = {
        str(path.relative_to(args.directory)): {"sha256": sha256(path), "size": path.stat().st_size}
        for path in files
    }
    changed = [name for name, entry in manifest.items() if previous.get(name) != entry]
    removed = sorted(previous.keys() - manifest.keys())
    config["files"] = manifest
    config["precisions"] = ["fp32", *PRECISIONS]
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    report = json.loads(args.report.read_text())
    card = Path(hf_hub_download(args.repo_id, "README.md")).read_text()
    readme = args.directory / "README.precision.md"
    readme.write_text(update_model_card(card, precision_section(llm, args.repo_id, report)))
    print(f"{args.repo_id}: {len(changed)} new or changed files, {len(removed)} removed")
    if not args.upload:
        print("Dry run; pass --upload to commit.")
        return
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=args.directory / name)
                  for name in changed]
    operations += [CommitOperationDelete(path_in_repo=name) for name in removed]
    operations += [
        CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=config_path),
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme),
        CommitOperationAdd(path_in_repo="parity/precision_agreement.json", path_or_fileobj=args.report),
    ]
    HfApi().create_commit(
        repo_id=args.repo_id, repo_type="model", operations=operations,
        commit_message="Add fp16 and bf16 variants",
    )
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

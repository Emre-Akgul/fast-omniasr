"""Prepare and optionally upload a converted CTC model to Hugging Face Hub."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

# This is a repository tool, not an installed runtime module. Make the repository root
# importable when invoked directly as `python tools/publish_hub.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_onnx import CTC_MODEL_CARDS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_data_files(model_path: Path) -> list[Path]:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("Install onnx in the export environment") from exc

    model = onnx.load(str(model_path), load_external_data=False)
    paths = set()
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        metadata = {item.key: item.value for item in tensor.external_data}
        location = metadata.get("location")
        if not location:
            raise ValueError(f"External tensor {tensor.name!r} has no location")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe ONNX external-data location: {location}")
        paths.add(model_path.parent / relative)
    return sorted(paths)


def repo_name(model_card: str) -> str:
    suffix = model_card.removeprefix("omniASR_CTC_").replace("_v2", "-v2")
    return f"omniASR-CTC-{suffix}-ONNX"


def prepare_bundle(model_card: str, model_path: Path, tokenizer_path: Path) -> list[Path]:
    if model_path.name != "model.onnx":
        raise ValueError("Export to a file named model.onnx so external-data references stay valid")
    assets = [model_path, *external_data_files(model_path)]
    for path in assets + [tokenizer_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    bundled_tokenizer = model_path.parent / "tokenizer.model"
    if tokenizer_path.resolve() != bundled_tokenizer.resolve():
        shutil.copy2(tokenizer_path, bundled_tokenizer)
    assets.append(bundled_tokenizer)
    config = {
        "format": 1,
        "model_card": model_card,
        "files": {
            str(path.relative_to(model_path.parent)): {
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }
            for path in assets
        },
    }
    config_path = model_path.parent / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return [config_path, *assets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=CTC_MODEL_CARDS)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--namespace", default="EmreAkgul")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    files = prepare_bundle(args.model, args.onnx, args.tokenizer)
    repo_id = f"{args.namespace}/{repo_name(args.model)}"
    print(f"Prepared {repo_id}: {', '.join(path.name for path in files)}")
    if not args.upload:
        print("Dry run only; pass --upload to create and publish the repository.")
        return

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[hub]") from exc
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    operations = [
        CommitOperationAdd(
            path_in_repo=str(path.relative_to(args.onnx.parent)), path_or_fileobj=path
        )
        for path in files
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish {args.model} ONNX export",
    )
    print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

"""Provenance records for exported model artifacts.

export_onnx.py writes `provenance.json` beside each graph it exports: the exporter's own
revision, the export environment's package versions (with the VCS commit for packages
installed from a git checkout, such as an unreleased omnilingual-asr), the checkpoint,
weight and tokenizer hashes, the hashes of the files written, and the parity check of the
exported graph against the source model. The record describes how the artifacts were
made; it is not read by the runtime.
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

PROVENANCE_FORMAT = 1
PROVENANCE_FILE = "provenance.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Packages that determine what an export contains; the pinned, tested set is
# constraints/export.txt.
EXPORT_PACKAGES = (
    "omnilingual-asr",
    "fairseq2",
    "fairseq2n",
    "torch",
    "torchaudio",
    "onnx",
    "onnxruntime",
    "numpy",
    "sentencepiece",
    "fast-omniasr",
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path) -> str:
    """Hash a file, or a directory as its sorted relative paths and file hashes."""
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(f"{relative}\0{file_sha256(item)}\n".encode())
    return digest.hexdigest()


def weights_sha256(named_arrays) -> str:
    """Hash named numpy arrays by name, dtype, shape and bytes, independent of file format.

    The same weights give the same digest whether they came from an upstream model card or a
    local checkpoint file, so official exports, whose checkpoint fairseq2 downloads itself,
    are identified as well.
    """
    import numpy as np

    digest = hashlib.sha256()
    for name, array in sorted(named_arrays, key=lambda item: item[0]):
        array = np.ascontiguousarray(array)
        digest.update(f"{name}\0{array.dtype.str}\0{list(array.shape)}\n".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def distribution(name):
    """Return the installed version of `name`, plus its source when installed from VCS."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    record = {"version": dist.version}
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        try:
            origin = json.loads(direct_url)
        except ValueError:
            origin = {}
        # A local install's file:// URL is a path on the exporting machine, not a source.
        if "url" in origin and not origin["url"].startswith("file:"):
            record["url"] = origin["url"]
        vcs = origin.get("vcs_info") or {}
        if "commit_id" in vcs:
            record["commit"] = vcs["commit_id"]
        if (origin.get("dir_info") or {}).get("editable"):
            record["editable"] = True
    return record


def environment(packages=EXPORT_PACKAGES) -> dict:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: distribution(name) for name in packages},
    }


def _git(*args, cwd):
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def exporter(script, root=REPOSITORY_ROOT) -> dict:
    """Describe the fast-omniasr checkout that ran `script`."""
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    return {
        "script": script,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def write(directory, record) -> Path:
    """Write `record` as the directory's provenance.json, adding the format and timestamp."""
    record = {
        "format": PROVENANCE_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    path = Path(directory) / PROVENANCE_FILE
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    return path

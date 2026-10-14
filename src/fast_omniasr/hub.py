"""Download Hub assets declared by a repo's config.json and verify their sha256."""

import hashlib
import json
from pathlib import Path


def download_verified(repo_id: str, required: set[str], *, revision: str | None = None,
                      cache_dir: str | Path | None = None) -> tuple[dict, dict[str, Path]]:
    """Return the repo's config.json and a {filename: local path} map of every listed file.

    config.json must contain a `files` map of {filename: {"sha256", "size"}} that
    includes every name in `required`.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[hub] for huggingface_hub") from exc

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(repo_id, filename, revision=revision, cache_dir=cache_dir))

    def fetch_and_verify(filename: str, config: dict) -> Path:
        path = fetch(filename)
        expected = config.get("files", {}).get(filename, {}).get("sha256")
        if not expected:
            raise ValueError(f"config.json is missing a files.{filename}.sha256 entry")
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(
                f"{filename} checksum mismatch: downloaded asset does not match {repo_id}/config.json"
            )
        return path

    config = json.loads(fetch("config.json").read_text())
    files = config.get("files")
    if not isinstance(files, dict):
        raise TypeError("config.json is missing a files map")
    if not required.issubset(files):
        missing = ", ".join(sorted(required - files.keys()))
        raise ValueError(f"config.json is missing required file entries: {missing}")
    for filename in files:
        if not isinstance(files[filename], dict):
            raise TypeError(f"config.json has an invalid entry for {filename}")
        path = Path(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"config.json contains an unsafe filename: {filename}")

    # Smallest file first so corruption is caught before larger downloads when possible.
    verified = {}
    for filename in sorted(files, key=lambda name: files[name].get("size", 0)):
        verified[filename] = fetch_and_verify(filename, config)
    return config, verified

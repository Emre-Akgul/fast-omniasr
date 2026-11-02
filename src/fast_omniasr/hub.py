"""Download Hub assets declared by a repo's config.json and verify their sha256."""

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

DEFAULT_CACHE = Path.home() / ".cache" / "fast-omniasr" / "hub"

# A name usable verbatim as one directory: no separators, no leading dot, no "..".
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_MAX_COMPONENT = 200


def _path_component(name: str) -> str:
    """Map an arbitrary repo/revision name to a single, safe directory name.

    Names that are already safe (e.g. "main", "v1.0", a commit SHA) are kept verbatim so
    existing caches stay valid. Anything else ("refs/pr/1", "..", "/abs", "a\\b") is
    percent-encoded, dots included, so the result is never "." or "..", never contains a
    path separator, and always contains "%" -- it cannot collide with a verbatim name.
    Over-long results are replaced by a digest (still "%"-prefixed)."""
    if _SAFE_COMPONENT.fullmatch(name) and ".." not in name and len(name) <= _MAX_COMPONENT:
        return name
    if not name:
        return "%empty"  # quote() never emits "%e", so this can't collide
    # quote() leaves the RFC 3986 unreserved "." and "~" alone; encode them too. Still
    # injective because a literal "%" in the input always becomes "%25".
    encoded = quote(name, safe="-_").replace(".", "%2E").replace("~", "%7E")
    if len(encoded) > _MAX_COMPONENT:
        encoded = "%sha256-" + hashlib.sha256(name.encode()).hexdigest()
    return encoded


def _bundle_dir(cache_root: Path, repo_id: str, revision: str | None) -> Path:
    """The directory holding one repo revision, verified to stay inside `cache_root`."""
    local_dir = (cache_root / _path_component(repo_id.replace("/", "--"))
                 / _path_component(revision or "main"))
    # Lexical check (abspath collapses "..") rather than resolve(): users may legitimately
    # symlink a repo directory to another disk.
    root = Path(os.path.abspath(cache_root))
    if Path(os.path.abspath(local_dir)).parent.parent != root:
        raise ValueError(
            f"Refusing to cache {repo_id!r} at revision {revision!r} outside {cache_root}"
        )
    return local_dir


def download_verified(repo_id: str, required: set[str], *, revision: str | None = None,
                      cache_dir: str | Path | None = None,
                      select: Callable[[str], bool] | None = None
                      ) -> tuple[dict, dict[str, Path]]:
    """Return the repo's config.json and a {filename: local path} map of the listed files.

    config.json must contain a `files` map of {filename: {"sha256", "size"}} that
    includes every name in `required`. `select`, when given, limits downloads to the
    listed files it accepts (e.g. one precision's subdirectory).

    Files are materialized as regular files in one directory per repo and revision
    (under `cache_dir`, default ~/.cache/fast-omniasr/hub) rather than read through the
    Hugging Face cache's symlinks: ONNX Runtime 1.31+ rejects external tensor data whose
    resolved path leaves the model's directory, which every symlinked blob does.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[hub] for huggingface_hub") from exc

    local_dir = _bundle_dir(Path(cache_dir or DEFAULT_CACHE), repo_id, revision)

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(repo_id, filename, revision=revision, local_dir=local_dir))

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

    wanted = [name for name in files if select is None or select(name)]
    if not required.issubset(wanted):
        missing = ", ".join(sorted(required - set(wanted)))
        raise ValueError(f"required files are not selected for download: {missing}")
    # Smallest file first so corruption is caught before larger downloads when possible.
    verified = {}
    for filename in sorted(wanted, key=lambda name: files[name].get("size", 0)):
        verified[filename] = fetch_and_verify(filename, config)
    return config, verified

"""Download Hub assets declared by a repo's config.json and verify their sha256."""

import hashlib
import json
import os
import re
import tempfile
import warnings
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


_COMMIT = re.compile(r"[0-9a-f]{40}")
SNAPSHOT_METADATA = ".fast-omniasr-snapshot.json"


def _resolve_commit(repo_id: str, revision: str | None) -> tuple[str | None, str | None]:
    """(commit SHA the branch/tag currently points to, None) or (None, reason) when it
    can't be resolved right now -- offline mode, a network error, or an API failure."""
    revision = revision or "main"  # what downloads have always used, not the default branch
    if _COMMIT.fullmatch(revision):
        return revision, None  # already immutable
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(repo_id, revision=revision).sha
    except Exception as exc:  # noqa: BLE001 -- reported by the caller, never swallowed silently
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(sha, str) and _COMMIT.fullmatch(sha):
        return sha, None
    return None, f"unexpected commit {sha!r} from the Hub API"


def _read_snapshot(local_dir: Path) -> dict:
    try:
        record = json.loads((local_dir / SNAPSHOT_METADATA).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _write_snapshot(local_dir: Path, record: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    # A unique temporary name: concurrent writers must not replace each other's file.
    fd, tmp = tempfile.mkstemp(prefix=SNAPSHOT_METADATA + ".", dir=local_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(record, indent=2) + "\n")
        os.replace(tmp, local_dir / SNAPSHOT_METADATA)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


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

    A branch or tag is resolved to its commit SHA once, and every file (config.json
    included) is downloaded from that one commit, so a revision that moves mid-download
    can't produce a mixed bundle. The commit is recorded in `.fast-omniasr-snapshot.json`
    beside the files. If the revision can't be resolved (e.g. HF_HUB_OFFLINE=1), the
    recorded commit is reused; with no record, files are fetched by the revision name.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[hub] for huggingface_hub") from exc

    local_dir = _bundle_dir(Path(cache_dir or DEFAULT_CACHE), repo_id, revision)
    requested = revision or "main"
    commit, reason = _resolve_commit(repo_id, revision)
    if commit is None:
        recorded = _read_snapshot(local_dir)
        if (recorded.get("repo_id") == repo_id and recorded.get("revision") == requested
                and _COMMIT.fullmatch(str(recorded.get("commit", "")))):
            commit = recorded["commit"]
        if commit is not None and not reason.startswith("OfflineModeIsEnabled"):
            # Explicit offline mode (HF_HUB_OFFLINE=1) is intentional; anything else may
            # be a transient failure that leaves an older snapshot in use.
            warnings.warn(
                f"Could not resolve {repo_id}@{requested} ({reason}); using the previously "
                f"downloaded commit {commit[:12]}, which may not be the latest",
                stacklevel=2,
            )
        elif commit is None:
            warnings.warn(
                f"Could not resolve {repo_id}@{requested} to a commit ({reason}); "
                "downloading files by revision name without pinning",
                stacklevel=2,
            )
    pinned = commit or requested  # one immutable commit for every file when known

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(repo_id, filename, revision=pinned, local_dir=local_dir))

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
    if commit is not None:
        _write_snapshot(local_dir, {"repo_id": repo_id, "revision": requested, "commit": commit})
    return config, verified

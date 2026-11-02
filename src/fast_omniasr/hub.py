"""Download Hub assets declared by a repo's config.json and verify their sha256."""

import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import uuid
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


_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _bundle_lock(lock_path: Path):
    """Exclusive lock per bundle across threads and processes (advisory). A process-local
    lock comes first because on some filesystems (e.g. NFS) the file lock is per process
    and would not keep this process's own threads apart."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(os.path.abspath(lock_path), threading.Lock())
    with thread_lock, lock_path.open("a+b") as f:
        try:
            import fcntl
        except ImportError:  # Windows
            import msvcrt
            import time

            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as exc:  # LK_LOCK gives up after ~10 s of contention
                    if exc.errno not in (errno.EDEADLOCK, errno.EACCES):
                        raise
                    time.sleep(1)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            return
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, entry: dict) -> bool:
    if not path.is_file():
        return False
    size = entry.get("size")
    if isinstance(size, int) and path.stat().st_size != size:
        return False  # cheap rejection before hashing
    return _sha256(path) == entry.get("sha256")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)  # same filesystem: no data copied
    except OSError:
        shutil.copy2(source, target)


def _parse_config(config: dict, required: set[str],
                  select: Callable[[str], bool] | None) -> list[str]:
    """Validate config.json and return the files to materialize, smallest first."""
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
        if not files[filename].get("sha256"):
            raise ValueError(f"config.json is missing a files.{filename}.sha256 entry")
    wanted = [name for name in files if select is None or select(name)]
    if not required.issubset(wanted):
        missing = ", ".join(sorted(required - set(wanted)))
        raise ValueError(f"required files are not selected for download: {missing}")
    # Smallest file first so corruption is caught before larger downloads when possible.
    return sorted(wanted, key=lambda name: files[name].get("size", 0))


def _sidecars(local_dir: Path, kind: str) -> list[Path]:
    """This bundle's own staging/backup directories. "~" never occurs in a bundle
    directory name (`_path_component` percent-encodes it), so revision "v1" can never
    match "v1-rc"'s directories."""
    pattern = re.compile(rf"\.{kind}~{re.escape(local_dir.name)}~[0-9a-f]{{32}}")
    return [p for p in local_dir.parent.glob(f".{kind}~*") if pattern.fullmatch(p.name)]


def _sidecar(local_dir: Path, kind: str) -> Path:
    return local_dir.parent / f".{kind}~{local_dir.name}~{uuid.uuid4().hex}"


def _recover(local_dir: Path) -> None:
    """Under the bundle lock: undo a publish interrupted between its two renames, and
    remove staging/backup directories left behind by crashed processes."""
    backups = _sidecars(local_dir, "old")
    if not local_dir.exists() and backups:
        # Prefer a backup that still looks like a complete bundle (a crash during the
        # backup's rmtree can leave a partial one behind).
        complete = [b for b in backups
                    if (b / "config.json").is_file() and (b / SNAPSHOT_METADATA).is_file()]
        chosen = (complete or backups)[0]
        os.rename(chosen, local_dir)
        backups.remove(chosen)
    for leftover in [*backups, *_sidecars(local_dir, "staging")]:
        shutil.rmtree(leftover, ignore_errors=True)


def _publish(staging: Path, local_dir: Path) -> None:
    """Swap a complete, verified staging directory in for the published bundle."""
    if not local_dir.exists():
        os.rename(staging, local_dir)
        return
    backup = _sidecar(local_dir, "old")
    os.rename(local_dir, backup)
    try:
        os.rename(staging, local_dir)
    except BaseException:
        os.rename(backup, local_dir)  # put the previous bundle straight back
        raise
    shutil.rmtree(backup, ignore_errors=True)


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
    record = {"repo_id": repo_id, "revision": requested, "commit": commit}

    with _bundle_lock(local_dir.parent / f".{local_dir.name}.lock"):
        _recover(local_dir)
        reusable = _reuse_published(local_dir, record, required, select)
        if reusable is not None:
            return reusable
        staging = _sidecar(local_dir, "staging")
        try:
            config_path = Path(hf_hub_download(repo_id, "config.json", revision=pinned,
                                               local_dir=staging))
            config = json.loads(config_path.read_text())
            wanted = _parse_config(config, required, select)
            files = config["files"]
            carried = set()
            if local_dir.is_dir():
                # Carry over every file the new config still vouches for (hardlinked, so
                # unchanged multi-GB weights are not downloaded or copied again).
                for filename, entry in files.items():
                    if _matches(local_dir / filename, entry):
                        _link_or_copy(local_dir / filename, staging / filename)
                        carried.add(filename)
            for filename in wanted:
                if filename not in carried:
                    path = Path(hf_hub_download(repo_id, filename, revision=pinned,
                                                local_dir=staging))
                    if not _matches(path, files[filename]):
                        raise RuntimeError(
                            f"{filename} checksum mismatch: downloaded asset does not match "
                            f"{repo_id}/config.json"
                        )
            if commit is not None:
                _write_snapshot(staging, record)
            _publish(staging, local_dir)
        except BaseException:
            # The published bundle (if any) is untouched; drop the partial staging copy.
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return config, {filename: local_dir / filename for filename in wanted}


def _reuse_published(local_dir: Path, record: dict, required: set[str],
                     select: Callable[[str], bool] | None):
    """The published bundle as (config, paths) if it is the wanted commit and every
    selected file verifies against its config.json; otherwise None."""
    config_path = local_dir / "config.json"
    if not config_path.is_file():
        return None
    if record["commit"] is not None:
        recorded = _read_snapshot(local_dir)
        if recorded.get("commit") != record["commit"]:
            return None
    try:
        config = json.loads(config_path.read_text())
        wanted = _parse_config(config, required, select)
    except (ValueError, TypeError):
        return None
    files = config["files"]
    if not all(_matches(local_dir / name, files[name]) for name in wanted):
        return None
    return config, {name: local_dir / name for name in wanted}

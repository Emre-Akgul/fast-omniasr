"""Download Hub assets declared by a repo's config.json and verify their sha256."""

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ._locking import file_lock

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
        record = json.loads((local_dir / SNAPSHOT_METADATA).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _write_snapshot(local_dir: Path, record: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    # A unique temporary name: concurrent writers must not replace each other's file.
    fd, tmp = tempfile.mkstemp(prefix=SNAPSHOT_METADATA + ".", dir=local_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2) + "\n")
        _replace(tmp, local_dir / SNAPSHOT_METADATA)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace(source, target, attempts=50):
    """os.replace, retried briefly on Windows, where replacing a file another process is
    replacing (or has open) at the same moment fails with PermissionError."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == attempts - 1:
                raise
            time.sleep(0.01 * (attempt + 1))


# The per-bundle lock (portable; see _locking.file_lock).
_bundle_lock = file_lock


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


def _parse_config(config: dict, required, select: Callable[[str], bool] | None,
                  dependencies: bool = False) -> list[str]:
    """Validate config.json and return the files to materialize, smallest first: the
    required ones plus, unless `dependencies`, every listed file `select` accepts. The
    external data of the chosen graphs is added later (see _graph_closure)."""
    files = config.get("files")
    if not isinstance(files, dict):
        raise TypeError("config.json is missing a files map")
    if callable(required):
        required = set(required(config))
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
    if dependencies:
        wanted = [name for name in files if name in required]
    else:
        wanted = [name for name in files if select is None or select(name)]
    if not required.issubset(wanted):
        missing = ", ".join(sorted(required - set(wanted)))
        raise ValueError(f"required files are not selected for download: {missing}")
    # Smallest file first so corruption is caught before larger downloads when possible.
    return sorted(wanted, key=lambda name: files[name].get("size", 0))


def _graph_closure(root: Path, wanted: list[str], files: dict) -> list[str]:
    """External-data files the wanted ONNX graphs (already under `root`) reference and
    that aren't wanted yet, smallest first. Each must be listed in config.json, so it is
    checksum-verified like every other file."""
    from ._onnx_deps import graph_dependencies

    extra = set()
    for name in wanted:
        if name.endswith(".onnx"):
            for dependency in graph_dependencies(name, root / name):
                if dependency not in files:
                    raise ValueError(
                        f"{name} references external data {dependency}, which config.json "
                        "does not list"
                    )
                extra.add(dependency)
    extra -= set(wanted)
    return sorted(extra, key=lambda name: files[name].get("size", 0))


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


class LocalBundleNotFoundError(FileNotFoundError):
    """`local_files_only=True` found no complete, verified copy in the cache."""


@dataclass(frozen=True)
class Snapshot:
    """Where a loaded model came from: the requested repo and revision, the commit it
    resolved to (None if it couldn't be resolved, e.g. offline on first download), and
    the local bundle directory."""

    repo_id: str
    revision: str
    commit: str | None
    path: Path


@dataclass(frozen=True)
class CachedBundle(Snapshot):
    size_bytes: int = 0


def _delete_call(repo_id: str, revision: str | None, cache_dir) -> str:
    where = "" if cache_dir is None else f", cache_dir={str(cache_dir)!r}"
    return f"fast_omniasr.hub.delete_cached({repo_id!r}, {revision or 'main'!r}{where})"


def _recovery_hint(repo_id: str, revision: str | None, cache_dir) -> str:
    return (f"Retry the download, or remove the cached copy with "
            f"{_delete_call(repo_id, revision, cache_dir)} and load again.")


# Hub errors where loading an older local copy is no answer (the repo, revision or file
# doesn't exist, or access is denied).
_NOT_TRANSIENT = {"RepositoryNotFoundError", "RevisionNotFoundError", "EntryNotFoundError",
                  "GatedRepoError", "DisabledRepoError"}


def _transient(exc: BaseException) -> bool:
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "LocalEntryNotFoundError" in names:
        # Raised when the Hub can't be reached (network down, timeout, HF_HUB_OFFLINE),
        # although it subclasses EntryNotFoundError.
        return True
    if names & _NOT_TRANSIENT:
        return False
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403, 404):
        return False
    return isinstance(exc, OSError) or type(exc).__module__.startswith(
        ("huggingface_hub", "requests", "urllib3", "httpx"))


def _note(exc: BaseException, text: str) -> None:
    if hasattr(exc, "add_note"):  # Python 3.11+
        exc.add_note(text)
    else:  # Python 3.10: the hint can't ride on the exception, so warn instead
        warnings.warn(text, stacklevel=3)


def download_verified(repo_id: str, required, *, revision: str | None = None,
                      cache_dir: str | Path | None = None,
                      select: Callable[[str], bool] | None = None,
                      dependencies: bool = False,
                      local_files_only: bool = False,
                      return_snapshot: bool = False):
    """Return the repo's config.json and a {filename: local path} map of the listed files.

    config.json must contain a `files` map of {filename: {"sha256", "size"}} that
    includes every name in `required` (a set, or a function of the config returning
    one). With `dependencies=True` only the required files are downloaded, plus the
    external-data files their ONNX graphs reference wherever they live in the repo (the
    graph's dependency closure). Otherwise `select`, when given, limits downloads to the
    listed files it accepts, and all listed files are downloaded without it; the external
    data of every downloaded graph is added in either case.

    Files are materialized as regular files in one directory per repo and revision
    (under `cache_dir`, default ~/.cache/fast-omniasr/hub) rather than read through the
    Hugging Face cache's symlinks: ONNX Runtime 1.31+ rejects external tensor data whose
    resolved path leaves the model's directory, which every symlinked blob does.

    A branch or tag is resolved to its commit SHA once, and every file (config.json
    included) is downloaded from that one commit, so a revision that moves mid-download
    can't produce a mixed bundle. The commit is recorded in `.fast-omniasr-snapshot.json`
    beside the files. If the revision can't be resolved (e.g. HF_HUB_OFFLINE=1), the
    recorded commit is reused; with no record, files are fetched by the revision name.

    `local_files_only=True` never contacts the Hub: the cached bundle (the recorded
    commit of a branch or tag) is verified and returned, or LocalBundleNotFoundError
    explains what is missing.

    Returns (config, paths), or (config, paths, Snapshot) with `return_snapshot=True`;
    the Snapshot's commit is the one the returned files were verified for (None when it
    couldn't be determined).
    """
    if local_files_only:
        result = _load_local(repo_id, required, revision, cache_dir, select, dependencies)
    else:
        result = _download(repo_id, required, revision, cache_dir, select, dependencies)
    config, paths, snapshot = result
    return (config, paths, snapshot) if return_snapshot else (config, paths)


def _recorded_commit(local_dir: Path, repo_id: str, requested: str) -> str | None:
    recorded = _read_snapshot(local_dir)
    if (recorded.get("repo_id") == repo_id and recorded.get("revision") == requested
            and _COMMIT.fullmatch(str(recorded.get("commit", "")))):
        return recorded["commit"]
    return None


def _download(repo_id, required, revision, cache_dir, select, dependencies):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[hub] for huggingface_hub") from exc

    local_dir = _bundle_dir(Path(cache_dir or DEFAULT_CACHE), repo_id, revision)
    requested = revision or "main"
    commit, reason = _resolve_commit(repo_id, revision)
    if commit is None:
        commit = _recorded_commit(local_dir, repo_id, requested)
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
        reusable = _reuse_published(local_dir, record, required, select, dependencies)
        if reusable is not None:
            # Read under the lock: the commit of exactly the files being returned.
            used = commit or _recorded_commit(local_dir, repo_id, requested)
            return (*reusable, Snapshot(repo_id, requested, used, local_dir))
        staging = _sidecar(local_dir, "staging")
        try:
            config_path = Path(hf_hub_download(repo_id, "config.json", revision=pinned,
                                               local_dir=staging))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            wanted = _parse_config(config, required, select, dependencies)
            files = config["files"]
            carried = set()
            if local_dir.is_dir():
                # Carry over every file the new config still vouches for (hardlinked, so
                # unchanged multi-GB weights are not downloaded or copied again).
                for filename, entry in files.items():
                    if _matches(local_dir / filename, entry):
                        _link_or_copy(local_dir / filename, staging / filename)
                        carried.add(filename)
            def fetch(names):
                for filename in names:
                    if filename not in carried:
                        path = Path(hf_hub_download(repo_id, filename, revision=pinned,
                                                    local_dir=staging))
                        if not _matches(path, files[filename]):
                            raise RuntimeError(
                                f"{filename} checksum mismatch: downloaded asset does not "
                                f"match {repo_id}/config.json. The previously cached copy "
                                "(if any) is untouched. "
                                + _recovery_hint(repo_id, revision, cache_dir)
                            )

            fetch(wanted)
            # Graphs are verified first, then parsed for the external data they need.
            closure = _graph_closure(staging, wanted, files)
            fetch(closure)
            wanted += closure
            if commit is not None:
                _write_snapshot(staging, record)
            _publish(staging, local_dir)
        except BaseException as exc:
            # The published bundle (if any) is untouched; drop the partial staging copy.
            shutil.rmtree(staging, ignore_errors=True)
            if _transient(exc):
                _note(exc, f"Downloading {repo_id}@{requested} failed; nothing in the cache "
                           "was changed. If a complete copy was downloaded before, "
                           "local_files_only=True loads it without the network.")
            raise
    return (config, {filename: local_dir / filename for filename in wanted},
            Snapshot(repo_id, requested, commit, local_dir))


def _reuse_published(local_dir: Path, record: dict, required,
                     select: Callable[[str], bool] | None, dependencies: bool = False):
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
        config = json.loads(config_path.read_text(encoding="utf-8"))
        wanted = _parse_config(config, required, select, dependencies)
    except (ValueError, TypeError):
        return None
    files = config["files"]
    if not all(_matches(local_dir / name, files[name]) for name in wanted):
        return None
    try:
        closure = _graph_closure(local_dir, wanted, files)
    except ValueError:
        return None  # re-download; a fresh copy reports the problem
    if not all(_matches(local_dir / name, files[name]) for name in closure):
        return None
    return config, {name: local_dir / name for name in wanted + closure}


def _diagnose(local_dir: Path, required, select, dependencies) -> tuple[list, list]:
    """(missing, damaged) file names of a cached bundle that failed reuse."""
    try:
        config = json.loads((local_dir / "config.json").read_text(encoding="utf-8"))
        wanted = _parse_config(config, required, select, dependencies)
    except (OSError, ValueError, TypeError):
        return ["config.json"], []
    files = config["files"]
    try:
        present = [n for n in wanted if (local_dir / n).is_file()]
        names = wanted + _graph_closure(local_dir, present, files)
    except ValueError:
        names = wanted
    missing = [n for n in names if not (local_dir / n).is_file()]
    damaged = [n for n in names if n not in missing and not _matches(local_dir / n, files[n])]
    return missing, damaged


def _load_local(repo_id, required, revision, cache_dir, select, dependencies):
    local_dir = _bundle_dir(Path(cache_dir or DEFAULT_CACHE), repo_id, revision)
    requested = revision or "main"
    load_online = ("Load it once without local_files_only (with network access) to "
                   "download it.")
    if not local_dir.is_dir() and not _sidecars(local_dir, "old"):
        # Nothing to lock, recover or verify; create nothing. (A backup alone is a publish
        # interrupted between its renames, which _recover restores below.)
        raise LocalBundleNotFoundError(
            f"local_files_only=True: {repo_id}@{requested} is not in the cache "
            f"({local_dir}). {load_online}")
    try:
        lock = _bundle_lock(local_dir.parent / f".{local_dir.name}.lock")
        lock.__enter__()
    except OSError:
        lock = None  # a read-only cache (e.g. baked into an image): verify without locking
    try:
        if lock is not None:
            _recover(local_dir)
        # Read under the lock, so a concurrent publish can't pair a stale record with
        # newer files.
        commit = revision if revision and _COMMIT.fullmatch(revision) else None
        commit = commit or _recorded_commit(local_dir, repo_id, requested)
        record = {"repo_id": repo_id, "revision": requested, "commit": commit}
        reusable = _reuse_published(local_dir, record, required, select, dependencies)
        if reusable is not None:
            return (*reusable, Snapshot(repo_id, requested, commit, local_dir))
        missing, damaged = _diagnose(local_dir, required, select, dependencies)
    finally:
        if lock is not None:
            lock.__exit__(None, None, None)
    parts = []
    if missing:
        parts.append(f"missing {', '.join(missing)} (the cached copy holds other files, "
                     "e.g. another precision)")
    if damaged:
        parts.append(f"{', '.join(damaged)} fail checksum verification")
    detail = "; ".join(parts) or "the cached copy is for a different commit"
    advice = load_online
    if damaged:
        advice += (" To discard the damaged copy first, run "
                   f"{_delete_call(repo_id, revision, cache_dir)}.")
    raise LocalBundleNotFoundError(
        f"local_files_only=True: {repo_id}@{requested} in {local_dir}: {detail}. {advice}")


def _cache_root(cache_dir) -> Path:
    return Path(cache_dir or DEFAULT_CACHE)


_SIDECAR = re.compile(r"\.(?:staging|old)~(.+)~[0-9a-f]{32}")


def list_cached(cache_dir: str | Path | None = None) -> list[CachedBundle]:
    """Every published bundle in the cache, with its recorded commit and size on disk.
    In-progress staging copies and lock files are not listed. A bundle being replaced
    while this runs may be skipped."""
    from urllib.parse import unquote

    root = _cache_root(cache_dir)
    bundles = []
    for repo in sorted(p for p in root.glob("*") if p.is_dir() and not p.name.startswith(".")):
        for local_dir in sorted(p for p in repo.iterdir()
                                if p.is_dir() and not p.name.startswith(".")):
            recorded = _read_snapshot(local_dir)
            repo_id = recorded.get("repo_id") or unquote(repo.name).replace("--", "/", 1)
            revision = recorded.get("revision") or unquote(local_dir.name)
            commit = recorded.get("commit")
            try:
                size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            except FileNotFoundError:
                continue  # swapped out by a concurrent publish
            bundles.append(CachedBundle(
                repo_id, revision, commit if _COMMIT.fullmatch(str(commit or "")) else None,
                local_dir, size))
    return bundles


def delete_cached(repo_id: str, revision: str = "main", *, all_revisions: bool = False,
                  cache_dir: str | Path | None = None) -> list[Path]:
    """Delete the cached bundle of `repo_id` at `revision` ("main" by default, as in
    from_pretrained), or every revision with `all_revisions=True`. Each bundle is removed
    under the same lock downloads use, so a concurrent download is never cut short, and
    leftovers of crashed downloads go with it. Returns the deleted bundle directories."""
    from .options import check_flag

    check_flag("all_revisions", all_revisions)
    root = _cache_root(cache_dir)
    if all_revisions:
        repo_dir = _bundle_dir(root, repo_id, "main").parent
        names = set()
        if repo_dir.is_dir():
            for entry in repo_dir.iterdir():
                sidecar = _SIDECAR.fullmatch(entry.name)
                if sidecar:
                    names.add(sidecar.group(1))
                elif entry.is_dir() and not entry.name.startswith("."):
                    names.add(entry.name)
        targets = [repo_dir / name for name in sorted(names)]
    else:
        targets = [_bundle_dir(root, repo_id, revision)]
    deleted = []
    for local_dir in targets:
        if not local_dir.parent.is_dir():
            continue  # nothing cached for this repo; create nothing
        with _bundle_lock(local_dir.parent / f".{local_dir.name}.lock"):
            _recover(local_dir)  # also clears staging/backup leftovers
            if local_dir.exists():
                shutil.rmtree(local_dir)
                deleted.append(local_dir)
    return deleted

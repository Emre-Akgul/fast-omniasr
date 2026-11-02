"""Portable cache locking (issue #18): POSIX flock, Windows msvcrt, and thread safety."""

import builtins
import errno
import multiprocessing
import sys
import threading
import time
import types

import pytest

from fast_omniasr import _locking, tensorrt_cache
from fast_omniasr._locking import file_lock


def _hold(lock_path, log_path, tag):
    with file_lock(lock_path):
        with open(log_path, "a") as log:
            log.write(f"{tag}-start\n")
        time.sleep(0.3)
        with open(log_path, "a") as log:
            log.write(f"{tag}-end\n")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX flock path")
def test_processes_never_hold_the_lock_together(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_hold, args=(tmp_path / "x.lock", tmp_path / "log", t))
             for t in ("a", "b")]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30)
        assert proc.exitcode == 0
    events = (tmp_path / "log").read_text().split()
    assert len(events) == 4
    assert events[0].split("-")[0] == events[1].split("-")[0]  # first holder finished first
    assert events[0].endswith("start") and events[1].endswith("end")


def test_threads_are_serialized(tmp_path):
    inside, overlaps = [], []

    def worker():
        with file_lock(tmp_path / "t.lock"):
            inside.append(1)
            overlaps.append(len(inside) > 1)
            time.sleep(0.01)
            inside.pop()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not any(overlaps)


@pytest.fixture
def windows(monkeypatch):
    """Simulate Windows: `import fcntl` fails and a fake msvcrt records lock calls."""
    real_import = builtins.__import__

    def no_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl on Windows")
        return real_import(name, *args, **kwargs)

    calls, failures = [], []

    def locking(fd, mode, nbytes):
        if failures:
            raise failures.pop(0)
        calls.append((mode, nbytes))

    msvcrt = types.SimpleNamespace(LK_LOCK="LK_LOCK", LK_UNLCK="LK_UNLCK", locking=locking)
    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.delitem(sys.modules, "fcntl", raising=False)
    monkeypatch.setattr(_locking.time, "sleep", lambda s: None)
    return types.SimpleNamespace(calls=calls, failures=failures)


def test_windows_locks_and_unlocks_one_byte(windows, tmp_path):
    with file_lock(tmp_path / "w.lock"):
        assert windows.calls == [("LK_LOCK", 1)]
    assert windows.calls == [("LK_LOCK", 1), ("LK_UNLCK", 1)]


def test_windows_retries_while_contended(windows, tmp_path):
    windows.failures.extend([OSError(errno.EDEADLOCK, "busy"), OSError(errno.EACCES, "busy")])
    with file_lock(tmp_path / "w.lock"):
        pass
    assert windows.calls == [("LK_LOCK", 1), ("LK_UNLCK", 1)]


def test_windows_raises_real_errors_instead_of_spinning(windows, tmp_path):
    windows.failures.append(OSError(errno.EBADF, "bad file descriptor"))
    with pytest.raises(OSError, match="bad file descriptor"), file_lock(tmp_path / "w.lock"):
        pytest.fail("must not enter the locked block")


def test_tensorrt_cache_works_without_fcntl(windows, tmp_path, monkeypatch):
    builds = []

    def fake_build_engine(onnx_path, output_path, **kwargs):
        builds.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"engine")

    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(bytes([0x08, 0x08]))  # a minimal valid ModelProto (ir_version = 8)
    monkeypatch.setattr(tensorrt_cache, "build_engine", fake_build_engine)
    monkeypatch.setattr(tensorrt_cache, "_local_environment",
                        lambda: {"tensorrt_version": "10", "tensorrt_cuda": "cu12",
                                 "compute_capability": "8.0", "device_name": "Example GPU"})
    first = tensorrt_cache.get_or_build_engine(onnx, precision="fp16", cache_dir=tmp_path / "c")
    again = tensorrt_cache.get_or_build_engine(onnx, precision="fp16", cache_dir=tmp_path / "c")
    assert first == again and len(builds) == 1
    assert windows.calls.count(("LK_LOCK", 1)) == 2  # the engine cache took the msvcrt lock

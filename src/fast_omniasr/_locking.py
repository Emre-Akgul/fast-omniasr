"""One portable, exclusive lock per path, shared by the Hub and TensorRT engine caches."""

import contextlib
import errno
import os
import threading
import time
from pathlib import Path

_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def file_lock(lock_path: Path):
    """Exclusive lock on `lock_path` across threads and processes (advisory).

    A process-local lock comes first because on some filesystems (e.g. NFS) the file lock
    is per process and would not keep this process's own threads apart. The file lock is
    `fcntl.flock` on POSIX and `msvcrt.locking` on Windows."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(os.path.abspath(lock_path), threading.Lock())
    with thread_lock, lock_path.open("a+b") as f:
        try:
            import fcntl
        except ImportError:  # Windows
            with _msvcrt_locked(f):
                yield
            return
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


@contextlib.contextmanager
def _msvcrt_locked(f):
    import msvcrt

    f.seek(0)
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            break
        except OSError as exc:  # LK_LOCK gives up after ~10 s of contention; keep waiting
            if exc.errno not in (errno.EDEADLOCK, errno.EACCES):
                raise
            time.sleep(1)
    try:
        yield
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

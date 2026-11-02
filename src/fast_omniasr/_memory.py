"""Best-effort return of freed heap memory to the operating system."""

import ctypes
import functools
import sys


@functools.cache
def _malloc_trim():
    """glibc's malloc_trim, looked up once without spawning a subprocess; None when the
    allocator isn't glibc (e.g. musl) or no C library can be loaded."""
    for name in (None, "libc.so.6"):  # None: symbols already loaded into the process
        try:
            trim = ctypes.CDLL(name).malloc_trim
        except (OSError, AttributeError):
            continue
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return trim
    return None


def release_freed_memory() -> None:
    """After dropping a large ONNX Runtime session, glibc keeps the freed pages in its
    own pools, so the process RSS stays high even though nothing references them.
    `malloc_trim(0)` hands them back. It walks every heap arena, so on a very large
    multithreaded heap it can take tens of milliseconds. A no-op off Linux/glibc."""
    if not sys.platform.startswith("linux"):
        return
    trim = _malloc_trim()
    if trim is not None:
        trim(0)

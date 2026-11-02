"""Best-effort return of freed heap memory to the operating system."""

import ctypes
import ctypes.util
import sys


def release_freed_memory() -> None:
    """After dropping a large ONNX Runtime session, glibc keeps the freed pages in its
    own pools, so the process RSS stays high even though nothing references them.
    `malloc_trim(0)` hands them back. A no-op where glibc isn't the allocator."""
    if not sys.platform.startswith("linux"):
        return
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = libc.malloc_trim
    except (OSError, AttributeError):  # musl/other allocators, or no libc found
        return
    trim(0)

"""Validation shared by every runtime, so the same option fails the same way everywhere."""

import operator

DEVICES = ("cpu", "cuda")
DEFAULT_THREADS = 4


def check_device(device) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    return device


def check_threads(threads) -> int:
    """A positive integer of any integer type (int, numpy integers); bools rejected."""
    try:
        value = None if isinstance(threads, bool) else operator.index(threads)
    except TypeError:
        value = None
    if value is None or value < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    return value


def resolve_threads(threads) -> int:
    """`None` means the default thread count; anything else must pass check_threads."""
    return DEFAULT_THREADS if threads is None else check_threads(threads)


def check_flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return value

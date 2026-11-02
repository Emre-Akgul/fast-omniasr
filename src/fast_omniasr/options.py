"""Validation shared by every runtime, so the same option fails the same way everywhere."""

DEVICES = ("cpu", "cuda")


def check_device(device) -> str:
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    return device


def check_threads(threads) -> int:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    return threads


def check_flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return value

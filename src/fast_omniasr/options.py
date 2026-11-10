"""Validation shared by every runtime, so the same option fails the same way everywhere."""

import operator

DEVICES = ("cpu", "cuda")
DEFAULT_THREADS = 4

# ONNX Runtime CUDA provider options shared by the CTC and LLM runtimes. TF32 stays off so
# GPU results track the float32 reference. cuDNN 9.27 and newer pick a convolution that
# needs a ~17.6 GB workspace for every input under ~2.6 s (41,000 samples) under the
# EXHAUSTIVE and HEURISTIC algorithm searches, so those requests fail to allocate. DEFAULT
# uses one fixed algorithm, worked for every length on every cuDNN tested (9.10-9.28) and is
# ~5% slower on long clips. ONNX Runtime then logs "Conv ... running in Fallback mode" once
# per convolution for each new input length; that is expected (see #97).
CUDA_PROVIDER_OPTIONS = {"use_tf32": "0", "cudnn_conv_algo_search": "DEFAULT"}


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

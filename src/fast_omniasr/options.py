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


def check_device_index(device_index, device: str) -> int | None:
    """`None` (ONNX Runtime's GPU 0, or TensorRT's current CUDA device), or a GPU ordinal:
    a non-negative integer of any integer type, bools rejected, and only with "cuda"."""
    if device_index is None:
        return None
    try:
        value = None if isinstance(device_index, bool) else operator.index(device_index)
    except TypeError:
        value = None
    if value is None or value < 0:
        raise ValueError(f"device_index must be a non-negative integer, got {device_index!r}")
    if device != "cuda":
        raise ValueError(f"device_index only applies to device='cuda', got device={device!r}")
    return value


def cuda_provider_options(device_index: int | None = None) -> dict:
    """CUDA_PROVIDER_OPTIONS for one GPU; `None` leaves ONNX Runtime's default (GPU 0)."""
    options = dict(CUDA_PROVIDER_OPTIONS)
    if device_index is not None:
        options["device_id"] = str(device_index)
    return options


def provider_failure(provider: str, device_index: int | None) -> str:
    """Why a requested ONNX Runtime provider is missing from the session."""
    message = f"Requested provider {provider} did not initialize"
    if device_index is not None:
        message += (f" on device_index={device_index}; check that the GPU exists and is "
                    "visible (CUDA_VISIBLE_DEVICES). ONNX Runtime logged the reason above")
    return message


def check_threads(threads) -> int:
    """A positive integer of any integer type (int, numpy integers); bools rejected."""
    try:
        value = None if isinstance(threads, bool) else operator.index(threads)
    except TypeError:
        value = None
    if value is None or value < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}")
    return value


def check_batch_size(batch_size) -> int:
    """A positive integer of any integer type (int, numpy integers); bools rejected."""
    try:
        value = None if isinstance(batch_size, bool) else operator.index(batch_size)
    except TypeError:
        value = None
    if value is None or value < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    return value


def resolve_threads(threads) -> int:
    """`None` means the default thread count; anything else must pass check_threads."""
    return DEFAULT_THREADS if threads is None else check_threads(threads)


def check_flag(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, got {value!r}")
    return value

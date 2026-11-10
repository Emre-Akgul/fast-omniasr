"""Smoke-test an *installed* fast-omniasr (wheel, not the source tree) with one extra.

  python tests/packaging/smoke.py core|onnx|hub|llm|tensorrt|resample|minimum

Checks that the package imports from site-packages, that the extra's runtime dependency is
importable and works through the library's own code path, and that nothing pulls in a
dependency the extra doesn't declare. Needs no GPU, network or model files.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path


def absent(*names):
    present = [n for n in names if importlib.util.find_spec(n) is not None]
    assert not present, f"unexpected optional dependencies installed: {present}"


def core():
    import numpy as np

    import fast_omniasr
    from fast_omniasr.audio import prepare_audio
    from fast_omniasr.decoder import greedy_token_ids

    assert "site-packages" in fast_omniasr.__file__ or "dist-packages" in fast_omniasr.__file__, (
        f"fast_omniasr imported from {fast_omniasr.__file__}, not an installed wheel")
    audio = prepare_audio(np.zeros(16000, np.float32), 16000)
    assert audio.shape == (1, 16000)
    logits = np.zeros((1, 4, 5), np.float32)
    for frame, token in enumerate([1, 1, 0, 2]):
        logits[0, frame, token] = 1
    assert greedy_token_ids(logits) == [1, 2]  # repeats collapsed, blank (0) removed
    from importlib.metadata import entry_points

    from fast_omniasr.cli import main as cli

    scripts = entry_points(group="console_scripts")
    assert "fast-omniasr" in [script.name for script in scripts]
    assert cli(["diagnose", "--json"]) == 0  # works with no optional dependency
    return fast_omniasr


def onnx():
    import onnxruntime as ort

    from fast_omniasr.backends.onnx import ONNXBackend

    assert "CPUExecutionProvider" in ort.get_available_providers()
    from fast_omniasr.vad import get_vad_model

    get_vad_model()  # the bundled model shipped in the wheel and loads
    try:
        ONNXBackend(Path(tempfile.gettempdir()) / "missing-model.onnx")
    except Exception as exc:  # noqa: BLE001 -- ORT's own error for a missing file
        assert "missing-model.onnx" in str(exc) or "NO_SUCHFILE" in str(exc), exc
    else:
        raise AssertionError("loading a missing model succeeded")


def hub():
    from fast_omniasr import OmniASR, hub

    with tempfile.TemporaryDirectory() as cache:
        try:
            OmniASR.from_pretrained("owner/repo", cache_dir=cache, local_files_only=True)
        except hub.LocalBundleNotFoundError as exc:
            assert "is not in the cache" in str(exc)
        else:
            raise AssertionError("an empty cache loaded a model")
        assert hub.list_cached(cache) == []
    import huggingface_hub  # noqa: F401 -- the extra's dependency resolves


def llm():
    import torch

    from fast_omniasr.llm import OmniASRLLM  # noqa: F401

    assert torch.zeros(2).sum().item() == 0


def resample():
    import numpy as np

    from fast_omniasr.audio import prepare_audio

    stereo = np.zeros((44100, 2), np.float32)
    assert prepare_audio(stereo, 44100, convert_audio=True).shape == (1, 16000)


def tensorrt():
    # cuda-python is installable without a GPU; the bindings must import.
    try:
        from cuda.bindings import runtime  # noqa: F401  (cuda-python >= 12.6)
    except ImportError:
        from cuda import cudart  # noqa: F401  (older cuda-python)
    import fast_omniasr.tensorrt_cache  # noqa: F401


CHECKS = {"core": [], "onnx": [onnx], "hub": [hub], "llm": [llm], "tensorrt": [tensorrt],
          "resample": [resample], "minimum": [onnx, hub, resample]}
# Dependencies an extra must not drag in (each extra is tested on its own).
UNEXPECTED = {"core": ["onnxruntime", "torch", "huggingface_hub", "soxr"],
              "onnx": ["torch", "huggingface_hub", "soxr"],
              "hub": ["onnxruntime", "torch", "soxr"],
              "llm": ["huggingface_hub", "soxr"],
              "tensorrt": ["onnxruntime", "torch", "soxr"],
              "resample": ["onnxruntime", "torch", "huggingface_hub"],
              "minimum": ["torch"]}


def main(extra):
    module = core()
    for check in CHECKS[extra]:
        check()
    absent(*UNEXPECTED[extra])
    print(f"ok: {extra} ({module.__file__})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "core")

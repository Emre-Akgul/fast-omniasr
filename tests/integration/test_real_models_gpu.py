"""The published CTC model on a GPU, through every CUDA backend and precision: loading,
dynamic lengths, precision behavior, repeated calls and cleanup (#28).

Opt-in (GPU, downloads ~1.3 GB, builds the TensorRT engines on first use):
  OMNIASR_REAL_MODELS=1 pytest tests/integration/test_real_models_gpu.py

Configurations whose runtime is missing (onnxruntime-gpu, TensorRT, or TensorRT 11 for
'mixed') are skipped. Set OMNIASR_REQUIRE_GPU=1, as the GPU release workflow does, to make
those skips failures, so a misconfigured runner can't pass by skipping everything.
OMNIASR_CACHE_DIR overrides the download cache.
"""

import gc
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
PINNED = MANIFEST["models"]["ctc"]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.cuda,
    pytest.mark.skipif(os.environ.get("OMNIASR_REAL_MODELS") != "1",
                       reason="set OMNIASR_REAL_MODELS=1 to download and run the real models"),
]

CONFIGURATIONS = [("onnx", "fp32"), ("onnx", "fp16"), ("onnx", "bf16"),
                  ("tensorrt", "fp32"), ("tensorrt", "fp16"), ("tensorrt", "mixed")]
# Shortest input, the shortest lengths cuDNN once failed on (#97), a few seconds, and the
# default TensorRT profile's limit (30 s).
LENGTHS = [400, 16000, 33226, 41619, 117920, 240000, 480000]
# Memory a construct/close cycle may leave behind (allocator and driver noise); one engine
# or session of this model is several hundred MiB.
CYCLE_TOLERANCE_MB = 64
# Growth allowed over repeated requests once every length has been seen (as in
# tools/gpu_memory_soak.py, which runs a longer version of this check).
REPEAT_TOLERANCE_MB = 8


def _unavailable(reason):
    if os.environ.get("OMNIASR_REQUIRE_GPU") == "1":
        pytest.fail(f"OMNIASR_REQUIRE_GPU=1 but {reason}")
    pytest.skip(reason)


def _check_runtime(backend, precision):
    if backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError:
            _unavailable("onnxruntime is not installed")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            _unavailable("onnxruntime-gpu with a CUDA device is not available")
        return
    try:
        import tensorrt
    except ImportError:
        _unavailable("TensorRT is not installed")
    from fast_omniasr.tensorrt_builder import strongly_typed_only

    if precision == "mixed" and strongly_typed_only(tensorrt):
        # Refused by design on TensorRT 11 (#101); test_tensorrt_fleurs.py checks the message.
        pytest.skip("precision='mixed' isn't available with TensorRT 11")


def _load(backend, precision):
    from fast_omniasr import OmniASR

    _check_runtime(backend, precision)
    options = {"revision": PINNED["revision"], "backend": backend, "precision": precision,
               "cache_dir": os.environ.get("OMNIASR_CACHE_DIR") or None}
    if backend == "onnx":
        options["device"] = "cuda"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TensorRT FP16")
        return OmniASR.from_pretrained(PINNED["repo"], **options)


def _frames(samples):
    return (samples - 400) // 320 + 1


def _clips():
    return [(clip, *sf.read(FIXTURES / clip["file"], dtype="float32"))
            for clip in MANIFEST["clips"]]


def _error_rates():
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate_precisions",
                                                  ROOT / "tools" / "evaluate_precisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.error_rates


def _memory_reading():
    """A function returning this process's GPU memory in MiB from NVML (the device's, where
    NVML can't attribute memory to processes), or None without NVML."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("gpu_memory_soak",
                                                  ROOT / "tools" / "gpu_memory_soak.py")
    soak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soak)
    nvml = soak.NVML()

    def reading():
        process = nvml.process_mb()
        return process if process is not None else nvml.device_mb()

    return reading


@pytest.fixture(scope="module", params=CONFIGURATIONS, ids=lambda c: "-".join(c))
def gpu(request):
    backend, precision = request.param
    with _load(backend, precision) as model:
        yield backend, precision, model


def test_loads_on_the_gpu(gpu):
    backend, _, model = gpu
    assert f"{model.snapshot.repo_id}@{model.snapshot.commit}" == (
        f"{PINNED['repo']}@{PINNED['revision']}")
    if backend == "onnx":
        # The backend refuses a silent CPU fallback; check it is still the first provider.
        assert model.backend.session.get_providers()[0] == "CUDAExecutionProvider"
    else:
        from fast_omniasr.backends.tensorrt import TensorRTBackend
        from fast_omniasr.tensorrt_builder import DEFAULT_MAX_SAMPLES, DEFAULT_MIN_SAMPLES

        assert isinstance(model.backend, TensorRTBackend)
        assert (model.backend.min_samples, model.backend.max_samples) == (
            DEFAULT_MIN_SAMPLES, DEFAULT_MAX_SAMPLES)


def test_dynamic_lengths(gpu):
    backend, _, model = gpu
    vocabulary = None
    for n in LENGTHS:
        shape = model.transcribe_numpy(np.zeros(n, dtype=np.float32)).logits_shape
        assert shape[:2] == (1, _frames(n)), (n, shape)
        vocabulary = vocabulary or shape[2]
        assert shape[2] == vocabulary, (n, shape)
    if backend == "tensorrt":
        with pytest.raises(ValueError, match="engine profile"):
            model.transcribe_numpy(np.zeros(LENGTHS[-1] + 1, dtype=np.float32))
    # Past one call's limit, long-form transcription still covers the whole recording.
    result = model.transcribe_long(np.zeros(LENGTHS[-1] * 2, dtype=np.float32))
    assert result.chunks and result.chunks[-1].end == pytest.approx(60.0)


def test_precision_stays_within_the_cpu_ceilings(gpu, expect):
    """Every precision must stay within the CPU FP32 CER ceilings on the FLEURS fixtures,
    and match the exact transcript declared for this variant where there is one."""
    backend, precision, model = gpu
    error_rates = _error_rates()
    variant = f"onnx-cuda-{precision}" if backend == "onnx" else f"tensorrt-{precision}"
    model_key = f"{PINNED['repo']}@{PINNED['revision']}"
    for clip, waveform, sample_rate in _clips():
        text = model.transcribe_numpy(waveform, sample_rate).text
        cer = error_rates([clip["reference"]], [text])["cer"]
        assert cer <= clip["max_cer"]["ctc"], f"{variant} {clip['file']} CER {cer:.1f}: {text!r}"
        declared = (expect.declared(FIXTURES / clip["file"], model_key) or {})
        expected = declared.get("transcripts", {}).get(variant)
        if expected is not None:
            assert text == expected, variant


def test_repeated_calls_are_deterministic_and_flat(gpu):
    """The same input decodes the same however many requests of other lengths run in
    between, so no state (buffers, arenas, cached algorithms) leaks from one request into
    the next, and once every length has been seen the process's GPU memory stops growing."""
    _, _, model = gpu
    clips = _clips()
    rng = np.random.default_rng(0)
    noise = [(0.05 * rng.standard_normal(n)).astype(np.float32)
             for n in [480000, 400, 33226, 117920, 16000]]
    first = [model.transcribe_numpy(waveform, rate).token_ids for _, waveform, rate in clips]
    reading = _memory_reading()
    used = []
    for _ in range(4):
        for waveform in noise:
            model.transcribe_numpy(waveform)
        again = [model.transcribe_numpy(waveform, rate).token_ids for _, waveform, rate in clips]
        assert again == first
        used.append(reading())
    if used[0] is not None:
        assert max(used) - used[0] <= REPEAT_TOLERANCE_MB, used


@pytest.mark.parametrize("backend, precision", [("onnx", "fp32"), ("tensorrt", "fp16")],
                         ids=["onnx-fp32", "tensorrt-fp16"])
def test_close_releases_the_gpu(backend, precision):
    """close() is idempotent, refuses later use and gives the device memory back, so a
    long-running process can load and unload models without running out."""
    reading = _memory_reading()
    if reading() is None:
        _unavailable("NVML (pynvml) can't read GPU memory here")
    waveform = np.zeros(160000, dtype=np.float32)
    used = []
    for cycle in range(4):
        model = _load(backend, precision)
        loaded = reading()
        model.transcribe_numpy(waveform)
        model.close()
        model.close()
        with pytest.raises(RuntimeError, match="closed"):
            model.transcribe_numpy(waveform)
        del model
        gc.collect()
        used.append(reading())
        if cycle == 0:
            # The first cycle sets up the CUDA context and libraries for good; what the
            # model itself held must come back.
            assert loaded - used[0] > CYCLE_TOLERANCE_MB, (loaded, used[0])
    assert max(used) - used[0] <= CYCLE_TOLERANCE_MB, used

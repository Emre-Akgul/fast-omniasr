import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.cuda]


@pytest.fixture
def assets(clip):
    names = ["OMNIASR_TRT_FP32_ENGINE", "OMNIASR_TOKENIZER"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Set OMNIASR_TRT_FP32_ENGINE and OMNIASR_TOKENIZER (OMNIASR_TEST_WAV optional)")
    return [os.environ[name] for name in names] + [str(clip)]


def _within(model, lengths):
    """The lengths this engine's profile accepts (a custom profile may be narrower)."""
    return [n for n in lengths if model.backend.min_samples <= n <= model.backend.max_samples]


def test_real_transcript_without_training_frameworks(assets, expect, local_model_key):
    expected = expect.transcript(assets[2], local_model_key, "tensorrt-fp32")
    code = """
import json
import sys
import soundfile as sf
from fast_omniasr import OmniASR
model = OmniASR(sys.argv[1], sys.argv[2], backend="tensorrt")
result = model.transcribe(sys.argv[3])
samples = sf.info(sys.argv[3]).frames
assert result.logits_shape == (1, (samples - 400) // 320 + 1, int(sys.argv[4])), result
expected = json.loads(sys.argv[5])
if expected is not None:
    assert result.text == expected, result.text
assert result.text.strip(), "empty transcript"
assert not any(name in sys.modules for name in ['torch', 'fairseq2', 'transformers', 'onnxruntime'])
"""
    subprocess.run([sys.executable, "-c", code, *assets, str(expect.vocabulary(assets[1])),
                    json.dumps(expected)], check=True)


def test_dynamic_frame_counts(assets, expect):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    vocabulary = expect.vocabulary(assets[1])
    lengths = _within(model, [400, 16000, 80000, 117920, 240000, 480000])
    assert lengths, "the engine profile accepts none of the probe lengths"
    for n in lengths + [model.backend.min_samples, model.backend.max_samples]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, expect.frames(n), vocabulary)


def test_wav_and_numpy_agree(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    waveform, rate = sf.read(assets[2], dtype="float32")
    assert model.transcribe(assets[2]) == model.transcribe_numpy(waveform, rate)


def test_sequential_calls_of_varying_length(assets, expect):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    vocabulary = expect.vocabulary(assets[1])
    for n in _within(model, [16000, 480000, 80000]) + [model.backend.max_samples,
                                                       model.backend.min_samples]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, expect.frames(n), vocabulary)


def test_lengths_outside_the_engine_profile_raise(assets):
    """Uses the engine's own profile, so default (400..480000) and custom engines both work."""
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    low, high = model.backend.min_samples, model.backend.max_samples
    with pytest.raises(ValueError, match="engine profile"):
        model.transcribe_numpy(np.zeros(high + 1, dtype=np.float32))
    if low > 400:  # below 400 samples the input is rejected before reaching the engine
        with pytest.raises(ValueError, match="engine profile"):
            model.transcribe_numpy(np.zeros(low - 1, dtype=np.float32))


def test_buffers_are_reused_not_reallocated(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    pointers = (model.backend.input_ptr, model.backend.output_ptr)
    for n in _within(model, [16000, 480000, 80000, 240000]) + [model.backend.min_samples,
                                                               model.backend.max_samples]:
        model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert (model.backend.input_ptr, model.backend.output_ptr) == pointers


def test_close_is_idempotent_and_blocks_further_use(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    model.close()
    model.close()  # must not double-free or raise
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe(assets[2])


def test_context_manager_closes_on_exit(assets):
    from fast_omniasr import OmniASR
    with OmniASR(assets[0], assets[1], backend="tensorrt") as model:
        model.transcribe(assets[2])
    assert model.backend.closed


def test_repeated_construct_and_close_does_not_leak_gpu_memory(assets, expect):
    """Regression test: constructing/destroying many models in a long-running process must
    not exhaust GPU memory, since each engine alone can be gigabytes of device memory."""
    import gc

    from fast_omniasr import OmniASR
    expected = (1, expect.frames(sf.info(assets[2]).frames), expect.vocabulary(assets[1]))
    for _ in range(5):
        with OmniASR(assets[0], assets[1], backend="tensorrt") as model:
            result = model.transcribe(assets[2])
            assert result.logits_shape == expected
        del model
        gc.collect()


def test_concurrent_calls_match_sequential(assets):
    import threading

    from fast_omniasr import OmniASR
    with OmniASR(assets[0], assets[1], backend="tensorrt") as model:
        rng = np.random.default_rng(0)
        lengths = _within(model, [16000, 32000, 48000, 64000, 16000, 80000, 24000, 40000])
        if not lengths:
            pytest.skip("the engine profile accepts none of the concurrency probe lengths")
        inputs = [(0.1 * rng.standard_normal(n)).astype(np.float32) for n in lengths]
        sequential = [model.backend.infer(x[None]) for x in inputs]
        results, errors = [None] * len(inputs), []
        barrier = threading.Barrier(len(inputs))

        def worker(i):
            try:
                barrier.wait()
                for _ in range(3):
                    results[i] = model.backend.infer(inputs[i][None])
                    np.testing.assert_array_equal(results[i], sequential[i])
            except Exception as exc:  # noqa: BLE001 -- surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(inputs))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []

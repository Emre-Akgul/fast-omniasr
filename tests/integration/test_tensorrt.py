import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.cuda]


@pytest.fixture
def assets():
    names = ["OMNIASR_TRT_FP32_ENGINE", "OMNIASR_TOKENIZER", "OMNIASR_TEST_WAV"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Set OMNIASR_TRT_FP32_ENGINE, OMNIASR_TOKENIZER and OMNIASR_TEST_WAV")
    return [os.environ[name] for name in names]


@pytest.fixture
def fp16_assets():
    names = ["OMNIASR_TRT_FP16_ENGINE", "OMNIASR_TOKENIZER", "OMNIASR_TRT_FP16_TEST_WAV"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Set OMNIASR_TRT_FP16_ENGINE, OMNIASR_TOKENIZER and OMNIASR_TRT_FP16_TEST_WAV")
    return [os.environ[name] for name in names]


def test_real_transcript_without_training_frameworks(assets):
    code = """
import sys
from fast_omniasr import OmniASR
model = OmniASR(sys.argv[1], sys.argv[2], backend="tensorrt")
result = model.transcribe(sys.argv[3])
assert result.text == 'now i want to return to the conservation of mechanical energy'
assert result.logits_shape == (1, 249, 10288)
assert not any(name in sys.modules for name in ['torch', 'fairseq2', 'transformers', 'onnxruntime'])
"""
    subprocess.run([sys.executable, "-c", code, *assets], check=True)


def test_dynamic_frame_counts(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    for n in [16000, 80000, 117920, 240000, 480000]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, (n - 400) // 320 + 1, 10288)


def test_wav_and_numpy_agree(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    waveform, rate = sf.read(assets[2], dtype="float32")
    assert model.transcribe(assets[2]) == model.transcribe_numpy(waveform, rate)


def test_sequential_calls_of_varying_length(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    for n in [16000, 480000, 80000]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, (n - 400) // 320 + 1, 10288)


def test_length_outside_profile_raises(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    with pytest.raises(ValueError, match="engine profile"):
        model.transcribe_numpy(np.zeros(8000, dtype=np.float32))
    with pytest.raises(ValueError, match="engine profile"):
        model.transcribe_numpy(np.zeros(500_000, dtype=np.float32))


def test_buffers_are_reused_not_reallocated(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], backend="tensorrt")
    pointers = (model.backend.input_ptr, model.backend.output_ptr)
    for n in [16000, 480000, 80000, 240000]:
        model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert (model.backend.input_ptr, model.backend.output_ptr) == pointers


def test_fp16_known_experimental_transcript(fp16_assets):
    from fast_omniasr import OmniASR
    model = OmniASR(fp16_assets[0], fp16_assets[1], backend="tensorrt")
    result = model.transcribe(fp16_assets[2])
    assert result.text == (
        "  t   consen o mcnc   penulm   at w  k  lit it  mt c   m  o ng     "
        "iceae  tl n o    l  il t fal  w  convt"
    )

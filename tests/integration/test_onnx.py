import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def assets():
    names = ["OMNIASR_ONNX", "OMNIASR_TOKENIZER", "OMNIASR_TEST_WAV"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Set OMNIASR_ONNX, OMNIASR_TOKENIZER and OMNIASR_TEST_WAV")
    return [os.environ[name] for name in names]


def test_real_transcript_without_training_frameworks(assets):
    code = """
import sys
from fast_omniasr import OmniASR
model = OmniASR(sys.argv[1], sys.argv[2])
result = model.transcribe(sys.argv[3])
assert result.text == 'now i want to return to the conservation of mechanical energy'
assert result.logits_shape == (1, 249, 10288)
assert not any(name in sys.modules for name in ['torch', 'fairseq2', 'transformers'])
"""
    subprocess.run([sys.executable, "-c", code, *assets], check=True)


def test_dynamic_frame_counts(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1])
    for n in [16000, 80000, 117920, 240000, 480000]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, (n - 400) // 320 + 1, 10288)


def test_wav_and_numpy_agree(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1])
    waveform, rate = sf.read(assets[2], dtype="float32")
    assert model.transcribe(assets[2]) == model.transcribe_numpy(waveform, rate)

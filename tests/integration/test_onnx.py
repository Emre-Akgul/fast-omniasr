import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def assets(clip):
    names = ["OMNIASR_ONNX", "OMNIASR_TOKENIZER"]
    if not all(os.environ.get(name) for name in names):
        pytest.skip("Set OMNIASR_ONNX and OMNIASR_TOKENIZER (OMNIASR_TEST_WAV optional)")
    return [os.environ[name] for name in names] + [str(clip)]


def test_real_transcript_without_training_frameworks(assets, expect, local_model_key):
    expected = expect.transcript(assets[2], local_model_key, "onnx-cpu-fp32")
    code = """
import json
import sys
import soundfile as sf
from fast_omniasr import OmniASR
model = OmniASR(sys.argv[1], sys.argv[2])
result = model.transcribe(sys.argv[3])
samples = sf.info(sys.argv[3]).frames
assert result.logits_shape == (1, (samples - 400) // 320 + 1, int(sys.argv[4])), result
expected = json.loads(sys.argv[5])
if expected is not None:
    assert result.text == expected, result.text
assert result.text.strip(), "empty transcript"
assert not any(name in sys.modules for name in ['torch', 'fairseq2', 'transformers'])
"""
    subprocess.run([sys.executable, "-c", code, *assets, str(expect.vocabulary(assets[1])),
                    json.dumps(expected)], check=True)


def test_dynamic_frame_counts(assets, expect):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1])
    vocabulary = expect.vocabulary(assets[1])
    for n in [400, 16000, 80000, 117920, 240000, 480000]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, expect.frames(n), vocabulary)


def test_wav_and_numpy_agree(assets):
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1])
    waveform, rate = sf.read(assets[2], dtype="float32")
    assert model.transcribe(assets[2]) == model.transcribe_numpy(waveform, rate)


def test_short_inputs_on_cuda(assets, expect):
    """cuDNN 9.27+ asked for a ~17.6 GB convolution workspace for every input under ~2.6 s
    under ORT's default algorithm search, so these lengths failed on CUDA (#97)."""
    ort = pytest.importorskip("onnxruntime")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("needs onnxruntime-gpu with a CUDA device")
    from fast_omniasr import OmniASR
    model = OmniASR(assets[0], assets[1], device="cuda")
    cpu = OmniASR(assets[0], assets[1])
    vocabulary = expect.vocabulary(assets[1])
    for n in [400, 16000, 33226, 39404, 40886, 41619]:
        result = model.transcribe_numpy(np.zeros(n, dtype=np.float32))
        assert result.logits_shape == (1, expect.frames(n), vocabulary)
    waveform, rate = sf.read(assets[2], dtype="float32")
    for seconds in (1, 2):
        clip = waveform[: seconds * rate]
        assert model.transcribe_numpy(clip, rate).text == cpu.transcribe_numpy(clip, rate).text

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ID = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"


@pytest.fixture
def wav():
    if not os.environ.get("OMNIASR_TEST_WAV"):
        pytest.skip("Set OMNIASR_TEST_WAV")
    return os.environ["OMNIASR_TEST_WAV"]


def test_from_pretrained_transcribes(wav):
    from fast_omniasr import OmniASR
    model = OmniASR.from_pretrained(REPO_ID, backend="onnx")
    result = model.transcribe(wav)
    assert result.text == "now i want to return to the conservation of mechanical energy"


def test_from_pretrained_offline_after_first_download(wav, monkeypatch):
    from fast_omniasr import OmniASR
    OmniASR.from_pretrained(REPO_ID, backend="onnx")  # populate the cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = OmniASR.from_pretrained(REPO_ID, backend="onnx")
    result = model.transcribe(wav)
    assert result.text == "now i want to return to the conservation of mechanical energy"


def test_from_pretrained_rejects_non_onnx_backend():
    from fast_omniasr import OmniASR
    with pytest.raises(ValueError, match="backend='onnx' only"):
        OmniASR.from_pretrained(REPO_ID, backend="tensorrt")


def test_from_pretrained_detects_corrupted_asset(tmp_path):
    from huggingface_hub import hf_hub_download

    from fast_omniasr import OmniASR
    path = hf_hub_download(REPO_ID, "tokenizer.model", cache_dir=tmp_path)
    with open(os.path.realpath(path), "r+b") as f:
        f.write(b"corrupted")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        OmniASR.from_pretrained(REPO_ID, backend="onnx", cache_dir=tmp_path)

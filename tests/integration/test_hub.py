import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ID = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"


@pytest.fixture
def network():
    if os.environ.get("OMNIASR_HUB_TESTS") != "1" and not os.environ.get("OMNIASR_TEST_WAV"):
        pytest.skip("Set OMNIASR_HUB_TESTS=1 (or OMNIASR_TEST_WAV) to download from the Hub")


def _check(model, clip, expect):
    result = model.transcribe(clip)
    expected = expect.transcript(clip, expect.published(model.snapshot), "onnx-cpu-fp32")
    if expected is not None:
        assert result.text == expected
    assert result.text.strip()
    return result


def test_from_pretrained_transcribes(network, clip, expect):
    from fast_omniasr import OmniASR
    _check(OmniASR.from_pretrained(REPO_ID, backend="onnx"), clip, expect)


def test_from_pretrained_offline_after_first_download(network, clip, expect, monkeypatch):
    from fast_omniasr import OmniASR
    OmniASR.from_pretrained(REPO_ID, backend="onnx")  # populate the cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _check(OmniASR.from_pretrained(REPO_ID, backend="onnx"), clip, expect)
    _check(OmniASR.from_pretrained(REPO_ID, backend="onnx", local_files_only=True), clip, expect)


def test_from_pretrained_tensorrt_requires_precision():
    from fast_omniasr import OmniASR
    with pytest.raises(ValueError, match="requires precision"):
        OmniASR.from_pretrained(REPO_ID, backend="tensorrt")


def test_from_pretrained_detects_corrupted_asset(network, tmp_path, monkeypatch):
    """A real download whose tokenizer arrives corrupted fails the checksum check, and the
    error explains how to recover."""
    import huggingface_hub

    from fast_omniasr import OmniASR
    real = huggingface_hub.hf_hub_download

    def corrupting(repo_id, filename, **kwargs):
        path = real(repo_id, filename, **kwargs)
        if filename == "tokenizer.model":
            with open(path, "ab") as f:
                f.write(b"corrupted")
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", corrupting)
    with pytest.raises(RuntimeError, match="checksum mismatch.*delete_cached"):
        OmniASR.from_pretrained(REPO_ID, backend="onnx", cache_dir=tmp_path)

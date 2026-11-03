
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ID = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"


def _check(model, clip, expect):
    result = model.transcribe(clip)
    key = expect.published(model.snapshot)
    expected = expect.transcript(clip, key, "onnx-cpu-fp32")
    if expected is not None:
        assert result.text == expected
    declared = expect.declared(clip, key)
    if declared is not None:
        assert result.logits_shape == (1, declared["frames"], declared["vocabulary_size"])
    assert result.text.strip()
    return result


def test_from_pretrained_transcribes(network, clip, expect):
    from fast_omniasr import OmniASR
    model = OmniASR.from_pretrained(REPO_ID, backend="onnx", revision=expect.revision())
    _check(model, clip, expect)


def test_from_pretrained_offline_after_first_download(network, clip, expect, monkeypatch):
    from fast_omniasr import OmniASR
    revision = expect.revision()
    OmniASR.from_pretrained(REPO_ID, backend="onnx", revision=revision)  # populate the cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    _check(OmniASR.from_pretrained(REPO_ID, backend="onnx", revision=revision), clip, expect)
    _check(OmniASR.from_pretrained(REPO_ID, backend="onnx", revision=revision,
                                   local_files_only=True), clip, expect)


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

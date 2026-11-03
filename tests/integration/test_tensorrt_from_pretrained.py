import os
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.cuda]

REPO_ID = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"


@pytest.fixture
def network():
    if os.environ.get("OMNIASR_HUB_TESTS") != "1" and not os.environ.get("OMNIASR_TEST_WAV"):
        pytest.skip("Set OMNIASR_HUB_TESTS=1 (or OMNIASR_TEST_WAV) to download from the Hub")


def _expected(model, clip, expect, precision):
    return expect.transcript(clip, expect.published(model.snapshot), f"tensorrt-{precision}")


def test_default_profile_and_engine_cache(network, clip, expect, tmp_path):
    """from_pretrained's default profile (400..480,000 samples): download -> verify ->
    build -> cache -> transcribe, then a second construction reuses the cached engine."""
    from fast_omniasr import OmniASR
    from fast_omniasr.tensorrt_builder import DEFAULT_MAX_SAMPLES, DEFAULT_MIN_SAMPLES

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = OmniASR.from_pretrained(
            REPO_ID, backend="tensorrt", precision="fp16", engine_cache_dir=tmp_path,
        )
    assert any("FP16" in str(w.message) for w in caught)
    assert (model.backend.min_samples, model.backend.max_samples) == (
        DEFAULT_MIN_SAMPLES, DEFAULT_MAX_SAMPLES)
    vocabulary = model.backend.vocabulary_size
    for n in (DEFAULT_MIN_SAMPLES, 8000, DEFAULT_MAX_SAMPLES):  # 8000 is inside the default
        assert model.transcribe_numpy(np.zeros(n, dtype=np.float32)).logits_shape == (
            1, expect.frames(n), vocabulary)
    with pytest.raises(ValueError, match="engine profile"):
        model.transcribe_numpy(np.zeros(DEFAULT_MAX_SAMPLES + 1, dtype=np.float32))
    result = model.transcribe(clip)
    expected = _expected(model, clip, expect, "fp16")
    if expected is not None:
        assert result.text == expected

    engine_files_before = sorted(tmp_path.rglob("model.engine"))
    assert len(engine_files_before) == 1
    mtime_before = engine_files_before[0].stat().st_mtime

    second = OmniASR.from_pretrained(
        REPO_ID, backend="tensorrt", precision="fp16", engine_cache_dir=tmp_path,
    )
    assert second.transcribe(clip) == model.transcribe(clip)
    engine_files_after = sorted(tmp_path.rglob("model.engine"))
    assert len(engine_files_after) == 1
    assert engine_files_after[0].stat().st_mtime == mtime_before  # not rebuilt


def test_custom_profile_is_enforced(network, expect, tmp_path):
    """A custom profile narrows the accepted lengths; the engine rejects the rest."""
    from fast_omniasr import OmniASR
    model = OmniASR.from_pretrained(
        REPO_ID, backend="tensorrt", precision="fp16", engine_cache_dir=tmp_path,
        min_samples=16000, opt_samples=32000, max_samples=64000,
    )
    assert (model.backend.min_samples, model.backend.max_samples) == (16000, 64000)
    assert model.transcribe_numpy(np.zeros(16000, np.float32)).logits_shape[1] == expect.frames(16000)
    for n in (8000, 64001):
        with pytest.raises(ValueError, match="engine profile"):
            model.transcribe_numpy(np.zeros(n, dtype=np.float32))


def test_fp32_matches_reference(network, clip, expect, tmp_path):
    from fast_omniasr import OmniASR
    model = OmniASR.from_pretrained(
        REPO_ID, backend="tensorrt", precision="fp32", engine_cache_dir=tmp_path,
    )
    result = model.transcribe(clip)
    expected = _expected(model, clip, expect, "fp32")
    assert expected is not None or os.environ.get("OMNIASR_TEST_WAV"), \
        "the default clip must have a declared fp32 transcript"
    if expected is not None:
        assert result.text == expected

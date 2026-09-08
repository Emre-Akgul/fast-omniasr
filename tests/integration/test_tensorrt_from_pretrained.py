import os
import warnings

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.cuda]

REPO_ID = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"
KNOWN_TRANSCRIPT = "now i want to return to the conservation of mechanical energy"


@pytest.fixture
def wav():
    if not os.environ.get("OMNIASR_TEST_WAV"):
        pytest.skip("Set OMNIASR_TEST_WAV")
    return os.environ["OMNIASR_TEST_WAV"]


def test_first_call_builds_second_call_reuses_cache(wav, tmp_path):
    """The Day 7 success criterion: download -> verify -> build -> cache -> transcribe,
    then a second construction hits the cache instead of rebuilding."""
    from fast_omniasr import OmniASR

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = OmniASR.from_pretrained(
            REPO_ID, backend="tensorrt", precision="fp16", engine_cache_dir=tmp_path,
        )
    assert any("FP16" in str(w.message) for w in caught)
    result = model.transcribe(wav)
    assert result.text == KNOWN_TRANSCRIPT

    engine_files_before = sorted(tmp_path.rglob("model.engine"))
    assert len(engine_files_before) == 1
    mtime_before = engine_files_before[0].stat().st_mtime

    second = OmniASR.from_pretrained(
        REPO_ID, backend="tensorrt", precision="fp16", engine_cache_dir=tmp_path,
    )
    assert second.transcribe(wav) == model.transcribe(wav)
    engine_files_after = sorted(tmp_path.rglob("model.engine"))
    assert len(engine_files_after) == 1
    assert engine_files_after[0].stat().st_mtime == mtime_before  # not rebuilt


def test_fp32_matches_reference(wav, tmp_path):
    from fast_omniasr import OmniASR
    model = OmniASR.from_pretrained(
        REPO_ID, backend="tensorrt", precision="fp32", engine_cache_dir=tmp_path,
    )
    assert model.transcribe(wav).text == KNOWN_TRANSCRIPT

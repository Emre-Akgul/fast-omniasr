"""Real published models, end to end on CPU, against committed FLEURS fixtures.

Opt-in (downloads ~8 GB): OMNIASR_REAL_MODELS=1 pytest tests/integration/test_real_models_cpu.py
Model repos/revisions come from tests/fixtures/fleurs/manifest.json and can be overridden
with OMNIASR_CTC_REPO / OMNIASR_CTC_REVISION / OMNIASR_LLM_REPO / OMNIASR_LLM_REVISION;
OMNIASR_CACHE_DIR overrides the download cache (default ~/.cache/fast-omniasr/hub).
"""

import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("OMNIASR_REAL_MODELS") != "1",
                       reason="set OMNIASR_REAL_MODELS=1 to download and run the real models"),
]

_spec = importlib.util.spec_from_file_location("evaluate_precisions",
                                               ROOT / "tools" / "evaluate_precisions.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)


def _model(kind):
    pinned = MANIFEST["models"][kind]
    repo = os.environ.get(f"OMNIASR_{kind.upper()}_REPO", pinned["repo"])
    revision = os.environ.get(f"OMNIASR_{kind.upper()}_REVISION", pinned["revision"])
    return repo, revision


def _cache_dir():
    return os.environ.get("OMNIASR_CACHE_DIR") or None


def _cer(reference, hypothesis):
    return _metrics.error_rates([reference], [hypothesis])["cer"]


def _audio(clip):
    waveform, sample_rate = sf.read(FIXTURES / clip["file"], dtype="float32")
    return waveform, sample_rate


@pytest.fixture(scope="module")
def ctc():
    from fast_omniasr import OmniASR

    repo, revision = _model("ctc")
    with OmniASR.from_pretrained(repo, revision=revision, device="cpu",
                                 cache_dir=_cache_dir()) as model:
        yield model


@pytest.fixture(scope="module")
def llm():
    from fast_omniasr.llm import OmniASRLLM

    repo, revision = _model("llm")
    with OmniASRLLM.from_pretrained(repo, revision=revision, device="cpu",
                                 cache_dir=_cache_dir()) as model:
        yield model


@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda c: c["fleurs_language"])
def test_ctc_transcribes_fleurs_clip(ctc, clip):
    waveform, sample_rate = _audio(clip)
    first = ctc.transcribe_numpy(waveform, sample_rate)
    again = ctc.transcribe_numpy(waveform, sample_rate)
    assert first.text == again.text and first.token_ids == again.token_ids  # deterministic
    cer = _cer(clip["reference"], first.text)
    assert cer <= clip["max_cer"]["ctc"], f"CTC CER {cer:.1f} on {clip['file']}: {first.text!r}"


@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda c: c["fleurs_language"])
def test_llm_transcribes_fleurs_clip(llm, clip):
    waveform, sample_rate = _audio(clip)
    result = llm.transcribe(waveform, sample_rate=sample_rate, lang=clip["lang"])
    assert result.stop_reason == "eos"
    cer = _cer(clip["reference"], result.text)
    assert cer <= clip["max_cer"]["llm"], f"LLM CER {cer:.1f} on {clip['file']}: {result.text!r}"


def test_ctc_handles_silence_and_rejects_use_after_close():
    from fast_omniasr import OmniASR

    repo, revision = _model("ctc")
    model = OmniASR.from_pretrained(repo, revision=revision, device="cpu",
                                 cache_dir=_cache_dir())
    assert model.transcribe_numpy(np.zeros(16000, np.float32)).text.strip() == ""
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe_numpy(np.zeros(16000, np.float32))


def test_llm_warns_without_language_but_still_transcribes(llm):
    clip = MANIFEST["clips"][0]
    waveform, sample_rate = _audio(clip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = llm.transcribe(waveform, sample_rate=sample_rate)
    assert any("lang" in str(w.message) for w in caught)
    assert result.text.strip()

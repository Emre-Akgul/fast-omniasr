"""TensorRT engines of the published CTC model, at every precision, on the FLEURS fixtures.

Opt-in (GPU + TensorRT, downloads ~1.3 GB, builds three engines):
  OMNIASR_REAL_MODELS=1 pytest tests/integration/test_tensorrt_fleurs.py
"""

import importlib.util
import json
import os
import warnings
from pathlib import Path

import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "fleurs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.cuda,
    pytest.mark.skipif(os.environ.get("OMNIASR_REAL_MODELS") != "1",
                       reason="set OMNIASR_REAL_MODELS=1 to build and run real TensorRT engines"),
]

_spec = importlib.util.spec_from_file_location("evaluate_precisions",
                                               ROOT / "tools" / "evaluate_precisions.py")
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)


def _load(precision):
    pytest.importorskip("tensorrt")
    from fast_omniasr import OmniASR

    pinned = MANIFEST["models"]["ctc"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="TensorRT FP16")
        return OmniASR.from_pretrained(
            pinned["repo"], revision=pinned["revision"], backend="tensorrt",
            precision=precision, cache_dir=os.environ.get("OMNIASR_CACHE_DIR") or None,
        )


@pytest.fixture(scope="module", params=["fp32", "fp16", "mixed"])
def engine(request):
    with _load(request.param) as model:
        yield request.param, model


@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda c: c["fleurs_language"])
def test_engine_transcribes_fleurs_clip(engine, clip):
    precision, model = engine
    waveform, sample_rate = sf.read(FIXTURES / clip["file"], dtype="float32")
    first = model.transcribe_numpy(waveform, sample_rate)
    assert model.transcribe_numpy(waveform, sample_rate).token_ids == first.token_ids
    cer = _metrics.error_rates([clip["reference"]], [first.text])["cer"]
    # The same ceilings as the CPU FP32 run: every precision must stay within them.
    assert cer <= clip["max_cer"]["ctc"], f"{precision} CER {cer:.1f}: {first.text!r}"


def test_mixed_matches_fp32_on_the_fixtures():
    """The point of 'mixed': on these clips it decodes exactly like FP32 (pure FP16 need
    not), so a recipe regression to FP16-level behaviour shows up here."""
    texts = {}
    for precision in ("fp32", "mixed"):
        with _load(precision) as model:
            texts[precision] = [
                model.transcribe_numpy(*sf.read(FIXTURES / c["file"], dtype="float32")).token_ids
                for c in MANIFEST["clips"]]
    assert texts["mixed"] == texts["fp32"]

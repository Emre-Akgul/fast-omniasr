"""TensorRT engines of the published CTC model, at every precision, on the FLEURS fixtures.

Opt-in (GPU + TensorRT, downloads ~1.3 GB, builds three engines):
  OMNIASR_REAL_MODELS=1 pytest tests/integration/test_tensorrt_fleurs.py
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
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))

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


def _decision_logit_error(logits, reference):
    """Largest error on the FP32-chosen token's logit over all frames: what decides the
    greedy CTC output (tail logits of the 10,288-entry vocabulary don't)."""
    chosen = reference[0].argmax(-1)
    frames = np.arange(len(chosen))
    return float(np.abs(logits[0][frames, chosen] - reference[0][frames, chosen]).max())


def test_mixed_tracks_fp32_much_closer_than_fp16():
    """'mixed' must decode like FP32 on the fixtures and keep the decision logits much
    closer to FP32 than plain FP16 does (about 6x on these clips). If the recipe regressed
    to plain FP16 (pins ignored, OBEY dropped), the two errors would be equal and this
    fails; plain FP16 also decodes these three clips like FP32, so token equality alone
    could not tell them apart."""
    from fast_omniasr.audio import prepare_audio

    logits, tokens = {}, {}
    for precision in ("fp32", "fp16", "mixed"):
        with _load(precision) as model:
            audio = [prepare_audio(*sf.read(FIXTURES / c["file"], dtype="float32"))
                     for c in MANIFEST["clips"]]
            logits[precision] = [model.backend.infer(x) for x in audio]
            tokens[precision] = [model.transcribe_numpy(x[0]).token_ids for x in audio]
    assert tokens["mixed"] == tokens["fp32"]
    error = {p: sum(_decision_logit_error(a, b) for a, b in zip(logits[p], logits["fp32"]))
             for p in ("fp16", "mixed")}
    assert error["mixed"] < 0.5 * error["fp16"], error

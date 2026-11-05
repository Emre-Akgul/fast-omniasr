"""The published models on edge-case audio, on CPU: output and LLM stop reasons (#32).

Opt-in (downloads ~8 GB):
  OMNIASR_REAL_MODELS=1 pytest tests/integration/test_real_models_edge_cases.py
Clips and expectations: tests/fixtures/edge_cases (its README says what is recorded behaviour
rather than a target). Models and overrides as in test_real_models_cpu.py.
"""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PINNED = json.loads((ROOT / "tests/fixtures/fleurs/manifest.json").read_text(encoding="utf-8"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("OMNIASR_REAL_MODELS") != "1",
                       reason="set OMNIASR_REAL_MODELS=1 to download and run the real models"),
]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clips = _load("edge_case_clips", ROOT / "tests/fixtures/edge_cases/clips.py")
_metrics = _load("evaluate_precisions", ROOT / "tools/evaluate_precisions.py")
MANIFEST = clips.manifest()


def _options(kind):
    pinned = PINNED["models"][kind]
    repo = os.environ.get(f"OMNIASR_{kind.upper()}_REPO", pinned["repo"])
    return repo, {
        "revision": os.environ.get(f"OMNIASR_{kind.upper()}_REVISION", pinned["revision"]),
        "device": "cpu",
        "cache_dir": os.environ.get("OMNIASR_CACHE_DIR") or None,
    }


def _cer(reference, text):
    return _metrics.error_rates([reference], [text])["cer"]


@pytest.fixture(scope="module")
def ctc():
    from fast_omniasr import OmniASR

    repo, options = _options("ctc")
    with OmniASR.from_pretrained(repo, **options) as model:
        yield model


@pytest.fixture(scope="module")
def llm():
    from fast_omniasr.llm import OmniASRLLM

    repo, options = _options("llm")
    with OmniASRLLM.from_pretrained(repo, **options) as model:
        yield model


@pytest.mark.parametrize("name", MANIFEST["clips"])
def test_ctc_on_edge_case(ctc, name):
    entry = MANIFEST["clips"][name]
    waveform = clips.waveform(name)
    result = ctc.transcribe_numpy(waveform)
    assert ctc.transcribe_numpy(waveform).token_ids == result.token_ids  # deterministic
    clips.check_text(result.text, entry["ctc"], clips.reference(entry), _cer)


def test_ctc_length_limits(ctc):
    with pytest.raises(ValueError, match="At least 400"):
        ctc.transcribe_numpy(np.zeros(clips.MIN_SAMPLES - 1, np.float32))
    # 30 seconds limits the LLM and TensorRT's default profile; ONNX CTC has no upper limit.
    assert ctc.transcribe_numpy(np.zeros(clips.MAX_SAMPLES + 1, np.float32)).text == ""


@pytest.mark.parametrize("name", MANIFEST["clips"])
def test_llm_on_edge_case(llm, name):
    entry = MANIFEST["clips"][name]
    expected = entry["llm"]
    budget = expected.get("max_new_tokens", 512)
    result = llm.transcribe(clips.waveform(name), lang=MANIFEST["language"],
                            max_new_tokens=budget)
    assert result.stop_reason == expected["stop_reason"], result.text
    if expected["stop_reason"] == "max_new_tokens":
        assert len(result.token_ids) == budget
    else:
        assert len(result.token_ids) < budget
    clips.check_text(result.text, expected, clips.reference(entry), _cer)


def test_llm_budget_cuts_off_speech(llm):
    result = llm.transcribe(clips.waveform("clipped_speech"), lang=MANIFEST["language"],
                            max_new_tokens=4)
    assert result.stop_reason == "max_new_tokens" and len(result.token_ids) == 4


@pytest.mark.parametrize("samples,message", [
    (clips.MIN_SAMPLES - 1, "At least 400"),
    (clips.MAX_SAMPLES + 1, "at most 30 seconds"),
])
def test_llm_length_limits(llm, samples, message):
    with pytest.raises(ValueError, match=message):
        llm.transcribe(np.zeros(samples, np.float32), lang=MANIFEST["language"])

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


def test_llm_repetition_penalty_ends_the_loop_on_silence(llm):
    result = llm.transcribe(clips.waveform("silence"), lang=MANIFEST["language"],
                            repetition_penalty=1.2)
    assert result.stop_reason == "eos" and len(result.token_ids) < 64


def test_llm_repetition_penalty_keeps_legitimately_repeated_speech(llm):
    entry = MANIFEST["clips"]["repeated_sentence"]
    result = llm.transcribe(clips.waveform("repeated_sentence"), lang=MANIFEST["language"],
                            repetition_penalty=1.2)
    assert result.stop_reason == "eos"
    clips.check_text(result.text, entry["llm"], clips.reference(entry), _cer)


def test_ctc_long_audio_keeps_both_languages_in_order(ctc):
    result = ctc.transcribe_long(clips.waveform("longest_clip"))
    assert [chunk.end for chunk in result.chunks] == [15.0, 30.0]
    assert "people" not in result.text.split("sundular")[-1]
    assert "sundular" in result.text


def test_llm_long_audio_reports_each_chunk(llm):
    result = llm.transcribe_long(clips.waveform("longest_clip"), lang=MANIFEST["language"])
    assert [chunk.end for chunk in result.chunks] == [15.0, 30.0]
    assert result.stop_reason in ("eos", "max_new_tokens")
    assert result.text == " ".join(c.text for c in result.chunks if c.text)


@pytest.mark.parametrize("name", ["silence", "white_noise", "mains_hum"])
def test_vad_stops_the_llm_inventing_text_without_speech(llm, name):
    result = llm.transcribe_long(clips.waveform(name), lang=MANIFEST["language"], vad_filter=True)
    assert result.text == "" and result.chunks == ()


def test_vad_keeps_speech_and_original_times(ctc):
    silence = np.zeros(10 * clips.SAMPLE_RATE, np.float32)
    clip = clips.waveform("clipped_speech")
    result = ctc.transcribe_long(np.concatenate([silence, clip]), vad_filter=True)
    assert len(result.chunks) == 1 and 9.5 < result.chunks[0].start < 11
    entry = MANIFEST["clips"]["clipped_speech"]
    clips.check_text(result.text, entry["ctc"], clips.reference(entry), _cer)


def test_ctc_word_times_follow_the_audio(ctc):
    audio = clips.waveform("clipped_speech")
    plain = ctc.transcribe_numpy(audio)
    timed = ctc.transcribe_numpy(audio, word_timestamps=True)
    assert (timed.text, timed.token_ids) == (plain.text, plain.token_ids)
    assert " ".join(word.word for word in timed.words) == " ".join(plain.text.split())
    assert all(word.start < word.end for word in timed.words)
    assert timed.words[-1].end <= audio.size / 16000
    delayed = ctc.transcribe_numpy(np.r_[np.zeros(3 * 16000, audio.dtype), audio],
                                   word_timestamps=True)
    # Silence can change the model's wording, so compare the words both runs share.
    shared = list(zip(timed.words[:4], delayed.words[:4]))
    assert all(before.word == after.word for before, after in shared)
    assert all(abs(after.start - before.start - 3) <= 0.06 for before, after in shared)


def test_ctc_word_times_survive_vad(ctc):
    audio = clips.waveform("clipped_speech")
    gap = 5 * 16000
    result = ctc.transcribe_long(np.r_[audio, np.zeros(gap, audio.dtype), audio],
                                 vad_filter=True, word_timestamps=True)
    alone = ctc.transcribe_numpy(audio, word_timestamps=True)
    resumed = next(word for word in result.words if word.start > audio.size / 16000)
    assert resumed.word == alone.words[0].word
    assert abs(resumed.start - (audio.size + gap) / 16000 - alone.words[0].start) <= 0.1


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

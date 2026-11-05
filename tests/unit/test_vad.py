"""Silero VAD chunking, as in faster-whisper (#36)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fast_omniasr.audio import prepare_chunks
from fast_omniasr.vad import VadOptions, speech_chunks

FLEURS = Path(__file__).resolve().parents[1] / "fixtures" / "fleurs"
MANIFEST = json.loads((FLEURS / "manifest.json").read_text(encoding="utf-8"))


def speech(index=1):
    """A real FLEURS sentence at a normal level, about four seconds long."""
    waveform, _ = sf.read(FLEURS / MANIFEST["clips"][index]["file"], dtype="float32")
    return waveform / max(np.sqrt(np.mean(waveform ** 2)), 1e-6) * 0.03


@pytest.fixture(autouse=True)
def needs_onnxruntime():
    pytest.importorskip("onnxruntime")


def test_digital_silence_has_no_chunks():
    assert prepare_chunks(np.zeros(5 * 16000, np.float32), vad_filter=True) == []


def test_nonspeech_noise_has_no_chunks():
    noise = np.random.default_rng(0).normal(0, 1e-4, 5 * 16000).astype(np.float32)
    assert prepare_chunks(noise, vad_filter=True) == []


def test_speech_keeps_its_original_times_after_leading_silence():
    waveform = np.concatenate([np.zeros(10 * 16000, np.float32), speech()])
    chunks = prepare_chunks(waveform, vad_filter=True)
    assert len(chunks) == 1
    start, end, audio = chunks[0]
    assert 9.5 < start < 10.8
    assert end == pytest.approx(waveform.size / 16000, abs=0.5)
    assert audio.shape[1] < 6 * 16000  # the silence isn't in the chunk


def test_long_silence_between_sentences_is_skipped():
    gap = np.zeros(20 * 16000, np.float32)
    waveform = np.concatenate([speech(), gap, speech(1)])
    chunks = prepare_chunks(waveform, vad_filter=True, chunk_length=15)
    assert len(chunks) == 1  # both fit one 15 s chunk, joined without the silence
    assert chunks[0][1] > 20 and chunks[0][2].shape[1] < 11 * 16000


def test_chunks_stay_within_chunk_length():
    gap = np.zeros(8000, np.float32)
    waveform = np.concatenate([part for _ in range(8) for part in (speech(), gap)])
    chunks = prepare_chunks(waveform, vad_filter=True, chunk_length=10)
    assert len(chunks) > 2
    assert all(audio.shape[1] <= 10 * 16000 for *_, audio in chunks)
    assert [start for start, *_ in chunks] == sorted(start for start, *_ in chunks)


def test_cuts_land_in_silence():
    gap = np.zeros(8000, np.float32)
    clip = speech()
    waveform = np.concatenate([part for _ in range(6) for part in (clip, gap)])
    period = clip.size + gap.size
    for start, end, _ in prepare_chunks(waveform, vad_filter=True, chunk_length=10):
        boundary = (end * 16000) % period
        assert boundary > 56000 or boundary < 2000  # this sentence's speech ends near 56,000


def test_vad_options_are_used_and_chunk_length_still_caps_speech():
    spans = speech_chunks(speech(), 30, VadOptions(threshold=0.99, min_speech_duration_ms=30000))
    assert spans == []


def test_vad_options_need_vad_filter():
    with pytest.raises(ValueError, match="vad_filter=True"):
        prepare_chunks(speech(), vad_options=VadOptions())


def test_vad_filter_must_be_a_bool():
    with pytest.raises(TypeError, match="vad_filter"):
        prepare_chunks(speech(), vad_filter=1)


def test_non_finite_audio_is_rejected_before_vad():
    waveform = speech()
    waveform[10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        prepare_chunks(waveform, vad_filter=True)


def test_missing_onnxruntime_says_what_to_install(monkeypatch):
    import sys

    from fast_omniasr import vad

    vad.get_vad_model.cache_clear()
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"fast-omniasr\[onnx\]"):
        prepare_chunks(speech(), vad_filter=True)
    vad.get_vad_model.cache_clear()

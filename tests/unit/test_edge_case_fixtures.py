"""The edge-case clips rebuild exactly and really contain the edge case they name (#32)."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from fast_omniasr.audio import MIN_AUDIO_SAMPLES, prepare_audio
from fast_omniasr.llm.runtime import STOP_REASONS
from fast_omniasr.tensorrt_builder import DEFAULT_MAX_SAMPLES

_path = Path(__file__).resolve().parents[1] / "fixtures" / "edge_cases" / "clips.py"
_spec = importlib.util.spec_from_file_location("edge_case_clips", _path)
clips = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clips)

MANIFEST = clips.manifest()


def test_manifest_lists_every_recipe():
    assert set(MANIFEST["clips"]) == set(clips.RECIPES)


@pytest.mark.parametrize("name", MANIFEST["clips"])
def test_clip_matches_manifest(name):
    pcm, entry = clips.build(name), MANIFEST["clips"][name]
    assert pcm.dtype == np.int16 and pcm.ndim == 1
    assert pcm.size == entry["samples"]
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == entry["sha256"]


@pytest.mark.parametrize("name", MANIFEST["clips"])
def test_clip_is_accepted_and_normalizes_to_finite_audio(name):
    assert np.isfinite(prepare_audio(clips.waveform(name))).all()


@pytest.mark.parametrize("name", MANIFEST["clips"])
def test_expectations_are_well_formed(name):
    entry = MANIFEST["clips"][name]
    assert entry["llm"]["stop_reason"] in STOP_REASONS
    assert set(entry["ctc"]) <= {"text", "max_cer", "contains"}
    assert set(entry["llm"]) <= {"stop_reason", "max_new_tokens", "max_cer", "contains"}
    if "max_cer" in entry["ctc"] or "max_cer" in entry["llm"]:
        assert clips.reference(entry).strip()


def test_length_clips_sit_exactly_on_the_limits():
    assert clips.MIN_SAMPLES == MIN_AUDIO_SAMPLES
    assert clips.MAX_SAMPLES == DEFAULT_MAX_SAMPLES
    assert clips.build("shortest_clip").size == MIN_AUDIO_SAMPLES
    assert clips.build("longest_clip").size == DEFAULT_MAX_SAMPLES
    prepare_audio(clips.waveform("shortest_clip"))
    with pytest.raises(ValueError, match="At least 400"):
        prepare_audio(clips.waveform("shortest_clip")[:-1])


def test_no_speech_clips_have_the_expected_levels():
    assert not clips.build("silence").any()
    for name in ("white_noise", "mains_hum"):
        assert np.sqrt(np.mean(clips.waveform(name) ** 2)) > 0.05


def test_clipped_clip_is_really_clipped():
    pcm = clips.build("clipped_speech").astype(np.int32)
    assert np.mean(np.abs(pcm) >= 32767) > 0.2


def test_code_switching_halves_are_level_matched():
    pcm = clips.waveform("code_switching")
    english, turkish = pcm[:69120], pcm[69120 + 8000:]
    gap_db = 20 * np.log10(np.sqrt(np.mean(english ** 2)) / np.sqrt(np.mean(turkish ** 2)))
    assert abs(gap_db) < 1
    assert not pcm[69120:69120 + 8000].any()


def test_long_pause_is_faint_and_inside_the_sentence():
    pcm, middle = clips.waveform("long_pause"), 69120 // 2
    pause = pcm[middle:middle + 8 * 16000]
    assert 0 < np.sqrt(np.mean(pause ** 2)) < 2e-4
    assert np.abs(pcm[:middle]).max() > 0.05 and np.abs(pcm[middle + 8 * 16000:]).max() > 0.05


def test_repetition_clips_repeat():
    repeated = clips.build("repeated_sentence")
    period = repeated.size // 4
    assert np.array_equal(repeated[:period], repeated[period:2 * period])
    stutter = clips.build("stutter")
    np.testing.assert_array_equal(stutter[:4000], stutter[4000:8000])

import sys

import numpy as np
import pytest
import soundfile as sf

from fast_omniasr import OmniASR
from fast_omniasr.audio import load_audio, prepare_audio


def test_normalization_and_input_unchanged():
    source = np.tile(np.array([1, 3], dtype=np.float32), 400)
    original = source.copy()
    result = prepare_audio(source)
    assert result.shape == (1, 800)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(source, original)
    assert abs(result.mean()) < 1e-7
    assert result.var() == pytest.approx(1 / (1 + 1e-5), abs=1e-6)


def test_silence_is_finite():
    np.testing.assert_array_equal(prepare_audio(np.ones(16000)), np.zeros((1,16000)))


@pytest.mark.parametrize("wave,rate", [(np.zeros(399),16000), (np.zeros((800,2)),16000),
                                      (np.zeros(800),8000), (np.full(800,np.nan),16000)])
def test_invalid_inputs(wave, rate):
    with pytest.raises(ValueError):
        prepare_audio(wave, rate)


def _tone(rate, seconds=1.0, hertz=1000.0):
    time = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * hertz * time)).astype(np.float32)


@pytest.mark.parametrize("wave,rate", [(np.zeros(44100), 44100), (np.zeros((16000, 2)), 16000)])
def test_strict_by_default_and_points_to_the_option(wave, rate):
    with pytest.raises(ValueError, match="convert_audio=True"):
        prepare_audio(wave, rate)


def test_conversion_leaves_mono_16k_audio_identical():
    source = np.random.default_rng(0).normal(size=16000).astype(np.float32)
    np.testing.assert_array_equal(prepare_audio(source, 16000, convert_audio=True),
                                  prepare_audio(source, 16000))


def test_downmix_averages_channels_without_resampler(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)  # 16 kHz input never needs it
    left = np.random.default_rng(1).normal(size=16000).astype(np.float32)
    right = np.random.default_rng(2).normal(size=16000).astype(np.float32)
    converted = prepare_audio(np.stack([left, right], axis=1), convert_audio=True)
    np.testing.assert_array_almost_equal(converted, prepare_audio((left + right) / 2))


def test_opposite_channels_cancel_to_silence():
    tone = _tone(16000)
    converted = prepare_audio(np.stack([tone, -tone], axis=1), convert_audio=True)
    np.testing.assert_array_equal(converted, np.zeros((1, 16000)))


@pytest.mark.parametrize("rate", [8000, 22050, 44100, 48000])
def test_resampling_keeps_length_and_pitch(rate):
    pytest.importorskip("soxr")
    converted = prepare_audio(_tone(rate), rate, convert_audio=True)
    assert converted.shape == (1, 16000)
    spectrum = np.abs(np.fft.rfft(converted[0]))
    assert abs(spectrum.argmax() - 1000) <= 1  # one second, so bins are 1 Hz wide


def test_downmixes_then_resamples():
    pytest.importorskip("soxr")
    tone = _tone(48000)
    stereo = np.stack([tone, tone], axis=1)
    assert prepare_audio(stereo, 48000, convert_audio=True).shape == (1, 16000)


def test_missing_resampler_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)
    with pytest.raises(ImportError, match=r"fast-omniasr\[resample\]"):
        prepare_audio(np.zeros(44100), 44100, convert_audio=True)


@pytest.mark.parametrize("rate", [0, -8000, True, 44100.0, "44100", None])
def test_conversion_rejects_bad_sample_rates(rate):
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        prepare_audio(np.zeros(800), rate, convert_audio=True)


@pytest.mark.parametrize("shape", [(2, 16000), (16000, 0), (0, 2)])
def test_conversion_rejects_ambiguous_channel_layouts(shape):
    with pytest.raises(ValueError, match=r"\(samples, channels\)"):
        prepare_audio(np.zeros(shape), convert_audio=True)


def test_conversion_rejects_other_shapes():
    with pytest.raises(ValueError, match="mono"):
        prepare_audio(np.zeros((2, 2, 800)), convert_audio=True)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_conversion_rejects_non_finite_audio_before_resampling(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "soxr", None)  # would raise ImportError if reached
    wave = np.zeros(44100)
    wave[100] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        prepare_audio(wave, 44100, convert_audio=True)


def test_minimum_length_applies_after_conversion():
    pytest.importorskip("soxr")
    with pytest.raises(ValueError, match="At least 400"):
        prepare_audio(np.zeros(600), 44100, convert_audio=True)  # about 218 samples at 16 kHz


def test_empty_audio_is_rejected_by_the_length_check():
    with pytest.raises(ValueError, match="At least 400"):
        prepare_audio(np.zeros(0), 44100, convert_audio=True)


@pytest.mark.parametrize("flag", [1, "yes", None])
def test_convert_audio_must_be_a_bool(flag):
    with pytest.raises(TypeError, match="convert_audio"):
        prepare_audio(np.zeros(800), convert_audio=flag)


def test_load_audio_converts_a_stereo_file_only_when_asked(tmp_path):
    pytest.importorskip("soxr")
    path = tmp_path / "stereo.wav"
    tone = _tone(44100)
    sf.write(path, np.stack([tone, tone], axis=1), 44100, subtype="FLOAT")
    with pytest.raises(ValueError, match="convert_audio=True"):
        load_audio(path)
    assert load_audio(path, convert_audio=True).shape == (1, 16000)


class _Backend:
    def __init__(self):
        self.seen = []

    def infer(self, audio):
        self.seen.append(audio)
        return np.zeros((1, 2, 3), np.float32)


class _Tokenizer:
    def decode(self, ids):
        return ""


def _model():
    model = object.__new__(OmniASR)
    model.backend, model.tokenizer = _Backend(), _Tokenizer()
    return model


def test_ctc_entry_points_are_strict_unless_asked(tmp_path):
    stereo = np.zeros((16000, 2), np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 16000, subtype="FLOAT")
    model = _model()
    with pytest.raises(ValueError, match="convert_audio=True"):
        model.transcribe_numpy(stereo)
    with pytest.raises(ValueError, match="convert_audio=True"):
        model.transcribe(path)
    assert model.backend.seen == []
    model.transcribe_numpy(stereo, convert_audio=True)
    model.transcribe(path, convert_audio=True)
    assert [audio.shape for audio in model.backend.seen] == [(1, 16000)] * 2


def test_conversion_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        _model().transcribe_numpy(np.zeros(800), 16000, True)

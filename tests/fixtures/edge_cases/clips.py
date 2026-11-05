"""Deterministic edge-case clips, derived from the committed FLEURS audio.

Each recipe returns 16 kHz mono 16-bit PCM as an int16 array. Nothing here is committed
audio: the clips are rebuilt from tests/fixtures/fleurs (CC BY 4.0, see its README) and
seeded noise, and tests/unit/test_edge_case_fixtures.py checks each one against the
sha256 in manifest.json, so a change to a recipe can't silently change what's measured.
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
MIN_SAMPLES = 400
MAX_SAMPLES = 30 * SAMPLE_RATE
_FLEURS = Path(__file__).resolve().parent.parent / "fleurs"
_ENGLISH = "en_us_10233995782544396174.flac"
_TURKISH = "tr_tr_10147164563742619338.flac"
_NOISE_SEED = 32
# The two recordings are 34 dB apart and the English one is very quiet (RMS 4.8e-4). These
# fixed gains bring both to a normal speech level (RMS 0.03); the runtime normalizes each clip
# as a whole, so mixing them unlevelled would measure that gap instead of the edge case.
_ENGLISH_GAIN = 63.0
_TURKISH_GAIN = 1.2
_CLIPPED_GAIN = 2400.0
_ROOM_NOISE = 1e-4  # RMS, 50 dB below the levelled speech


def _speech(name, gain):
    pcm, sample_rate = sf.read(_FLEURS / name, dtype="int16")
    assert sample_rate == SAMPLE_RATE
    return pcm * gain


def _english(gain=_ENGLISH_GAIN):
    return _speech(_ENGLISH, gain)


def _turkish():
    return _speech(_TURKISH, _TURKISH_GAIN)


def _noise(samples, rms):
    """White noise at `rms` (a fraction of full scale), the same on every platform."""
    return np.random.default_rng(_NOISE_SEED).normal(0.0, rms * 32768.0, samples)


def _pcm(*parts):
    audio = np.concatenate([np.asarray(part, dtype=np.float64) for part in parts])
    return np.clip(np.rint(audio), -32768, 32767).astype(np.int16)


def silence():
    return _pcm(np.zeros(SAMPLE_RATE))


def white_noise():
    return _pcm(_noise(3 * SAMPLE_RATE, rms=0.1))


def mains_hum():
    time = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
    return _pcm(0.1 * 32768.0 * np.sin(2 * np.pi * 50.0 * time))


def clipped_speech():
    """English speech driven into full-scale clipping."""
    return _pcm(_english(_CLIPPED_GAIN))


def long_pause():
    """English speech with eight seconds of faint room noise inside the sentence."""
    speech = _english()
    middle = speech.size // 2
    return _pcm(speech[:middle], _noise(8 * SAMPLE_RATE, _ROOM_NOISE), speech[middle:])


def code_switching():
    """An English sentence, then a Turkish one at the same level."""
    return _pcm(_english(), np.zeros(SAMPLE_RATE // 2), _turkish())


def repeated_sentence():
    """The same sentence four times over."""
    speech, gap = _english(), np.zeros(SAMPLE_RATE // 4)
    return _pcm(*[part for _ in range(4) for part in (speech, gap)])


def stutter():
    """A quarter second of speech looped for twenty seconds, which invites a decoding loop."""
    return _pcm(np.tile(_english()[SAMPLE_RATE: SAMPLE_RATE + SAMPLE_RATE // 4], 80))


def shortest_clip():
    """Exactly the 400 samples the feature extractor needs."""
    return _pcm(_english()[SAMPLE_RATE: SAMPLE_RATE + MIN_SAMPLES])


def longest_clip():
    """Exactly 30 seconds: speech, then room noise to fill the limit."""
    speech = code_switching()
    return _pcm(speech, _noise(MAX_SAMPLES - speech.size, _ROOM_NOISE))


RECIPES = {recipe.__name__: recipe for recipe in (
    silence, white_noise, mains_hum, clipped_speech, long_pause, code_switching,
    repeated_sentence, stutter, shortest_clip, longest_clip,
)}


def build(name):
    """The clip as int16 PCM (hashed by the fixture tests)."""
    return RECIPES[name]()


def waveform(name):
    """The clip as the float32 waveform the runtimes take."""
    return (build(name) / 32768.0).astype(np.float32)


def manifest():
    return json.loads((Path(__file__).with_name("manifest.json")).read_text(encoding="utf-8"))


def reference(clip):
    """The transcript of a clip's speech, from the FLEURS manifest; empty if it has none."""
    fleurs = json.loads((_FLEURS / "manifest.json").read_text(encoding="utf-8"))
    sentences = {entry["fleurs_language"]: entry["reference"] for entry in fleurs["clips"]}
    return " ".join(sentences[language] for language in clip.get("reference", []))


def check_text(text, expected, reference, cer):
    """Assert a transcript meets a clip's `ctc`/`llm` entry; `cer(reference, text)` scores it."""
    if "text" in expected:
        assert text == expected["text"], repr(text)
    if "contains" in expected:
        assert expected["contains"] in text, text
    if "max_cer" in expected:
        score = cer(reference, text)
        assert score <= expected["max_cer"], f"CER {score:.1f}: {text!r}"

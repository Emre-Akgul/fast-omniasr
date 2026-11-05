"""Batch-one 16 kHz preprocessing matching the reference waveform normalization."""
import operator
from pathlib import Path

import numpy as np
import soundfile as sf

from .options import check_flag

MIN_AUDIO_SAMPLES = 400
SAMPLE_RATE = 16000
MIN_SAMPLE_RATE = 1000  # of audio to convert
MAX_SAMPLE_RATE = 1000000


def _sample_rate(sample_rate) -> int:
    """A whole number of hertz (16000.0 counts, as in strict mode) in a sane range: output
    size scales with the ratio, so a wrong rate would otherwise exhaust memory or hang."""
    value = None
    if isinstance(sample_rate, (float, np.floating)):
        value = int(sample_rate) if float(sample_rate).is_integer() else None
    elif not isinstance(sample_rate, bool):
        try:
            value = operator.index(sample_rate)
        except TypeError:
            pass
    if value is None or not MIN_SAMPLE_RATE <= value <= MAX_SAMPLE_RATE:
        raise ValueError(f"sample_rate must be a whole number from {MIN_SAMPLE_RATE} to "
                         f"{MAX_SAMPLE_RATE} Hz, got {sample_rate!r}")
    return value


def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import soxr
    except ImportError as exc:
        raise ImportError(
            "convert_audio=True needs soxr to resample: pip install fast-omniasr[resample]"
        ) from exc
    return soxr.resample(audio, sample_rate, SAMPLE_RATE, quality="HQ")


def _convert(audio: np.ndarray, sample_rate) -> np.ndarray:
    """Downmix (samples, channels) audio to mono and resample it to 16 kHz."""
    sample_rate = _sample_rate(sample_rate)
    if audio.ndim == 2:
        if not 0 < audio.shape[1] <= audio.shape[0]:
            raise ValueError(
                f"Expected multichannel audio shaped (samples, channels), got {audio.shape}"
            )
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim == 1 and audio.size and sample_rate != SAMPLE_RATE:
        if not np.isfinite(audio).all():  # the resampler would smear these over the clip
            raise ValueError("Audio contains NaN or infinity")
        return _resample(audio, sample_rate)
    return audio


def prepare_audio(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                  convert_audio: bool = False) -> np.ndarray:
    """Validate and normalize one clip. Strict by default: anything other than mono 16 kHz
    is rejected, so what's transcribed is exactly what was passed in. `convert_audio=True`
    first downmixes (samples, channels) audio by averaging the channels and resamples to
    16 kHz (the `resample` extra)."""
    check_flag("convert_audio", convert_audio)
    if not convert_audio and sample_rate != SAMPLE_RATE:
        raise ValueError(
            "Expected 16 kHz audio; resample before transcription or pass convert_audio=True"
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if convert_audio:
        audio = _convert(audio, sample_rate)
    if audio.ndim != 1:
        hint = "" if convert_audio else "; downmix before transcription or pass convert_audio=True"
        raise ValueError(f"Expected a mono, one-dimensional waveform{hint}")
    if audio.size < MIN_AUDIO_SAMPLES:
        raise ValueError(
            f"At least {MIN_AUDIO_SAMPLES} samples are required by the feature extractor"
        )
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + np.float32(1e-5))
    return np.ascontiguousarray(audio[None, :], dtype=np.float32)


def load_audio(path: str | Path, *, convert_audio: bool = False) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    return prepare_audio(audio, sample_rate, convert_audio=convert_audio)

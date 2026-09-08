"""Batch-one 16 kHz preprocessing matching the reference waveform normalization."""
from pathlib import Path

import numpy as np
import soundfile as sf


def prepare_audio(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    if sample_rate != 16000:
        raise ValueError("Expected 16 kHz audio; resample before transcription")
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("Expected a mono, one-dimensional waveform")
    if audio.size < 400:
        raise ValueError("At least 400 samples are required by the feature extractor")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + np.float32(1e-5))
    return np.ascontiguousarray(audio[None, :], dtype=np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    return prepare_audio(audio, sample_rate)

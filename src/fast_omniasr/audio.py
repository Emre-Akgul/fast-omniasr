"""Batch-one 16 kHz preprocessing matching the reference waveform normalization."""
import math
import numbers
import operator
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from .options import check_flag

MIN_AUDIO_SAMPLES = 400
SAMPLE_RATE = 16000
MIN_SAMPLE_RATE = 1000  # of audio to convert
MAX_SAMPLE_RATE = 1000000
DEFAULT_CHUNK_LENGTH = 15.0  # seconds; see docs/long-audio.md


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


def _mono_16k(waveform, sample_rate, convert_audio: bool) -> np.ndarray:
    """The clip as float32 mono 16 kHz, converted if asked, before any normalization."""
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
    return audio


def _normalize(audio: np.ndarray) -> np.ndarray:
    if audio.size < MIN_AUDIO_SAMPLES:
        raise ValueError(
            f"At least {MIN_AUDIO_SAMPLES} samples are required by the feature extractor"
        )
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    audio = (audio - audio.mean()) / np.sqrt(audio.var() + np.float32(1e-5))
    return np.ascontiguousarray(audio[None, :], dtype=np.float32)


def prepare_audio(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                  convert_audio: bool = False) -> np.ndarray:
    """Validate and normalize one clip. Strict by default: anything other than mono 16 kHz
    is rejected, so what's transcribed is exactly what was passed in. `convert_audio=True`
    first downmixes (samples, channels) audio by averaging the channels and resamples to
    16 kHz (the `resample` extra)."""
    return _normalize(_mono_16k(waveform, sample_rate, convert_audio))


def load_audio(path: str | Path, *, convert_audio: bool = False) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32")
    return prepare_audio(audio, sample_rate, convert_audio=convert_audio)


class AudioChunk(NamedTuple):
    start: float  # seconds into the recording
    end: float
    audio: np.ndarray
    spans: tuple[tuple[int, int], ...]  # the parts of the recording it holds, in samples


def silence_before(spans: tuple[tuple[int, int], ...], seconds: float) -> float:
    """Seconds to add to a time in a chunk to get the time in the recording, which is what
    faster-whisper's SpeechTimestampsMap gives. A time past the last span uses that span."""
    kept = 0
    for start, end in spans:
        kept += end - start
        if seconds * SAMPLE_RATE < kept:
            break
    return (start - (kept - (end - start))) / SAMPLE_RATE


def _check_chunk_length(chunk_length, maximum: float | None) -> int:
    """Chunk length in seconds, as a sample count."""
    try:
        valid = (isinstance(chunk_length, numbers.Real) and not isinstance(chunk_length, bool)
                 and math.isfinite(chunk_length * SAMPLE_RATE)
                 and chunk_length * SAMPLE_RATE >= MIN_AUDIO_SAMPLES)
    except OverflowError:  # an integer too large for a float
        valid = False
    if not valid:
        raise ValueError(f"chunk_length must be a number of seconds, at least "
                         f"{MIN_AUDIO_SAMPLES / SAMPLE_RATE}, got {chunk_length!r}")
    if maximum is not None and chunk_length > maximum:
        raise ValueError(f"chunk_length must be at most {maximum} seconds, got {chunk_length!r}")
    return int(chunk_length * SAMPLE_RATE)


def _speech_chunks(mono: np.ndarray, chunk_length: float, options) -> list:
    from .vad import speech_chunks  # needs onnxruntime

    chunks = []
    for spans in speech_chunks(mono, chunk_length, options):
        speech = np.concatenate([mono[span["start"]:span["end"]] for span in spans])
        if speech.size >= MIN_AUDIO_SAMPLES:
            chunks.append(AudioChunk(spans[0]["start"] / SAMPLE_RATE,
                                     spans[-1]["end"] / SAMPLE_RATE, _normalize(speech),
                                     tuple((span["start"], span["end"]) for span in spans)))
    return chunks


def prepare_chunks(audio: str | Path | np.ndarray, sample_rate: int = SAMPLE_RATE, *,
                   chunk_length: float = DEFAULT_CHUNK_LENGTH, convert_audio: bool = False,
                   max_chunk_length: float | None = None, vad_filter: bool = False,
                   vad_options=None):
    """Cut one recording at fixed intervals, with no overlap: a list of AudioChunk. Each
    chunk is normalized on its own, like a separate clip.
    A last chunk too short to transcribe (under 25 ms) is dropped.

    `vad_filter=True` instead finds speech with Silero VAD, as faster-whisper does: chunks
    hold only speech, joined up to `chunk_length`, and start and end stay times in the
    original recording. Audio with no speech gives an empty list."""
    size = _check_chunk_length(chunk_length, max_chunk_length)
    check_flag("vad_filter", vad_filter)
    if vad_options is not None and not vad_filter:
        raise ValueError("vad_options needs vad_filter=True")
    if isinstance(audio, (str, Path)):
        audio, sample_rate = sf.read(audio, dtype="float32")
    mono = _mono_16k(audio, sample_rate, convert_audio)
    if mono.size < MIN_AUDIO_SAMPLES:
        _normalize(mono)  # the usual error
    if vad_filter:
        if not np.isfinite(mono).all():
            raise ValueError("Audio contains NaN or infinity")
        return _speech_chunks(mono, size / SAMPLE_RATE, vad_options)
    starts = range(0, mono.size, size)
    return [AudioChunk(start / SAMPLE_RATE, min(start + size, mono.size) / SAMPLE_RATE,
                       _normalize(mono[start:start + size]),
                       ((start, min(start + size, mono.size)),))
            for start in starts if mono.size - start >= MIN_AUDIO_SAMPLES]

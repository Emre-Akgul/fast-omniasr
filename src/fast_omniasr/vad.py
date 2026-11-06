"""Silero voice activity detection, as used by faster-whisper.

Adapted from faster-whisper's vad.py (MIT, SYSTRAN), itself adapted from
https://github.com/snakers4/silero-vad (MIT, Silero Team). The model is
assets/silero_vad_v6.onnx. Both licenses are in assets/. Needs onnxruntime.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MODEL = Path(__file__).parent / "assets" / "silero_vad_v6.onnx"


@dataclass
class VadOptions:
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      neg_threshold: Silence threshold for determining the end of speech. If a probability is lower
        than neg_threshold, it is always considered silence. Values higher than neg_threshold
        are only considered speech if the previous sample was classified as speech; otherwise,
        they are treated as silence. This parameter helps refine the detection of speech
         transitions, ensuring smoother segment boundaries.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than min_silence_at_max_speech (if any), to prevent aggressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
      min_silence_at_max_speech: Minimum silence duration in ms which is used to avoid abrupt cuts
          when max_speech_duration_s is reached.
      use_max_poss_sil_at_max_speech: Whether to use the maximum possible silence at
          max_speech_duration_s or not. If not, the last silence is used.
    """

    threshold: float = 0.5
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    min_silence_at_max_speech: int = 98
    use_max_poss_sil_at_max_speech: bool = True


def get_speech_timestamps(
    audio: np.ndarray,
    vad_options: VadOptions | None = None,
    sampling_rate: int = 16000,
    **kwargs,
) -> list[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      sampling rate: Sampling rate of the audio.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    threshold = vad_options.threshold
    neg_threshold = vad_options.neg_threshold
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    window_size_samples = 512
    speech_pad_ms = vad_options.speech_pad_ms
    min_silence_at_max_speech = vad_options.min_silence_at_max_speech
    use_max_poss_sil_at_max_speech = vad_options.use_max_poss_sil_at_max_speech

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    audio_length_samples = len(audio)

    model = get_vad_model()

    padded_audio = np.pad(
        audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
    )
    speech_probs = model(padded_audio)

    triggered = False
    speeches = []
    current_speech = {}
    possible_ends = []

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and (cur_sample - current_speech["start"] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech["end"] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if (
                not use_max_poss_sil_at_max_speech
                and sil_dur_now > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches


@functools.lru_cache
def get_vad_model():
    """The VAD model, loaded once per process. Inference keeps no state between calls."""
    return SileroVADModel(_MODEL)


class SileroVADModel:
    def __init__(self, path):
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                "vad_filter=True needs onnxruntime: pip install fast-omniasr[onnx]"
            ) from e

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 4

        self.session = onnxruntime.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

    def __call__(
        self, audio: np.ndarray, num_samples: int = 512, context_size_samples: int = 64
    ):
        assert audio.ndim == 1, "Input should be a 1D array"
        assert (
            audio.shape[0] % num_samples == 0
        ), "Input size should be a multiple of num_samples"

        h = np.zeros((1, 1, 128), dtype="float32")
        c = np.zeros((1, 1, 128), dtype="float32")
        context = np.zeros(
            (1, context_size_samples),
            dtype="float32",
        )

        batched_audio = audio.reshape(-1, num_samples)
        context = batched_audio[..., -context_size_samples:]
        context[-1] = 0
        context = np.roll(context, 1, 0)
        batched_audio = np.concatenate([context, batched_audio], 1)

        batched_audio = batched_audio.reshape(-1, num_samples + context_size_samples)

        encoder_batch_size = 10000
        num_segments = batched_audio.shape[0]
        outputs = []
        for i in range(0, num_segments, encoder_batch_size):
            output, h, c = self.session.run(
                None,
                {"input": batched_audio[i : i + encoder_batch_size], "h": h, "c": c},
            )
            outputs.append(output)

        out = np.concatenate(outputs, axis=0)

        return out


def _vad_options(options, chunk_length: float) -> VadOptions:
    """As in faster-whisper's batched pipeline: a VadOptions is used as given, while the
    default and a dict get `max_speech_duration_s` set to the chunk length."""
    if isinstance(options, VadOptions):
        return options
    if options is None:
        return VadOptions(min_silence_duration_ms=160, max_speech_duration_s=chunk_length)
    if isinstance(options, dict):
        return VadOptions(**{**options, "max_speech_duration_s": chunk_length})
    raise TypeError(f"vad_options must be a VadOptions or a dict, got {options!r}")


def speech_chunks(audio: np.ndarray, chunk_length: float, options=None,
                  sampling_rate: int = 16000) -> list[list[dict[str, int]]]:
    """Group the speech in `audio` into chunks of at most `chunk_length` seconds of audio,
    like faster-whisper's batched pipeline: a chunk is a list of {"start", "end"} sample
    spans. Raises ValueError for a span longer than a chunk, as upstream does."""
    spans = get_speech_timestamps(audio, _vad_options(options, chunk_length), sampling_rate)
    limit = chunk_length * sampling_rate
    chunks, current, duration = [], [], 0
    for span in spans:
        length = span["end"] - span["start"]
        if length > limit:
            raise ValueError(
                f"Speech at {span['start'] / sampling_rate:.2f}s lasts "
                f"{length / sampling_rate:.2f}s, longer than the {chunk_length}s chunk length; "
                "use a larger chunk_length, a smaller speech_pad_ms, or set max_speech_duration_s"
            )
        if current and duration + length > limit:
            chunks.append(current)
            current, duration = [], 0
        current.append(span)
        duration += length
    if current:
        chunks.append(current)
    return chunks

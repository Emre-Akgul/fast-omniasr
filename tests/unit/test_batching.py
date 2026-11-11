"""Batched CTC inference: padded batch graphs, bucketing and parity with one clip (#37)."""

import sys
import types
from typing import ClassVar

import numpy as np
import pytest
import soundfile as sf

from fast_omniasr import OmniASR
from fast_omniasr.audio import pad_batch, prepare_audio
from fast_omniasr.backends.onnx import ONNXBackend
from fast_omniasr.decoder import FRAME_STRIDE, FRAME_WINDOW

VOCABULARY = 7


def noise(samples, seed=0):
    return np.random.default_rng(seed).normal(size=samples).astype(np.float32)


def frames(samples):
    return (samples - FRAME_WINDOW) // FRAME_STRIDE + 1


def clip_logits(row: np.ndarray) -> np.ndarray:
    """A stand-in model whose frames depend only on their own window of samples."""
    count = frames(row.size)
    logits = np.zeros((count, VOCABULARY), np.float32)
    for frame in range(count):
        window = row[frame * FRAME_STRIDE:frame * FRAME_STRIDE + FRAME_WINDOW]
        logits[frame, int(abs(window.sum()) * 10) % VOCABULARY] = 1
    return logits


def batch_logits(audio: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """What a batch graph returns: rows padded to the longest, with garbage past each."""
    out = np.full((len(lengths), frames(audio.shape[1]), VOCABULARY), np.nan, np.float32)
    for row, length in enumerate(lengths):
        logits = clip_logits(audio[row, :length])
        out[row, :len(logits)] = logits
    return out


class Backend:
    """infer() and infer_batch() as the ONNX backend provides them, recording each call."""

    def __init__(self, batched=True):
        self.calls = []
        if not batched:
            self.infer_batch = None

    def infer(self, waveform):
        self.calls.append([waveform.shape[1]])
        return clip_logits(waveform[0])[None]

    def infer_batch(self, waveforms):
        self.calls.append([waveform.shape[1] for waveform in waveforms])
        audio, lengths = pad_batch(waveforms)
        logits = batch_logits(audio, lengths)
        return [logits[row:row + 1, :frames(length)] for row, length in enumerate(lengths)]


class Tokenizer:
    def decode(self, ids):
        return " ".join(f"t{i}" for i in ids)

    def split_words(self, ids):
        return [[index] for index in range(len(ids))]


def model(backend=None):
    instance = object.__new__(OmniASR)
    instance.backend, instance.tokenizer = backend or Backend(), Tokenizer()
    return instance


CLIPS = [noise(samples, seed) for seed, samples in
         enumerate([16000, 400, 48000, 8000, 32000, 1200, 24000])]


def test_pad_batch_zero_pads_to_the_longest_clip():
    audio, lengths = pad_batch([np.ones((1, 3), np.float32), np.full((1, 5), 2, np.float32)])
    assert audio.dtype == np.float32 and lengths.dtype == np.int64
    np.testing.assert_array_equal(lengths, [3, 5])
    np.testing.assert_array_equal(audio, [[1, 1, 1, 0, 0], [2, 2, 2, 2, 2]])
    with pytest.raises(ValueError, match="at least one"):
        pad_batch([])


@pytest.mark.parametrize("word_timestamps", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 3, 8])
def test_batched_results_match_transcribing_each_clip_alone(batch_size, word_timestamps):
    alone = [model().transcribe_numpy(clip, word_timestamps=word_timestamps) for clip in CLIPS]
    batched = model().transcribe_batch(CLIPS, batch_size=batch_size,
                                       word_timestamps=word_timestamps)
    assert batched == alone
    assert any(result.token_ids for result in batched)


def test_clips_are_bucketed_by_length_and_returned_in_order():
    backend = Backend()
    results = model(backend).transcribe_batch(CLIPS, batch_size=3)
    assert backend.calls == [[48000, 32000, 24000], [16000, 8000, 1200], [400]]
    assert [result.logits_shape[1] for result in results] == [frames(c.size) for c in CLIPS]


def test_backends_without_batching_run_clips_one_by_one():
    backend = Backend(batched=False)
    results = model(backend).transcribe_batch(CLIPS, batch_size=4)
    assert sorted(backend.calls) == sorted([[clip.size] for clip in CLIPS])
    assert results == [model().transcribe_numpy(clip) for clip in CLIPS]


def test_files_and_waveforms_can_be_mixed(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, CLIPS[0] * 0.1, 16000, subtype="FLOAT")
    results = model().transcribe_batch([path, str(path), CLIPS[2] * 0.1])
    assert results[0] == results[1] == model().transcribe(path)
    assert results[2] == model().transcribe_numpy(CLIPS[2] * 0.1)


def test_an_empty_list_gives_no_results():
    assert model().transcribe_batch([]) == []


@pytest.mark.parametrize("audios", ["clip.wav", np.zeros((2, 800), np.float32)])
def test_a_single_clip_is_not_mistaken_for_a_list(audios):
    with pytest.raises(TypeError, match="list of file paths or waveforms"):
        model().transcribe_batch(audios)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "8", None])
def test_invalid_batch_sizes_are_rejected(batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        model().transcribe_batch(CLIPS, batch_size=batch_size)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        model().transcribe_long(CLIPS[2], chunk_length=1, batch_size=batch_size)


def test_every_clip_is_validated_before_any_inference():
    backend = Backend()
    with pytest.raises(ValueError, match="At least 400"):
        model(backend).transcribe_batch([CLIPS[0], noise(399)])
    assert backend.calls == []


def test_a_closed_model_refuses_batches():
    instance = model()
    instance.backend.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        instance.transcribe_batch(CLIPS)


@pytest.mark.parametrize("word_timestamps", [False, True])
def test_batched_long_audio_matches_one_chunk_at_a_time(word_timestamps):
    recording = noise(int(5.5 * 16000), seed=9)
    one = model().transcribe_long(recording, chunk_length=1, word_timestamps=word_timestamps)
    backend = Backend()
    batched = model(backend).transcribe_long(recording, chunk_length=1, batch_size=4,
                                             word_timestamps=word_timestamps)
    assert batched == one
    assert backend.calls == [[16000] * 4, [16000, 8000]]


# The ONNX backend with a batch graph's I/O signature.

class Session:
    runs: ClassVar[list] = []

    def __init__(self, path, sess_options=None, providers=None):
        self.inputs = [types.SimpleNamespace(name="audio", type="tensor(float)",
                                             shape=["batch", "samples"]),
                       types.SimpleNamespace(name="audio_lengths", type="tensor(int64)",
                                             shape=["batch"])]
        self.outputs = [types.SimpleNamespace(name="logits",
                                              shape=["batch", "frames", VOCABULARY]),
                        types.SimpleNamespace(name="logit_lengths", shape=["batch"])]

    def disable_fallback(self):
        pass

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, feeds):
        audio, lengths = feeds["audio"], feeds["audio_lengths"]
        assert audio.dtype == np.float32 and lengths.dtype == np.int64
        Session.runs.append(lengths.tolist())
        outputs = {"logits": batch_logits(audio, lengths),
                   "logit_lengths": frames(lengths).astype(np.int64)}
        return [outputs[name] for name in names]


@pytest.fixture
def fake_ort(monkeypatch):
    Session.runs = []
    ort = types.SimpleNamespace(
        InferenceSession=Session,
        SessionOptions=lambda: types.SimpleNamespace(),
        get_available_providers=lambda: ["CPUExecutionProvider"],
        preload_dlls=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return Session


def normalized(clips):
    return [prepare_audio(clip) for clip in clips]


def test_a_batch_graph_runs_the_whole_batch_in_one_call(fake_ort):
    backend = ONNXBackend("model.onnx")
    assert backend.batched and backend.vocabulary_size == VOCABULARY
    waveforms = normalized(CLIPS[:3])
    rows = backend.infer_batch(waveforms)
    assert fake_ort.runs == [[16000, 400, 48000]]
    for waveform, row in zip(waveforms, rows, strict=True):
        assert np.isfinite(row).all()  # the padded frames were cut off
        np.testing.assert_array_equal(row, clip_logits(waveform[0])[None])
        np.testing.assert_array_equal(row, backend.infer(waveform))
    assert backend.infer_batch([]) == []


def test_a_batch_graph_answers_single_clips_too(fake_ort):
    waveform = prepare_audio(CLIPS[0])
    logits = ONNXBackend("model.onnx").infer(waveform)
    assert fake_ort.runs == [[16000]]
    np.testing.assert_array_equal(logits, clip_logits(waveform[0])[None])


def test_batch_graph_frame_counts_are_checked(fake_ort, monkeypatch):
    def run(self, names, feeds):
        return [batch_logits(feeds["audio"], feeds["audio_lengths"]),
                np.array([0, 3], np.int64)]

    monkeypatch.setattr(Session, "run", run)
    with pytest.raises(RuntimeError, match="frame counts"):
        ONNXBackend("model.onnx").infer_batch(normalized(CLIPS[:2]))


def test_batch_inputs_are_validated(fake_ort):
    backend = ONNXBackend("model.onnx")
    with pytest.raises(ValueError, match=r"float32 waveform \[1, samples\]"):
        backend.infer_batch([np.zeros((2, 800), np.float32)])
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.infer_batch(normalized(CLIPS[:2]))


@pytest.mark.parametrize(("change", "message"), [
    (lambda s: setattr(s.inputs[1], "type", "tensor(int32)"), "int64 audio_lengths"),
    (lambda s: setattr(s.inputs[0], "shape", ["samples"]), r"audio input \[batch, samples\]"),
    (lambda s: s.outputs.pop(), "logits and logit_lengths"),
    (lambda s: s.inputs.reverse(), "one float32 input named audio"),
])
def test_malformed_batch_graphs_are_rejected(fake_ort, monkeypatch, change, message):
    init = Session.__init__

    def broken(self, *args, **kwargs):
        init(self, *args, **kwargs)
        change(self)

    monkeypatch.setattr(Session, "__init__", broken)
    with pytest.raises(ValueError, match=message):
        ONNXBackend("model.onnx")


def test_a_batch_one_graph_runs_batches_clip_by_clip(fake_ort, monkeypatch):
    calls = []

    def init(self, *args, **kwargs):
        self.inputs = [types.SimpleNamespace(name="audio", type="tensor(float)",
                                             shape=[1, "samples"])]
        self.outputs = [types.SimpleNamespace(name="logits", shape=[1, "frames", VOCABULARY])]

    def run(self, names, feeds):
        assert set(feeds) == {"audio"}
        calls.append(feeds["audio"].shape[1])
        return [clip_logits(feeds["audio"][0])[None]]

    monkeypatch.setattr(Session, "__init__", init)
    monkeypatch.setattr(Session, "run", run)
    backend = ONNXBackend("model.onnx")
    assert not backend.batched
    rows = backend.infer_batch(normalized(CLIPS[:3]))
    assert calls == [16000, 400, 48000] and [row.shape[1] for row in rows] == [
        frames(16000), frames(400), frames(48000)]

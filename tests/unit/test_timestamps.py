"""CTC token and word timestamps (#40)."""

import numpy as np
import pytest

from fast_omniasr import OmniASR, Word
from fast_omniasr.audio import prepare_chunks, silence_before
from fast_omniasr.decoder import greedy_token_ids, greedy_tokens
from fast_omniasr.tokenizer import Tokenizer

PIECES = {0: "<blank>", 1: "<pad>", 2: "a", 3: "b", 4: " ", 5: "▁c"}


def logits(frames, vocabulary=6):
    values = np.zeros((1, len(frames), vocabulary + 1), np.float32)
    values[0, np.arange(len(frames)), frames] = 1
    return values


class Processor:
    def id_to_piece(self, token):
        return PIECES[token]

    def is_control(self, token):
        return token < 2

    def decode(self, ids):
        return "".join(PIECES[i] for i in ids if i > 1).replace("▁", " ").strip()


def tokenizer():
    instance = object.__new__(Tokenizer)
    instance._processor = Processor()
    return instance


class Backend:
    def __init__(self, *outputs):
        self.outputs, self.seen = list(outputs), []

    def infer(self, audio):
        self.seen.append(audio)
        return logits(self.outputs.pop(0))


def model(*outputs):
    instance = object.__new__(OmniASR)
    instance.backend, instance.tokenizer = Backend(*outputs), tokenizer()
    return instance


def audio_for(frames):
    return np.random.default_rng(0).normal(size=frames * 320 + 80).astype(np.float32)


def test_tokens_keep_their_frames_through_the_collapse():
    assert greedy_tokens(logits([0, 2, 2, 0, 2, 3, 3, 3, 0])) == [
        (2, 1, 3), (2, 4, 5), (3, 5, 8)]


def test_a_token_at_either_end_of_the_clip_keeps_its_frames():
    assert greedy_tokens(logits([2, 0, 3])) == [(2, 0, 1), (3, 2, 3)]


def test_all_blank_and_empty_logits_have_no_tokens():
    assert greedy_tokens(logits([0, 0])) == []
    assert greedy_tokens(np.zeros((1, 0, 6), np.float32)) == []


@pytest.mark.parametrize("frames", [[0, 2, 2, 0, 2, 3], [2, 2, 2], [0, 0], [3, 0, 3, 4, 5]])
def test_token_ids_are_the_same_as_before(frames):
    values = logits(frames)
    ids = values[0].argmax(-1)
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    assert greedy_token_ids(values) == ids[ids != 0].tolist()


@pytest.mark.parametrize("ids, words", [
    ([], []),
    ([2, 3], [[0, 1]]),
    ([2, 4, 3], [[0], [2]]),
    ([4, 2, 4], [[1]]),
    ([2, 4, 4, 3], [[0], [3]]),
    ([2, 5, 3], [[0], [1, 2]]),
    ([5, 2], [[0, 1]]),
    ([2, 1, 3], [[0, 2]]),
    ([1], []),
])
def test_word_boundaries(ids, words):
    assert tokenizer().split_words(ids) == words


def test_word_and_token_times_come_from_frames():
    result = model([0, 2, 2, 3, 0, 4, 0, 2]).transcribe_numpy(audio_for(8),
                                                              word_timestamps=True)
    assert result.text == "ab a"
    assert result.words == (Word(0.02, 0.08, "ab"), Word(0.14, 0.16, "a"))
    assert result.token_times == ((0.02, 0.06), (0.06, 0.08), (0.1, 0.12), (0.14, 0.16))
    assert len(result.token_times) == len(result.token_ids)


def test_timestamps_are_off_by_default():
    result = model([2]).transcribe_numpy(audio_for(1))
    assert result.words == () and result.token_times == ()


def test_timestamps_leave_the_output_alone():
    frames = [0, 2, 2, 3, 0, 4, 0, 2]
    plain = model(frames).transcribe_numpy(audio_for(8))
    timed = model(frames).transcribe_numpy(audio_for(8), word_timestamps=True)
    assert (timed.text, timed.token_ids, timed.logits_shape) == (
        plain.text, plain.token_ids, plain.logits_shape)


def test_silence_has_no_words():
    result = model([0, 0]).transcribe_numpy(audio_for(2), word_timestamps=True)
    assert result.words == () and result.token_times == ()


@pytest.mark.parametrize("given", [99, 101, 50, 200])
def test_a_frame_count_that_misses_the_stride_is_refused(given):
    with pytest.raises(ValueError, match="320 samples per frame"):
        model([2] * given).transcribe_numpy(audio_for(100), word_timestamps=True)


@pytest.mark.parametrize("samples, frames", [
    (400, 1), (639, 1), (720, 2), (960, 2), (16000, 49), (480000, 1499), (480320, 1500)])
def test_every_length_from_the_real_model_is_accepted(samples, frames):
    audio = np.random.default_rng(0).normal(size=samples).astype(np.float32)
    assert model([0] * frames).transcribe_numpy(audio, word_timestamps=True).words == ()


def test_malformed_logits_get_the_usual_error():
    instance = model([2])
    instance.backend.infer = lambda audio: np.zeros((5, 6), np.float32)
    with pytest.raises(ValueError, match=r"Expected logits"):
        instance.transcribe_numpy(audio_for(5), word_timestamps=True)


def test_an_unknown_token_alone_rejoins_the_text():
    PIECES[6] = "<unk>"
    try:
        instance = model([6])
        instance.tokenizer._processor.decode = lambda ids: " ? "
        assert instance.transcribe_numpy(audio_for(1), word_timestamps=True).words == (
            Word(0.0, 0.02, "?"),)
    finally:
        del PIECES[6]


def test_transcribe_reads_files(tmp_path):
    import soundfile as sf

    path = tmp_path / "clip.wav"
    sf.write(path, audio_for(2) / 10, 16000)
    assert model([2, 3]).transcribe(path, word_timestamps=True).words == (
        Word(0.0, 0.04, "ab"),)


@pytest.mark.parametrize("call", ["transcribe_numpy", "transcribe_long"])
def test_the_flag_must_be_a_bool(call):
    with pytest.raises(TypeError, match="word_timestamps"):
        getattr(model([2]), call)(audio_for(1), word_timestamps=1)


def test_silence_before_follows_the_spans():
    spans = ((1000, 2000), (5000, 5500))
    assert silence_before(spans, 0.0) == 1000 / 16000
    assert silence_before(spans, 999 / 16000) == 1000 / 16000
    assert silence_before(spans, 1000 / 16000) == 4000 / 16000
    assert silence_before(spans, 1400 / 16000) == 4000 / 16000
    assert silence_before(spans, 1 / 2) == 4000 / 16000  # past the end


def test_fixed_chunks_record_their_samples():
    chunks = prepare_chunks(audio_for(101)[:2 * 16000 + 400], chunk_length=1)
    assert [chunk.spans for chunk in chunks] == [
        ((0, 16000),), ((16000, 32000),), ((32000, 32400),)]


def test_long_audio_times_are_in_the_recording():
    second = 49  # frames in one second
    result = model([0, 2, *[0] * (second - 2)], [3, 0, *[0] * (second - 2)]
                   ).transcribe_long(np.random.default_rng(0).normal(size=2 * 16000)
                                     .astype(np.float32), chunk_length=1,
                                     word_timestamps=True)
    assert result.words == (Word(0.02, 0.04, "a"), Word(1.0, 1.02, "b"))
    assert [chunk.words for chunk in result.chunks] == [
        (Word(0.02, 0.04, "a"),), (Word(1.0, 1.02, "b"),)]
    assert result.token_times == ((0.02, 0.04), (1.0, 1.02))
    assert result.chunks[1].token_times == ((1.0, 1.02),)


def test_long_audio_times_skip_what_the_vad_removed(monkeypatch):
    from fast_omniasr import vad

    spans = [[{"start": 16000, "end": 24000}, {"start": 80000, "end": 88000}]]
    monkeypatch.setattr(vad, "speech_chunks", lambda *args: spans)
    frames = [0] * 49
    frames[1], frames[2], frames[30] = 2, 4, 3  # "a b": 0.02 s and 0.6 s into the chunk
    result = model(frames).transcribe_long(
        np.random.default_rng(0).normal(size=100000).astype(np.float32),
        vad_filter=True, word_timestamps=True)
    assert result.words == (Word(1.02, 1.04, "a"), Word(5.1, 5.12, "b"))


def test_a_word_across_a_cut_moves_as_one():
    spans = ((0, 1000), (5000, 6000))
    from fast_omniasr.model import _shifted

    assert _shifted(0.04, 0.07, spans) == (0.04, 0.07)  # midpoint before the cut
    assert _shifted(0.05, 0.1, spans) == (0.3, 0.35)  # midpoint after it

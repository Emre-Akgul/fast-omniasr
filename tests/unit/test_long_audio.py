"""Fixed-length chunking for long recordings (#35)."""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from fast_omniasr import OmniASR
from fast_omniasr.audio import DEFAULT_CHUNK_LENGTH, _normalize, prepare_chunks
from fast_omniasr.llm import OmniASRLLM


def noise(samples, seed=0):
    return np.random.default_rng(seed).normal(size=samples).astype(np.float32)


def test_chunks_cover_the_recording_in_order_without_overlap():
    chunks = prepare_chunks(noise(45 * 16000), chunk_length=15)
    assert [(start, end) for start, end, _ in chunks] == [(0, 15), (15, 30), (30, 45)]
    assert [audio.shape for *_, audio in chunks] == [(1, 240000)] * 3


def test_the_last_chunk_is_shorter():
    chunks = prepare_chunks(noise(20 * 16000), chunk_length=15)
    assert [(start, end) for start, end, _ in chunks] == [(0, 15), (15, 20)]


def test_each_chunk_is_normalized_on_its_own():
    waveform = np.concatenate([noise(16000) * 0.5, noise(16000, 1) * 0.001])
    chunks = prepare_chunks(waveform, chunk_length=1)
    np.testing.assert_array_equal(chunks[1][2], _normalize(waveform[16000:]))


def test_a_clip_shorter_than_a_chunk_is_one_chunk():
    chunks = prepare_chunks(noise(800), chunk_length=15)
    assert len(chunks) == 1 and chunks[0][1] == 0.05


@pytest.mark.parametrize("tail,count", [(399, 2), (400, 3)])
def test_a_last_chunk_below_the_minimum_is_dropped(tail, count):
    assert len(prepare_chunks(noise(2 * 16000 + tail), chunk_length=1)) == count


def test_a_recording_below_the_minimum_is_rejected():
    with pytest.raises(ValueError, match="At least 400"):
        prepare_chunks(noise(399))


@pytest.mark.parametrize("value", [0, -1, 0.02, float("nan"), float("inf"), True, "15", None,
                                   10**400, 1e308])
def test_invalid_chunk_lengths_are_rejected(value):
    with pytest.raises(ValueError, match="chunk_length"):
        prepare_chunks(noise(800), chunk_length=value)


def test_fractional_chunk_lengths_keep_correct_times():
    chunks = prepare_chunks(noise(16000), chunk_length=0.25)
    assert [(start, end) for start, end, _ in chunks] == [
        (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]


def test_conversion_is_opt_in_like_the_strict_api():
    with pytest.raises(ValueError, match="convert_audio=True"):
        prepare_chunks(np.zeros((16000, 2), np.float32))
    assert len(prepare_chunks(np.zeros((16000, 2), np.float32), convert_audio=True)) == 1


def test_non_finite_audio_is_rejected():
    waveform = noise(16000)
    waveform[100] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        prepare_chunks(waveform)


class Backend:
    def __init__(self):
        self.seen = []

    def infer(self, audio):
        self.seen.append(audio)
        logits = np.zeros((1, 1, 5), np.float32)
        logits[0, 0, len(self.seen)] = 1
        return logits


class Tokenizer:
    def decode(self, ids):
        return "".join(f"t{i}" for i in ids)


def ctc():
    model = object.__new__(OmniASR)
    model.backend, model.tokenizer = Backend(), Tokenizer()
    return model


def test_ctc_joins_text_in_order_and_reports_chunks():
    model = ctc()
    result = model.transcribe_long(noise(3 * 16000), chunk_length=1)
    assert result.text == "t1 t2 t3" and result.token_ids == (1, 2, 3)
    assert [(c.start, c.end, c.text) for c in result.chunks] == [
        (0, 1, "t1"), (1, 2, "t2"), (2, 3, "t3")]
    assert result.logits_shape == (1, 1, 5)


def test_ctc_logits_shape_is_the_last_chunks():
    model = ctc()
    shapes = iter([(1, 2, 5), (1, 1, 5)])

    def infer(audio):
        return np.zeros(next(shapes), np.float32)

    model.backend.infer = infer
    assert model.transcribe_long(noise(2 * 16000), chunk_length=1).logits_shape == (1, 1, 5)


def test_ctc_empty_chunks_add_no_spaces():
    model = ctc()
    model.tokenizer = type("Silent", (), {"decode": lambda self, ids: "" if ids == [2] else "x"})()
    assert model.transcribe_long(noise(3 * 16000), chunk_length=1).text == "x x"


def test_ctc_reads_files(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, noise(2 * 16000) * 0.1, 16000, subtype="FLOAT")
    assert len(ctc().transcribe_long(path, chunk_length=1).chunks) == 2


def test_the_default_chunk_length_is_used():
    model = ctc()
    result = model.transcribe_long(noise(int(DEFAULT_CHUNK_LENGTH * 16000) + 8000))
    assert result.chunks[0].end == DEFAULT_CHUNK_LENGTH and len(result.chunks) == 2


def test_transcribe_keeps_no_chunks():
    assert ctc().transcribe_numpy(noise(16000)).chunks == ()


class Onnx:
    """Fake decoder: emits one token, then EOS, unless `endless` chunks hit the budget."""

    def __init__(self, endless=()):
        self.endless, self.chunk, self.step, self.waves = set(endless), 0, 0, []

    def encode(self, waveform):
        self.waves.append(waveform)
        self.chunk, self.step = self.chunk + 1, 0
        return np.zeros((1, 4, 8), np.float32)

    def _logits(self):
        token = 9 + self.chunk if self.step == 0 or self.chunk in self.endless else 2
        self.step += 1
        logits = np.zeros((1, 64), np.float32)
        logits[0, token] = 1
        return logits

    def prefill(self, context, token):
        return self._logits(), []

    def decode_step(self, token, cache):
        return self._logits(), cache


def llm(endless=()):
    model = object.__new__(OmniASRLLM)
    model.backend = "onnx"
    model.config = {"bos_idx": 0, "eos_idx": 2, "max_seq_len": 8192}
    model.tokenizer = Tokenizer()
    model.onnx = Onnx(endless)
    return model


def long(model, seconds=3, **options):
    with pytest.warns(UserWarning, match="lang") as caught:
        result = model.transcribe_long(noise(seconds * 16000), chunk_length=1, **options)
    assert len(caught) == 1  # warned once, not per chunk
    return result


def test_llm_joins_text_and_reports_chunks():
    result = long(llm())
    assert result.text == "t10 t11 t12" and result.token_ids == [10, 11, 12]
    assert result.stop_reason == "eos"
    assert [(c.start, c.end, c.stop_reason) for c in result.chunks] == [
        (0, 1, "eos"), (1, 2, "eos"), (2, 3, "eos")]


def test_llm_normalizes_each_chunk_exactly_once():
    model = llm()
    waveform = noise(2 * 16000) * 0.1
    with pytest.warns(UserWarning):
        model.transcribe_long(waveform, chunk_length=1)
    np.testing.assert_array_equal(model.onnx.waves[1], _normalize(waveform[16000:]))


def test_llm_reports_a_cut_off_chunk():
    result = long(llm(endless=[2]), max_new_tokens=3)
    assert result.stop_reason == "max_new_tokens"
    assert [c.stop_reason for c in result.chunks] == ["eos", "max_new_tokens", "eos"]


def test_llm_reports_the_first_cut_off_reason():
    model = llm(endless=[1, 3])
    budgets = iter(["max_new_tokens", "context_limit"])
    real = model._transcribe_onnx

    def relabel(*args):
        result = real(*args)
        if result.stop_reason == "eos":
            return result
        return replace(result, stop_reason=next(budgets))

    model._transcribe_onnx = relabel
    assert long(model, max_new_tokens=3).stop_reason == "max_new_tokens"


def test_llm_passes_the_language_to_every_chunk():
    model = llm()
    row = np.ones(8, np.float32)
    model.language_embedding = lambda lang: row
    seen = []
    real = model._transcribe_onnx

    def spy(waveform, budget, language, penalty):
        seen.append(language)
        return real(waveform, budget, language, penalty)

    model._transcribe_onnx = spy
    model.transcribe_long(noise(3 * 16000), chunk_length=1, lang="eng_Latn")
    assert len(seen) == 3 and all(language is row for language in seen)


def test_llm_empty_chunks_add_no_spaces():
    model = llm()
    model.tokenizer = type("Silent", (), {"decode": lambda self, ids: "" if ids == [11] else "x"})()
    assert long(model).text == "x x"


def test_llm_warning_points_at_the_caller():
    with pytest.warns(UserWarning, match="lang") as caught:
        llm().transcribe_long(noise(16000), chunk_length=1)
    assert caught[0].filename == __file__


def test_llm_budget_applies_per_chunk():
    result = long(llm(endless=[1, 2, 3]), max_new_tokens=3)
    assert [len(c.token_ids) for c in result.chunks] == [3, 3, 3]


def test_llm_accepts_recordings_past_the_clip_limit():
    assert len(long(llm(), seconds=31).chunks) == 31


def test_llm_rejects_chunks_past_the_clip_limit():
    with pytest.raises(ValueError, match="at most 30"):
        llm().transcribe_long(noise(16000), chunk_length=31)


@pytest.mark.parametrize("options", [{"max_new_tokens": 0}, {"repetition_penalty": 0}])
def test_llm_checks_options_before_reading_audio(options):
    with pytest.raises(ValueError, match="max_new_tokens|repetition_penalty"):
        llm().transcribe_long("missing.wav", **options)


def test_llm_rejects_a_closed_model():
    model = llm()
    model.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        model.transcribe_long(noise(16000))

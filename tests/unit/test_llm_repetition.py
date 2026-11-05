"""repetition_penalty matches faster-whisper's rule and is inert at its default (#42)."""

import numpy as np
import pytest

from fast_omniasr.llm import OmniASRLLM
from fast_omniasr.llm.runtime import _next_token

EOS = 2


class Text:
    def decode(self, ids):
        return " ".join(map(str, ids))


class Onnx:
    """Fake decoder: the same scores at every step, with EOS once `limit` tokens are out."""

    def __init__(self, scores, limit):
        self.scores, self.limit, self.steps = np.array(scores, np.float32), limit, 0

    def encode(self, waveform):
        return np.zeros((1, 4, 8), np.float32)

    def _logits(self):
        logits = self.scores.copy()
        if self.steps >= self.limit:
            logits[EOS] = 100.0
        self.steps += 1
        return logits[None]

    def prefill(self, context, token):
        return self._logits(), []

    def decode_step(self, token, cache):
        return self._logits(), cache


def model(scores, limit=3):
    instance = object.__new__(OmniASRLLM)
    instance.backend = "onnx"
    instance.config = {"bos_idx": 0, "eos_idx": EOS, "max_seq_len": 8192}
    instance.tokenizer = Text()
    instance.onnx = Onnx(scores, limit)
    return instance


def transcribe(instance, **options):
    with pytest.warns(UserWarning, match="lang"):
        return instance.transcribe(np.zeros(1600, np.float32), **options)


def test_positive_scores_are_divided():
    logits = np.array([[0.0, 4.0, 3.0]], np.float32)
    assert _next_token(logits, [1], 2.0) == 2


def test_negative_scores_are_multiplied():
    logits = np.array([[-9.0, -1.0, -1.5]], np.float32)
    assert _next_token(logits, [1], 2.0) == 2


def test_repeats_in_the_history_do_not_compound():
    logits = np.array([[0.0, 4.0, 1.9]], np.float32)
    assert _next_token(logits, [1, 1, 1, 1], 2.0) == 1  # 4 / 2, not 4 / 16


def test_penalty_below_one_favours_repeats_as_upstream_allows():
    logits = np.array([[0.0, 2.0, 3.0]], np.float32)
    assert _next_token(logits, [1], 0.5) == 1


def test_the_whole_history_is_penalized_not_just_the_last_token():
    logits = np.array([[0.0, 4.0, 3.0, 2.5]], np.float32)
    assert _next_token(logits, [1, 2], 2.0) == 3  # 2.5 beats 4 / 2 and 3 / 2


def test_only_generated_tokens_are_penalized():
    logits = np.array([[0.0, 4.0, 3.0]], np.float32)
    assert _next_token(logits, [2], 2.0) == 1


def test_scores_are_not_modified_in_place():
    logits = np.array([[0.0, 4.0, 3.0]], np.float32)
    _next_token(logits, [1], 2.0)
    np.testing.assert_array_equal(logits, [[0.0, 4.0, 3.0]])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_torch_logits_work_in_every_precision(dtype):
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[0.0, 4.0, 3.0]], dtype=getattr(torch, dtype))
    assert _next_token(logits, [1], 2.0) == 2


def test_bos_is_not_part_of_the_penalized_history():
    # Token 0 is BOS here: it stays the best choice, so it isn't being penalized.
    result = transcribe(model([5.0, 0.0, 0.0, 4.0, 3.0], limit=1), repetition_penalty=2.0)
    assert result.token_ids == [0]


def test_penalty_below_one_is_accepted():
    assert transcribe(model([0.0, 0.0, 0.0, 4.0, 3.0]), repetition_penalty=0.5).token_ids == [3] * 3


def test_default_keeps_the_exact_greedy_output():
    result = transcribe(model([0.0, 0.0, 0.0, 4.0, 3.0]))
    assert result.token_ids == [3, 3, 3] and result.stop_reason == "eos"


def test_penalty_of_one_is_the_default():
    scores = [0.0, 0.0, 0.0, 4.0, 3.0]
    assert transcribe(model(scores), repetition_penalty=1).token_ids == transcribe(
        model(scores)).token_ids


def test_penalty_steers_the_onnx_decode_loop():
    result = transcribe(model([0.0, 0.0, 0.0, 4.0, 3.0]), repetition_penalty=2.0)
    assert result.token_ids == [3, 4, 3]  # 3 is penalized to 2, then 4 to 1.5 and 3 to 2


@pytest.mark.parametrize("value", [0, -1, -0.5, float("nan"), float("inf"), True, np.bool_(True),
                                   "1.2", None, 10**400])
def test_invalid_penalties_are_rejected(value):
    with pytest.raises(ValueError, match="repetition_penalty"):
        model([0.0, 0.0, 0.0]).transcribe(np.zeros(1600, np.float32), repetition_penalty=value)


def test_integer_and_numpy_penalties_are_accepted():
    for value in (2, np.int64(2), np.float32(2.0)):
        assert transcribe(model([0.0, 0.0, 0.0, 4.0, 3.0]), repetition_penalty=value).token_ids

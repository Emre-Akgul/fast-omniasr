"""Generation protocol tests independent of heavyweight model assets."""

import json

import numpy as np
import pytest

from fast_omniasr.llm import OmniASRLLM

torch = pytest.importorskip("torch")


class Text:
    def decode(self, ids):
        return " ".join(map(str, ids))


def runtime(sequence):
    model = object.__new__(OmniASRLLM)
    model.device = torch.device("cpu")
    model.config = {"bos_idx": 0, "eos_idx": 2, "max_seq_len": 8192}
    model.tokenizer = Text()
    model.encoder = lambda audio: torch.zeros(1, 4, 8)
    calls = []

    def decoder(context, ids):
        calls.append(ids.tolist()[0])
        logits = torch.full((1, 10), -10.0)
        logits[0, sequence[len(calls) - 1]] = 10
        return logits

    model.decoder = decoder
    return model, calls


def test_greedy_preserves_repetitions_and_stops_at_eos():
    model, calls = runtime([5, 5, 2])
    result = model.transcribe(np.zeros(1600, dtype=np.float32))
    assert result.token_ids == [5, 5]  # No CTC collapse.
    assert result.text == "5 5"
    assert result.stop_reason == "eos"
    assert calls == [[0], [0, 5], [0, 5, 5]]


def test_length_limit_is_not_successful_completion():
    model, _ = runtime([5])
    assert model.transcribe(np.zeros(1600), max_new_tokens=1).stop_reason == "max_new_tokens"


def test_context_limit():
    model, _ = runtime([5])
    model.config["max_seq_len"] = 8
    assert model.transcribe(np.zeros(1600)).stop_reason == "context_limit"


@pytest.mark.parametrize("limit", [0, -1, True, 1.2])
def test_invalid_generation_limit(limit):
    model, _ = runtime([])
    with pytest.raises(ValueError, match="positive integer"):
        model.transcribe(np.zeros(1600), max_new_tokens=limit)


def test_long_audio_rejected():
    model, _ = runtime([])
    with pytest.raises(ValueError, match="30 seconds"):
        model.transcribe(np.zeros(480001))


def test_nonfinite_logits_rejected():
    model, _ = runtime([])
    model.decoder = lambda context, ids: torch.full((1, 10), float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        model.transcribe(np.zeros(1600))


def test_artifact_model_guard(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 1, "model": "CTC"}))
    with pytest.raises(ValueError, match="Unsupported"):
        OmniASRLLM(tmp_path)


def test_cached_runtime_prefills_once_and_only_steps_one_token():
    model, _ = runtime([])
    model.cached = True
    calls = []

    class Cached:
        def prefill(self, context, ids):
            calls.append(("prefill", ids.tolist()))
            return self.result(5, 5)

        def decode_step(self, ids, cache):
            calls.append(("step", ids.tolist(), cache))
            return self.result(5 if cache == 5 else 2, cache + 1)

        def result(self, token, cache):
            logits = torch.full((1, 10), -10.0)
            logits[0, token] = 10
            return logits, cache

    model.decoder = Cached()
    for _ in range(2):
        result = model.transcribe(np.zeros(1600, dtype=np.float32))
        assert result.token_ids == [5, 5]
        assert result.stop_reason == "eos"
    assert calls == [("prefill", [[0]]), ("step", [[5]], 5), ("step", [[5]], 6)] * 2


def test_onnx_runtime_prefills_once_and_only_steps_one_token():
    model, _ = runtime([])
    model.backend = "onnx"
    calls = []

    class Backend:
        def encode(self, waveform):
            calls.append(("encode", waveform.shape))
            return np.zeros((1, 4, 8), dtype=np.float32)

        def prefill(self, context, token):
            calls.append(("prefill", token.tolist()))
            return result(5), [np.zeros((1, 5, 1, 1), dtype=np.float32)] * 24

        def decode_step(self, token, cache):
            calls.append(("step", token.tolist(), cache[0].shape[1]))
            next_token = 5 if len(calls) == 3 else 2
            return result(next_token), [np.zeros((1, 6, 1, 1), dtype=np.float32)] * 24

    def result(token):
        logits = np.full((1, 10), -10.0, dtype=np.float32)
        logits[0, token] = 10.0
        return logits

    model.onnx = Backend()
    transcription = model.transcribe(np.zeros(1600, dtype=np.float32))
    assert transcription.token_ids == [5, 5]
    assert transcription.stop_reason == "eos"
    assert calls == [
        ("encode", (1, 1600)),
        ("prefill", [[0]]),
        ("step", [[5]], 5),
        ("step", [[5]], 6),
    ]

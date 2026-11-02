"""OmniASRLLM concurrency guarantees (issue #2): interleaved requests and first use."""

import json
import threading
import time
import warnings

import numpy as np
import pytest

from fast_omniasr.llm import OmniASRLLM

EOS = 2


class SharedStateBackend:
    """Stand-in for ONNXLLMBackend that, like the real one (lazy sessions, low_memory
    phasing, bound device buffers), keeps per-request state on the instance between
    encode, prefill and decode_step. Each request's tokens derive from its audio length,
    so unsynchronized interleaving produces mixed transcripts."""

    def __init__(self, delay=0.002):
        self.delay = delay
        self.active = self.max_active = 0
        self.closed = False

    def _enter(self):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)

    def _leave(self):
        self.active -= 1

    def encode(self, audio):
        self._enter()
        self.request = audio.shape[1] // 160  # per-request identity, kept on the instance
        self._leave()
        return np.zeros((1, 4, 8), np.float32)

    def _logits(self, token):
        logits = np.full((1, 1000), -10.0, np.float32)
        logits[0, token] = 10.0
        return logits

    def prefill(self, context, bos):
        self._enter()
        self.steps = 0
        token = self.request
        self._leave()
        return self._logits(token), []

    def decode_step(self, token, cache):
        self._enter()
        self.steps += 1
        out = EOS if self.steps >= 3 else self.request
        self._leave()
        return self._logits(out), cache

    def close(self):
        self.closed = True


class Tokens:
    def decode(self, ids):
        return " ".join(map(str, ids))


def make_model(backend, directory=None, config=None):
    model = object.__new__(OmniASRLLM)
    model.closed = False
    model.backend = "onnx"
    model.onnx = backend
    model.tokenizer = Tokens()
    model.directory = directory
    model.config = {"bos_idx": 0, "eos_idx": EOS, "max_seq_len": 8192, **(config or {})}
    return model


def _audio(i):
    return np.random.default_rng(i).standard_normal(16000 + 160 * i).astype(np.float32)


def _transcribe(model, i):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # no-lang quality warning
        return model.transcribe(_audio(i))


def test_concurrent_transcriptions_match_sequential():
    backend = SharedStateBackend()
    model = make_model(backend)
    ids = list(range(3, 11))
    sequential = {i: _transcribe(model, i).token_ids for i in ids}
    assert sequential[3] == [103, 103, 103]  # tokens are the request's identity
    results, errors = {}, []
    barrier = threading.Barrier(len(ids))

    def worker(i):
        try:
            barrier.wait()
            for round_ in range(3):
                results[(i, round_)] = _transcribe(model, i).token_ids
        except Exception as exc:  # noqa: BLE001 -- surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert backend.max_active == 1
    for (i, _), tokens in results.items():
        assert tokens == sequential[i]


def test_close_waits_for_in_flight_transcription():
    backend = SharedStateBackend(delay=0.05)
    model = make_model(backend)
    result = {}
    worker = threading.Thread(target=lambda: result.update(out=_transcribe(model, 5)))
    worker.start()
    deadline = time.monotonic() + 10
    while backend.active == 0:
        assert time.monotonic() < deadline, "transcription never started"
        time.sleep(0.001)
    model.close()  # blocks until the transcription above has finished
    worker.join()
    assert result["out"].token_ids == [105, 105, 105] and backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        _transcribe(model, 5)


def test_concurrent_first_use_loads_language_table_once(tmp_path, monkeypatch):
    (tmp_path / "languages.json").write_text(json.dumps({"eng_latn": 0, "tur_latn": 1}))
    np.save(tmp_path / "languages.npy", np.arange(16, dtype=np.float32).reshape(2, 8))
    model = make_model(SharedStateBackend(), tmp_path, {
        "languages": {"mapping": "languages.json", "embeddings": "languages.npy"}})
    loads = []
    real_load = np.load

    def slow_load(*args, **kwargs):
        loads.append(1)
        time.sleep(0.05)  # widen the race window
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", slow_load)
    results, errors = [], []
    barrier = threading.Barrier(8)

    def worker(code):
        try:
            barrier.wait()
            results.append((code, model.language_embedding(code)))
        except Exception as exc:  # noqa: BLE001 -- surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(c,))
               for c in ["eng_Latn", "tur_Latn"] * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == [] and len(loads) == 1
    for code, row in results:
        expected = np.arange(8, dtype=np.float32) + (0 if code == "eng_Latn" else 8)
        np.testing.assert_array_equal(row, expected)

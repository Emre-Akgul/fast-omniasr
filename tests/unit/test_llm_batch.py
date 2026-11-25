"""Greedy batching with different prefix lengths and independent stop conditions."""

from types import SimpleNamespace

import numpy as np

from fast_omniasr.llm import OmniASRLLM


def test_finished_rows_leave_the_cache_and_results_keep_input_order():
    model = object.__new__(OmniASRLLM)
    model.config = {"bos_idx": 0, "eos_idx": 2, "max_seq_len": 8}
    model.tokenizer = SimpleNamespace(decode=lambda ids: " ".join(map(str, ids)))
    calls = []

    def run_batch(context, tokens, cache, padding, past):
        calls.append(tokens.shape[0])
        logits = np.zeros((tokens.shape[0], 8), dtype=np.float32)
        if not past:
            assert padding.tolist() == [1, 0, 2]
            assert context[:, :, 0].tolist() == [[0, 1, 1], [2, 2, 2], [0, 0, 3]]
            logits[np.arange(3), [2, 5, 6]] = 1
            cache = [np.arange(3)[:, None]]
        else:
            assert past == 4
            assert padding.tolist() == [0]
            assert cache[0].tolist() == [[1]]
            assert tokens.tolist() == [[5]]
            logits[0, 7] = 1
        return logits, cache

    model.onnx = SimpleNamespace(run_batch=run_batch,
                                 select_cache_rows=lambda cache, rows: [v[rows] for v in cache])
    contexts = [np.full((n, 1), value) for value, n in [(1, 2), (2, 3), (3, 1)]]
    results = model._decode_batch(contexts, [4, 8, 1], 1)
    assert calls == [3, 1]
    assert [r.token_ids for r in results] == [[], [5, 7], [6]]
    assert [r.stop_reason for r in results] == ["eos", "context_limit", "max_new_tokens"]


def test_legacy_model_uses_single_clip_path():
    model = object.__new__(OmniASRLLM)
    model.backend = "onnx"
    model.onnx = SimpleNamespace(batched=False)
    model.transcribe = lambda clip, **kw: (clip, kw["max_new_tokens"])
    assert model.transcribe_batch(["a.wav", "b.wav"], max_new_tokens=[1, 3]) == [
        ("a.wav", 1), ("b.wav", 3)]

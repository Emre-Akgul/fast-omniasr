"""Configurable LLM beam search matches upstream's algorithm and leaves greedy as it was (#41)."""

import numpy as np
import pytest

from fast_omniasr.llm import OmniASRLLM

EOS = 2
VOCAB = 12


class Text:
    def decode(self, ids):
        return " ".join(map(str, ids))


def random_scores(prefix, seed):
    """Fixed logits for each prefix; EOS grows likelier as the prefix gets longer."""
    logits = np.random.default_rng([seed, *prefix]).normal(0, 2, VOCAB).astype(np.float32)
    logits[EOS] += 0.6 * len(prefix) - 2
    return logits


class Onnx:
    """Fake batch-one decoder whose cache is the token prefix it has consumed, so a beam
    that resumed from the wrong parent's cache gets that parent's scores."""

    def __init__(self, scores):
        self.scores, self.steps = scores, []

    def encode(self, waveform):
        return np.zeros((1, 4, 8), np.float32)

    def prefill(self, context, token):
        return self.scores(())[None], ()

    def decode_step(self, token, cache):
        assert token.shape == (1, 1) and token.dtype == np.int64
        prefix = (*cache, int(token[0, 0]))
        self.steps.append(prefix)
        return self.scores(prefix)[None], prefix


def model(scores):
    instance = object.__new__(OmniASRLLM)
    instance.backend = "onnx"
    instance.config = {"bos_idx": 0, "eos_idx": EOS, "max_seq_len": 8192}
    instance.tokenizer = Text()
    instance.onnx = Onnx(scores)
    return instance


def transcribe(instance, **options):
    return instance.transcribe(np.zeros(1600, np.float32), lang=None, **options)


def log_softmax(values):
    values = np.asarray(values, np.float64) - np.max(values)
    return values - np.log(np.exp(values).sum())


def upstream(scores, nbest, budget, length_norm=False):
    """omnilingual_asr's Wav2Vec2LlamaBeamSearchSeq2SeqGenerator for one clip, as fixed-size
    arrays: `nbest` rows from the start, duplicates scored -1e6, ended rows offered only
    EOS at their frozen score, and the top row after the last step returned."""
    rows = [[] for _ in range(nbest)]
    score = np.full(nbest, -1e6)
    score[0] = 0.0
    ended = np.zeros(nbest, bool)
    for t in range(budget):
        log_probs = np.stack([log_softmax(scores(tuple(row))) for row in rows])
        if length_norm and t > 0:
            candidates = (score[:, None] * t + log_probs) / (t + 1)
        else:
            candidates = score[:, None] + log_probs
        candidates[ended] = -np.inf
        candidates[ended, EOS] = score[ended]
        flat = candidates.reshape(-1)
        top = np.argsort(-flat, kind="stable")[:nbest]
        parents, tokens = np.divmod(top, VOCAB)
        rows = [rows[p] if ended[p] else [*rows[p], int(k)] for p, k in zip(parents, tokens)]
        score = np.where(ended[parents], score[parents], flat[top])
        ended = ended[parents] | (tokens == EOS)
        rows = [row[:-1] if row[-1:] == [EOS] else row for row in rows]
        if ended.all():
            break
    return rows[0], bool(ended[0])


@pytest.mark.parametrize("length_norm", [False, True])
@pytest.mark.parametrize("beam_size", [2, 3, 5])
@pytest.mark.parametrize("seed", range(12))
def test_matches_upstream_beam_search(seed, beam_size, length_norm):
    def scores(prefix):
        return random_scores(prefix, seed)

    result = transcribe(model(scores), beam_size=beam_size, length_norm=length_norm,
                        max_new_tokens=8)
    tokens, ended = upstream(scores, beam_size, 8, length_norm)
    assert result.token_ids == tokens
    assert result.stop_reason == ("eos" if ended else "max_new_tokens")


def table(probabilities):
    """Logits from {prefix: {token: probability}}; unlisted tokens are near impossible."""
    def scores(prefix):
        logits = np.full(VOCAB, -30.0, np.float32)
        for token, probability in probabilities.get(prefix, {EOS: 1.0}).items():
            logits[token] = np.log(probability)
        return logits
    return scores


# Greedy takes 3 (0.6), then 5 (0.34): 0.204. Beam search finds 4 then EOS: 0.38.
GARDEN_PATH = table({(): {3: 0.6, 4: 0.4}, (3,): {5: 0.34, 6: 0.33, EOS: 0.33},
                     (4,): {EOS: 0.95, 6: 0.05}})


def test_beam_search_finds_what_greedy_misses():
    assert transcribe(model(GARDEN_PATH)).token_ids == [3, 5]
    result = transcribe(model(GARDEN_PATH), beam_size=2)
    assert result.token_ids == [4]
    assert result.text == "4"
    assert result.stop_reason == "eos"


def test_beam_size_one_is_the_greedy_path():
    def scores(prefix):
        return random_scores(prefix, 7)

    greedy, single = model(scores), model(scores)
    expected = transcribe(greedy, max_new_tokens=20)
    assert transcribe(single, beam_size=1, max_new_tokens=20) == expected
    assert single.onnx.steps == greedy.onnx.steps


def test_length_norm_prefers_the_longer_transcript():
    # Sum: [4] scores log(0.45) = -0.80 against [3, 5, 6] at -0.91. Mean per token
    # (EOS included): -0.40 against -0.23.
    scores = table({(): {3: 0.55, 4: 0.45}, (3,): {5: 0.9, EOS: 0.1},
                    (3, 5): {6: 0.9, 7: 0.1}, (3, 5, 6): {EOS: 0.9, 7: 0.1}, (4,): {EOS: 1.0}})
    assert transcribe(model(scores), beam_size=2).token_ids == [4]
    assert transcribe(model(scores), beam_size=2, length_norm=True).token_ids == [3, 5, 6]


def test_the_token_budget_bounds_every_hypothesis():
    def scores(prefix):  # never ends
        logits = np.zeros(VOCAB, np.float32)
        logits[EOS] = -50.0
        return logits

    instance = model(scores)
    result = transcribe(instance, beam_size=3, max_new_tokens=4)
    assert len(result.token_ids) == 4
    assert result.stop_reason == "max_new_tokens"
    # The last step needs no decoder run, as in greedy decoding.
    assert max(len(prefix) for prefix in instance.onnx.steps) == 3


def test_context_limit_is_reported():
    instance = model(lambda prefix: np.zeros(VOCAB, np.float32))
    instance.config["max_seq_len"] = 9  # 4 context rows leave a budget of 2
    result = transcribe(instance, beam_size=2)
    assert len(result.token_ids) == 2
    assert result.stop_reason == "context_limit"


def test_the_repetition_penalty_applies_to_each_hypothesis():
    def scores(prefix):
        logits = np.zeros(VOCAB, np.float32)
        logits[3], logits[4] = 4.0, 3.0
        logits[EOS] = 100.0 if len(prefix) >= 3 else -1.0
        return logits

    assert transcribe(model(scores), beam_size=2).token_ids == [3, 3, 3]
    assert transcribe(model(scores), beam_size=2, repetition_penalty=2.0).token_ids == [3, 4, 3]


def test_nonfinite_logits_rejected():
    instance = model(lambda prefix: np.full(VOCAB, np.nan, np.float32))
    with pytest.raises(RuntimeError, match="non-finite"):
        transcribe(instance, beam_size=2)


@pytest.mark.parametrize("beam_size", [0, -1, True, 2.0, "5", None])
def test_invalid_beam_size(beam_size):
    with pytest.raises(ValueError, match="beam_size"):
        transcribe(model(GARDEN_PATH), beam_size=beam_size)
    with pytest.raises(ValueError, match="beam_size"):
        model(GARDEN_PATH).transcribe_long("missing.wav", beam_size=beam_size)


@pytest.mark.parametrize("length_norm", [1, "yes", None])
def test_invalid_length_norm(length_norm):
    with pytest.raises(TypeError, match="length_norm"):
        transcribe(model(GARDEN_PATH), beam_size=2, length_norm=length_norm)


def test_transcribe_long_beam_searches_every_chunk():
    result = model(GARDEN_PATH).transcribe_long(np.zeros(3 * 16000, np.float32),
                                                chunk_length=1, lang=None, beam_size=2)
    assert [chunk.token_ids for chunk in result.chunks] == [[4]] * 3
    assert result.text == "4 4 4"


@pytest.mark.parametrize("cached", [False, True])
def test_torch_baseline_beam_search_matches_onnx(cached):
    torch = pytest.importorskip("torch")

    def scores(prefix):
        return random_scores(prefix, 3)

    def logits(prefix):
        return torch.from_numpy(scores(tuple(prefix)))[None]

    class Decoder:
        def __call__(self, context, ids):
            return logits(ids[0, 1:].tolist())

        def prefill(self, context, ids):
            return logits([]), ()

        def decode_step(self, ids, cache):
            prefix = (*cache, int(ids[0, 0]))
            return logits(prefix), prefix

    instance = object.__new__(OmniASRLLM)
    instance.device = torch.device("cpu")
    instance.config = {"bos_idx": 0, "eos_idx": EOS, "max_seq_len": 8192}
    instance.tokenizer = Text()
    instance.encoder = lambda audio: torch.zeros(1, 4, 8)
    instance.decoder = Decoder()
    instance.cached = cached
    result = transcribe(instance, beam_size=4, max_new_tokens=10)
    assert result == transcribe(model(scores), beam_size=4, max_new_tokens=10)

"""CTC beam search with an optional language model and hotwords (#43)."""

import itertools
import math
import sys

import numpy as np
import pytest

from fast_omniasr import BeamSearchDecoder, OmniASR
from fast_omniasr.beam_search import KenLM, _arpa_unigrams, align
from fast_omniasr.decoder import greedy_token_ids, greedy_tokens
from fast_omniasr.tokenizer import Tokenizer

# Character-level like the OmniASR vocabularies: <s> is the CTC blank, " " a token of its own.
PIECES = ["<s>", "<pad>", "</s>", "<unk>", " ", "c", "a", "t", "r", "n", "f", "l", "y"]
ID = {piece: index for index, piece in enumerate(PIECES)}


class Processor:
    def get_piece_size(self):
        return len(PIECES)

    def id_to_piece(self, token):
        return PIECES[token]

    def is_control(self, token):
        return token in (0, 1, 2)

    def is_unknown(self, token):
        return token == 3

    def decode(self, ids):
        return "".join(PIECES[i] for i in ids if i > 3).strip()


def tokenizer():
    instance = object.__new__(Tokenizer)
    instance._processor = Processor()
    return instance


def frames(*distributions):
    """Logits [1, frames, vocabulary] from {piece: probability} per frame; the rest of each
    frame's mass is spread over the unused pieces."""
    rows = []
    for distribution in distributions:
        row = np.full(len(PIECES), 1e-6)
        for piece, probability in distribution.items():
            row[ID[piece]] = probability
        rows.append(np.log(row / row.sum()))
    return np.array(rows, dtype=np.float32)[None]


def spelled(*pieces):
    """Frames that clearly spell `pieces`, a blank between each (so repeats survive)."""
    rows = []
    for piece in pieces:
        rows += [piece if isinstance(piece, dict) else {piece: 0.98}, {"<s>": 0.98}]
    return frames(*rows)


CAT_OR_CAR = spelled("c", "a", {"t": 0.45, "r": 0.53})


def text(ids):
    return tokenizer().decode(ids)


def decode(logits, **options):
    return text(BeamSearchDecoder(**options).token_ids(logits, tokenizer()))


def test_beam_search_sums_alignments_where_greedy_takes_the_best_path():
    # Greedy picks blank, blank (0.16); "a" has three paths: aa, a-, -a (0.4025 together).
    logits = frames({"<s>": 0.4, "a": 0.35, "t": 0.25}, {"<s>": 0.4, "a": 0.35, "t": 0.25})
    assert greedy_token_ids(logits) == []
    assert BeamSearchDecoder(4).token_ids(logits, tokenizer()) == [ID["a"]]


def _best_labelling(log_probs, usable):
    """The most probable labelling, by summing every path (exhaustive, tiny inputs only)."""
    totals = {}
    for path in itertools.product(usable, repeat=log_probs.shape[0]):
        labels = tuple(token for index, token in enumerate(path)
                       if token != 0 and (index == 0 or path[index - 1] != token))
        probability = math.exp(sum(log_probs[t, token] for t, token in enumerate(path)))
        totals[labels] = totals.get(labels, 0.0) + probability
    return max(totals, key=totals.get)


@pytest.mark.parametrize("seed", range(8))
def test_a_wide_beam_finds_the_most_probable_labelling(seed):
    rng = np.random.default_rng(seed)
    usable = [ID["<s>"], ID["c"], ID["a"], ID["t"]]
    logits = np.full((1, 5, len(PIECES)), -30.0)
    logits[0][:, usable] = rng.normal(scale=1.5, size=(5, len(usable)))
    log_probs = logits[0] - np.log(np.exp(logits[0]).sum(axis=-1, keepdims=True))
    expected = list(_best_labelling(log_probs, usable))
    assert BeamSearchDecoder(64).token_ids(logits, tokenizer()) == expected


def test_clear_audio_decodes_like_greedy_with_the_same_times():
    logits = spelled("c", "a", "t", " ", "c", "a", "r")
    decoder = BeamSearchDecoder(4)
    assert decoder.token_ids(logits, tokenizer()) == greedy_token_ids(logits)
    assert decoder.tokens(logits, tokenizer()) == greedy_tokens(logits)


def test_repeated_letters_need_a_blank_between_them():
    logits = frames({"a": 0.98}, {"a": 0.98}, {"<s>": 0.98}, {"a": 0.98})
    assert BeamSearchDecoder(4).token_ids(logits, tokenizer()) == [ID["a"], ID["a"]]


def test_control_and_unknown_tokens_are_never_emitted():
    logits = frames({"<pad>": 0.5, "a": 0.3}, {"<unk>": 0.6, "<s>": 0.3})
    assert decode(logits) == "a"


def test_a_frame_of_only_unusable_tokens_still_extends_the_beam():
    logits = np.full((1, 3, len(PIECES)), -40.0, np.float32)
    logits[0, 0, ID["a"]] = 0.0
    logits[0, 1, ID["<pad>"]] = 0.0  # every usable token is below the pruning threshold
    logits[0, 1, ID["t"]] = -20.0
    logits[0, 2, ID["<s>"]] = 0.0
    assert decode(logits) == "at"


def test_empty_and_invalid_logits():
    assert BeamSearchDecoder().token_ids(np.empty((1, 0, len(PIECES))), tokenizer()) == []
    with pytest.raises(ValueError, match="NaN"):
        BeamSearchDecoder().token_ids(np.full((1, 2, len(PIECES)), np.nan), tokenizer())
    with pytest.raises(ValueError, match="13 entries"):
        BeamSearchDecoder().token_ids(np.zeros((1, 2, 5)), tokenizer())


def test_forced_alignment_gives_each_token_its_frames():
    logits = frames({"<s>": 0.9}, {"c": 0.9}, {"c": 0.9}, {"<s>": 0.9}, {"a": 0.9},
                    {"a": 0.5, "<s>": 0.4}, {"<s>": 0.9}, {"a": 0.9})
    log_probs = logits[0].astype(np.float64)
    ids = [ID["c"], ID["a"], ID["a"]]
    assert align(log_probs, ids) == [(ID["c"], 1, 3), (ID["a"], 4, 6), (ID["a"], 7, 8)]
    assert align(log_probs, []) == []
    with pytest.raises(ValueError, match="can't be aligned"):
        align(log_probs[:2], ids)


class WordLM:
    """A language model that likes some words and dislikes everything else."""

    def __init__(self, liked):
        self.liked, self.calls = liked, []

    def begin(self):
        return ()

    def score(self, state, word):
        self.calls.append(word)
        return (0.0 if word in self.liked else -5.0), state + (word,)

    def end(self, state):
        return 0.0


def test_a_language_model_overrules_a_close_acoustic_call():
    assert decode(CAT_OR_CAR) == "car"
    assert decode(CAT_OR_CAR, lm=WordLM({"cat"})) == "cat"
    assert decode(CAT_OR_CAR, lm=WordLM({"cat"}), lm_weight=0.0) == "car"


def test_the_language_model_scores_whole_words():
    lm = WordLM({"cat", "car"})
    BeamSearchDecoder(4, lm=lm).token_ids(spelled("c", "a", "t", " ", "c", "a", "r"),
                                         tokenizer())
    assert set(lm.calls) == {"cat", "car"}


def test_the_word_bonus_favours_splitting_words():
    canfly = spelled("c", "a", "n", {" ": 0.4, "<s>": 0.58}, "f", "l", "y")
    lm = WordLM({"can", "fly", "canfly"})
    assert decode(canfly, lm=lm, word_bonus=0.0) == "canfly"
    assert decode(canfly, lm=lm, word_bonus=3.0) == "can fly"


class PrefixLM(WordLM):
    """Also says, mid-word, whether the word can still become one of `known`."""

    def __init__(self, liked, known):
        super().__init__(liked)
        self.known = known

    def partial_score(self, prefix):
        return 0.0 if any(word.startswith(prefix) for word in self.known) else -100.0


def test_a_word_that_cannot_become_known_is_dropped_before_it_ends():
    # Both words score the same once complete; only the mid-word check keeps "car" out of
    # a one-hypothesis beam.
    assert decode(CAT_OR_CAR, beam_size=1, lm=WordLM({"cat", "car"})) == "car"
    assert decode(CAT_OR_CAR, beam_size=1, lm=PrefixLM({"cat", "car"}, {"cat"})) == "cat"


def test_hotwords_tip_a_close_call():
    assert decode(CAT_OR_CAR, hotwords=["cat"]) == "cat"
    assert decode(CAT_OR_CAR, hotwords=["CAT"]) == "cat"  # case-insensitive
    assert decode(CAT_OR_CAR, hotwords=["cat"], hotword_weight=0.0) == "car"
    assert decode(CAT_OR_CAR, hotwords=["catalog"]) == "car"  # a longer word isn't spelled


def test_hotword_phrases_span_words():
    canfly = spelled("c", "a", "n", {" ": 0.3, "<s>": 0.68}, "f", "l", "y")
    assert decode(canfly) == "canfly"
    assert decode(canfly, hotwords=["can  fly"]) == "can fly"


def test_hotwords_only_start_at_a_word_boundary():
    # "at" inside "cat" must not count as the hotword "at".
    logits = spelled({"c": 0.53, "<s>": 0.45}, "a", "t")
    assert decode(logits) == "cat"
    assert decode(logits, hotwords=["at"], hotword_weight=1.0) == "at"
    assert decode(spelled("c", "a", "t"), hotwords=["at"]) == "cat"


@pytest.mark.parametrize("options, error", [
    ({"beam_size": 0}, ValueError), ({"beam_size": True}, ValueError),
    ({"beam_size": 2.5}, ValueError), ({"lm_weight": math.nan}, ValueError),
    ({"word_bonus": "1"}, ValueError), ({"hotword_weight": math.inf}, ValueError),
    ({"hotwords": "cat"}, TypeError), ({"hotwords": ["cat", " "]}, ValueError),
    ({"hotwords": [3]}, ValueError), ({"lm": object()}, TypeError),
])
def test_invalid_options_are_refused(options, error):
    with pytest.raises(error):
        BeamSearchDecoder(**options)


def test_a_missing_language_model_file_is_reported(tmp_path):
    pytest.importorskip("kenlm")
    with pytest.raises(FileNotFoundError, match="Language model not found"):
        BeamSearchDecoder(lm=tmp_path / "missing.arpa")


def test_kenlm_is_an_optional_dependency(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "kenlm", None)
    with pytest.raises(ImportError, match=r"fast-omniasr\[lm\]"):
        KenLM(tmp_path / "lm.arpa")


# A tiny bigram model (KenLM needs at least bigrams): "cat" is likely, "car" isn't.
ARPA = """\\data\\
ngram 1=6
ngram 2=2

\\1-grams:
-1.5\t<unk>\t0
-99\t<s>\t-0.3
-0.7\t</s>\t0
-0.3\tcat\t-0.2
-3.0\tcar\t-0.2
-1.0\tfly\t-0.2

\\2-grams:
-0.1\t<s> cat
-0.2\tcat </s>

\\end\\
"""


def test_a_kenlm_model_is_loaded_from_a_path(tmp_path):
    pytest.importorskip("kenlm")
    path = tmp_path / "tiny.arpa"
    path.write_text(ARPA)
    decoder = BeamSearchDecoder(4, lm=str(path))
    assert isinstance(decoder.lm, KenLM)
    assert text(decoder.token_ids(CAT_OR_CAR, tokenizer())) == "cat"
    lm = decoder.lm
    assert lm.score(lm.begin(), "cat")[0] == pytest.approx(-0.1 * math.log(10), abs=1e-4)
    assert lm.score(lm.begin(), "car")[0] == pytest.approx(-3.3 * math.log(10), abs=1e-4)
    state = lm.score(lm.begin(), "cat")[1]
    assert lm.end(state) == pytest.approx(-0.2 * math.log(10), abs=1e-4)
    # Unknown words: <unk>'s probability (after <s>'s backoff) minus the penalty.
    unknown = (-0.3 - 1.5) * math.log(10) - 10.0
    assert lm.score(lm.begin(), "dog")[0] == pytest.approx(unknown, abs=1e-4)
    assert (lm.partial_score("ca"), lm.partial_score("fl"), lm.partial_score("cx")) == (
        0.0, 0.0, -10.0)
    gentle = KenLM(path, unknown_word_penalty=0.0)
    assert gentle.score(gentle.begin(), "dog")[0] == pytest.approx(unknown + 10.0, abs=1e-4)


def test_the_vocabulary_is_read_from_arpa_files_only(tmp_path):
    arpa, binary = tmp_path / "tiny.arpa", tmp_path / "tiny.bin"
    arpa.write_text(ARPA)
    binary.write_bytes(b"mmap lm http://kheafield.com/code format version 5\n\0\0")
    assert _arpa_unigrams(arpa) == ["car", "cat", "fly"]
    assert _arpa_unigrams(binary) is None


# -- through OmniASR ---------------------------------------------------------------------

class Backend:
    def __init__(self, logits):
        self.logits = logits

    def infer(self, audio):
        return self.logits

    def close(self):
        pass


def model(logits):
    instance = object.__new__(OmniASR)
    instance.backend, instance.tokenizer = Backend(logits), tokenizer()
    return instance


def audio(logits):
    return np.zeros(logits.shape[1] * 320 + 80, np.float32)


def test_greedy_stays_the_default():
    asr = model(CAT_OR_CAR)
    assert asr.transcribe_numpy(audio(CAT_OR_CAR)).text == "car"
    assert asr.transcribe_numpy(audio(CAT_OR_CAR), decoder=None).text == "car"


def test_every_transcribe_method_takes_the_decoder():
    asr = model(CAT_OR_CAR)
    decoder = BeamSearchDecoder(4, hotwords=["cat"])
    clip = audio(CAT_OR_CAR)
    assert asr.transcribe_numpy(clip, decoder=decoder).text == "cat"
    assert [r.text for r in asr.transcribe_batch([clip, clip], decoder=decoder)] == ["cat"] * 2
    assert asr.transcribe_long(clip, decoder=decoder).text == "cat"
    result = asr.transcribe_numpy(clip, decoder=decoder, word_timestamps=True)
    assert [word.word for word in result.words] == ["cat"]
    assert len(result.token_times) == 3


def test_a_wrong_decoder_is_refused():
    with pytest.raises(TypeError, match="BeamSearchDecoder"):
        model(CAT_OR_CAR).transcribe_numpy(audio(CAT_OR_CAR), decoder="beam")


def test_tokenizer_pieces_blank_out_control_tokens():
    assert tokenizer().pieces() == ("", "", "", "") + tuple(PIECES[4:])

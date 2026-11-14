"""CTC prefix beam search with optional word-level n-gram language model and hotwords.

Greedy decoding (decoder.py) stays the default; this module is used only when a
`BeamSearchDecoder` is passed to a transcribe call. The search itself needs only NumPy. A
KenLM language model needs the `kenlm` package (the `lm` extra) and is imported only when one
is loaded.

The OmniASR CTC vocabularies are character-level with a separate space token, so a word ends
at a space (or at the end of the clip). That is where the language model scores it."""

import bisect
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from .decoder import _check_logits
from .options import check_beam_size

DEFAULT_BEAM_SIZE = 8
DEFAULT_LM_WEIGHT = 0.5
DEFAULT_WORD_BONUS = 2.0
DEFAULT_HOTWORD_WEIGHT = 5.0
DEFAULT_UNKNOWN_WORD_PENALTY = 10.0
# Tokens less likely than this in a frame are not tried as extensions in that frame (unless
# none is likelier). Natural-log probability, as everywhere here.
TOKEN_MIN_LOGP = -8.0
_LN10 = math.log(10.0)


class LanguageModel(Protocol):
    """A word-level language model: natural-log probabilities of whole words in context.

    `begin()` is the state at the start of a clip, `score(state, word)` returns the word's
    log-probability and the state after it, and `end(state)` the log-probability of the
    clip ending there. States must not be changed in place: beams share them.

    It may also have `partial_score(prefix)`: a log-probability adjustment (0 or negative)
    for a word still being spelled, applied until the word ends. It lets the search drop a
    hypothesis early once its current word can no longer become a known word."""

    def begin(self): ...

    def score(self, state, word: str) -> tuple[float, object]: ...

    def end(self, state) -> float: ...


class KenLM:
    """A KenLM n-gram model (ARPA or binary) as a LanguageModel. Needs the `kenlm` package
    (`pip install "fast-omniasr[lm]"`). Words are looked up exactly as the CTC model spells
    them, so train the LM on text normalized the same way (for the released models:
    lowercase, without punctuation).

    A word outside the LM's vocabulary scores `<unk>`'s probability minus
    `unknown_word_penalty` (natural log). `<unk>` alone is often more likely than a rare
    real word, and without the penalty the search learns to glue words together into one
    unknown "word". For an ARPA file the vocabulary is also read up front, so a word being
    spelled is penalized as soon as no vocabulary word starts with it; a binary model can't
    list its vocabulary, so there the penalty applies to complete words only."""

    def __init__(self, path: str | Path,
                 unknown_word_penalty: float = DEFAULT_UNKNOWN_WORD_PENALTY):
        self.unknown_word_penalty = _check_weight("unknown_word_penalty", unknown_word_penalty)
        try:
            import kenlm
        except ImportError as exc:
            raise ImportError('A KenLM language model needs the kenlm package: '
                              'pip install "fast-omniasr[lm]"') from exc
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Language model not found: {path}")
        self.path = path
        self._kenlm = kenlm
        self._model = kenlm.Model(str(path))
        self._words = _arpa_unigrams(path)

    def partial_score(self, prefix: str) -> float:
        words = self._words
        if words is None:
            return 0.0
        index = bisect.bisect_left(words, prefix)
        if index < len(words) and words[index].startswith(prefix):
            return 0.0
        return -self.unknown_word_penalty

    def begin(self):
        state = self._kenlm.State()
        self._model.BeginSentenceWrite(state)
        return state

    def score(self, state, word: str):
        out = self._kenlm.State()
        logp = self._model.BaseScore(state, word, out) * _LN10
        if word not in self._model:
            logp -= self.unknown_word_penalty
        return logp, out

    def end(self, state) -> float:
        return self._model.BaseScore(state, "</s>", self._kenlm.State()) * _LN10


def _arpa_unigrams(path: Path) -> list[str] | None:
    """The sorted words of an ARPA file's unigram section, or None for a binary model."""
    with path.open("rb") as file:
        if file.read(64).lstrip()[:6] != b"\\data\\":
            return None
    words = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.strip() == "\\1-grams:":
                break
        for line in file:
            fields = line.split("\t")
            if len(fields) < 2:
                if line.strip():  # the next section
                    break
                continue
            word = fields[1].strip()
            if word not in ("<s>", "</s>", "<unk>"):
                words.append(word)
    return sorted(words)


def _check_weight(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) \
            or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _check_hotwords(hotwords) -> tuple[str, ...]:
    if isinstance(hotwords, str):
        raise TypeError("hotwords must be a list of words or phrases, not a single string")
    phrases = []
    for hotword in hotwords:
        if not isinstance(hotword, str) or not hotword.split():
            raise ValueError(f"hotwords must be non-empty strings, got {hotword!r}")
        phrases.append(" ".join(hotword.split()).lower())
    return tuple(dict.fromkeys(phrases))


class _Vocabulary:
    """What the search needs to know about each token id of one tokenizer."""

    def __init__(self, tokenizer):
        texts = tokenizer.pieces()
        self.size = len(texts)
        self.texts = texts
        self.space = np.array([text.isspace() for text in texts])
        # Control and unknown tokens decode to nothing useful and are never proposed.
        self.usable = np.array([bool(text) for text in texts])


class BeamSearchDecoder:
    """CTC prefix beam search, optionally fused with a word-level language model and biased
    toward hotwords. Pass it as `decoder=` to `OmniASR.transcribe*()`; without it decoding
    is greedy.

    Each hypothesis is ranked by its CTC log-probability (summed over every alignment of
    its tokens) plus `lm_weight` times the LM log-probability of its completed words plus
    `word_bonus` per completed word (the LM terms only apply with `lm`). `lm` is a path to
    a KenLM ARPA/binary file (the `lm` extra) or any object implementing `LanguageModel`;
    it is loaded once, here, and can be shared across calls and threads.

    `hotwords` are words or phrases (case-insensitive) to favour: a hypothesis whose
    current word is a prefix of a hotword gains the matched fraction of `hotword_weight`,
    and every complete occurrence (bounded by spaces) keeps `hotword_weight`. A hotword
    can only win where the acoustic model gives it some probability, but too high a weight
    can put it where it wasn't said.

    `beam_size` hypotheses survive each frame. Tokens below e^TOKEN_MIN_LOGP in a frame
    are not tried there (unless none is likelier); control and unknown tokens are never
    emitted."""

    def __init__(self, beam_size: int = DEFAULT_BEAM_SIZE, *, lm=None,
                 lm_weight: float = DEFAULT_LM_WEIGHT, word_bonus: float = DEFAULT_WORD_BONUS,
                 hotwords=(), hotword_weight: float = DEFAULT_HOTWORD_WEIGHT):
        self.beam_size = check_beam_size(beam_size)
        self.lm_weight = _check_weight("lm_weight", lm_weight)
        self.word_bonus = _check_weight("word_bonus", word_bonus)
        self.hotword_weight = _check_weight("hotword_weight", hotword_weight)
        self.hotwords = _check_hotwords(hotwords)
        if isinstance(lm, (str, Path)):
            lm = KenLM(lm)
        elif lm is not None and not all(callable(getattr(lm, name, None))
                                        for name in ("begin", "score", "end")):
            raise TypeError("lm must be a path to a KenLM model or an object with "
                            "begin(), score(state, word) and end(state)")
        self.lm = lm
        self._vocabularies = {}

    def _vocabulary(self, tokenizer) -> _Vocabulary:
        key = id(tokenizer)
        cached = self._vocabularies.get(key)
        if cached is None or cached[0] is not tokenizer:
            cached = (tokenizer, _Vocabulary(tokenizer))
            self._vocabularies[key] = cached
        return cached[1]

    def token_ids(self, logits: np.ndarray, tokenizer, blank_id: int = 0) -> list[int]:
        """The best token sequence for `logits` [1, frames, vocabulary]."""
        log_probs = _log_softmax(_check_logits(logits)[0])
        return self._search(log_probs, self._vocabulary(tokenizer), blank_id)

    def tokens(self, logits: np.ndarray, tokenizer,
               blank_id: int = 0) -> list[tuple[int, int, int]]:
        """(token id, first frame, frame after its last) for each token of the best
        sequence, as greedy_tokens() gives them: the frames come from the most likely
        alignment of that sequence (CTC forced alignment)."""
        log_probs = _log_softmax(_check_logits(logits)[0])
        ids = self._search(log_probs, self._vocabulary(tokenizer), blank_id)
        return align(log_probs, ids, blank_id)

    # -- search ---------------------------------------------------------------------------

    def _search(self, log_probs: np.ndarray, vocabulary: _Vocabulary, blank_id: int):
        if log_probs.shape[1] != vocabulary.size:
            raise ValueError(f"Logits have {log_probs.shape[1]} classes but the tokenizer "
                             f"has {vocabulary.size} entries")
        if log_probs.shape[0] == 0:
            return []
        usable = vocabulary.usable.copy()
        usable[blank_id] = True
        root = _Context.start(self.lm)
        # prefix -> [log p(ending in blank), log p(ending in a token), context]
        beams = {(): [0.0, -math.inf, root]}
        contexts = {(): root}
        for frame in log_probs:
            candidates = np.flatnonzero((frame >= TOKEN_MIN_LOGP) & usable)
            if candidates.size == 0:  # e.g. a frame that is almost all <pad>
                candidates = np.array([np.where(usable, frame, -np.inf).argmax()])
            following = {}
            for prefix, (blank, token, _) in beams.items():
                total = _logaddexp(blank, token)
                last = prefix[-1] if prefix else None
                for candidate in candidates.tolist():
                    logp = float(frame[candidate])
                    if candidate == blank_id:
                        _add(following, prefix, contexts, 0, total + logp)
                        continue
                    extended = prefix + (candidate,)
                    if extended not in contexts:
                        contexts[extended] = contexts[prefix].extend(
                            vocabulary.texts[candidate], bool(vocabulary.space[candidate]),
                            self)
                    if candidate == last:
                        _add(following, prefix, contexts, 1, token + logp)  # repeat collapses
                        _add(following, extended, contexts, 1, blank + logp)
                    else:
                        _add(following, extended, contexts, 1, total + logp)
            ranked = sorted(following.items(), key=lambda item: -self._score(item[1]))
            beams = dict(ranked[:self.beam_size])
            contexts = {prefix: state[2] for prefix, state in beams.items()}
        finals = [(_logaddexp(blank, token) + context.final_score(self), prefix)
                  for prefix, (blank, token, context) in beams.items()]
        return list(max(finals, key=lambda item: item[0])[1])

    def _score(self, state) -> float:
        return _logaddexp(state[0], state[1]) + state[2].score(self)


def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b == -math.inf:
        return a
    return a + math.log1p(math.exp(b - a))


def _add(following, prefix, contexts, slot, value):
    entry = following.get(prefix)
    if entry is None:
        entry = following[prefix] = [-math.inf, -math.inf, contexts[prefix]]
    entry[slot] = _logaddexp(entry[slot], value)


class _Context:
    """The language-model and hotword state of one prefix. It depends only on the prefix's
    text, so two paths reaching the same prefix share it."""

    __slots__ = ("at_word_start", "hotword_bonus", "lm_logp", "lm_state", "matches",
                 "partial_logp", "word", "words")

    def __init__(self, lm_state, lm_logp, words, word, matches, hotword_bonus, at_word_start,
                 partial_logp=0.0):
        self.lm_state = lm_state
        self.lm_logp = lm_logp  # summed over completed words
        self.partial_logp = partial_logp  # the LM's partial_score() of `word`
        self.words = words  # completed words
        self.word = word  # the word being spelled
        self.matches = matches  # (hotword index, characters matched) still possible
        self.hotword_bonus = hotword_bonus  # from completed hotword occurrences
        self.at_word_start = at_word_start

    @classmethod
    def start(cls, lm):
        return cls(lm.begin() if lm is not None else None, 0.0, 0, "", (), 0.0, True)

    def extend(self, text: str, is_space: bool, decoder: BeamSearchDecoder) -> "_Context":
        lm_state, lm_logp, words, word = self.lm_state, self.lm_logp, self.words, self.word
        if is_space:
            if word:
                lm_state, lm_logp, words = self._complete_word(decoder)
            word = ""
        else:
            word += text
        matches, bonus = self._advance_hotwords(text, decoder)
        partial = getattr(decoder.lm, "partial_score", None)
        partial_logp = partial(word) if word and partial is not None else 0.0
        return _Context(lm_state, lm_logp, words, word, matches, bonus, is_space, partial_logp)

    def _complete_word(self, decoder):
        if decoder.lm is None:
            return None, 0.0, self.words + 1
        logp, state = decoder.lm.score(self.lm_state, self.word)
        return state, self.lm_logp + logp, self.words + 1

    def _advance_hotwords(self, text, decoder):
        hotwords = decoder.hotwords
        bonus, matches, at_start = self.hotword_bonus, self.matches, self.at_word_start
        if not hotwords:
            return (), bonus
        for char in text.lower():
            advanced = []
            for index, count in matches:
                phrase = hotwords[index]
                if count == len(phrase):
                    if char.isspace():
                        bonus += decoder.hotword_weight  # a whole occurrence
                elif phrase[count] == char:
                    advanced.append((index, count + 1))
            if at_start and not char.isspace():
                advanced.extend((index, 1) for index, phrase in enumerate(hotwords)
                                if phrase[0] == char)
            matches, at_start = tuple(advanced), char.isspace()
        return matches, bonus

    def score(self, decoder: BeamSearchDecoder) -> float:
        value = self.hotword_bonus
        if self.matches:
            value += decoder.hotword_weight * max(
                count / len(decoder.hotwords[index]) for index, count in self.matches)
        if decoder.lm is not None:
            value += (decoder.lm_weight * (self.lm_logp + self.partial_logp)
                      + decoder.word_bonus * self.words)
        return value

    def final_score(self, decoder: BeamSearchDecoder) -> float:
        """The score once the clip ends: the last word completes, and so may a hotword."""
        value = self.hotword_bonus
        if any(count == len(decoder.hotwords[index]) for index, count in self.matches):
            value += decoder.hotword_weight
        if decoder.lm is not None:
            lm_state, lm_logp, words = self.lm_state, self.lm_logp, self.words
            if self.word:
                lm_state, lm_logp, words = self._complete_word(decoder)
            lm_logp += decoder.lm.end(lm_state)
            value += decoder.lm_weight * lm_logp + decoder.word_bonus * words
        return value


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def align(log_probs: np.ndarray, ids: list[int], blank_id: int = 0) -> list[tuple[int, int, int]]:
    """CTC forced alignment: the most likely frame path that spells `ids` (Viterbi over
    blank-interleaved states). Returns (token id, first frame, frame after its last) per
    token, like greedy_tokens()."""
    if not ids:
        return []
    frames = log_probs.shape[0]
    states = [blank_id]
    for token in ids:
        states += [token, blank_id]
    states = np.array(states)
    count = len(states)
    emit = log_probs[:, states]
    # A token state may be entered from two states back unless that is the same token.
    skip = np.zeros(count, dtype=bool)
    skip[3::2] = states[3::2] != states[1:-2:2]
    score = np.full(count, -np.inf)
    score[:2] = emit[0, :2]
    back = np.zeros((frames, count), dtype=np.int8)
    for t in range(1, frames):
        from_one = np.r_[-np.inf, score[:-1]]
        from_two = np.where(skip, np.r_[-np.inf, -np.inf, score[:-2]], -np.inf)
        choices = np.stack([score, from_one, from_two])
        back[t] = choices.argmax(axis=0)
        score = choices.max(axis=0) + emit[t]
    state = count - 1 if count == 1 or score[-1] >= score[-2] else count - 2
    if not np.isfinite(score[state]):
        raise ValueError(f"{len(ids)} tokens can't be aligned to {frames} frames")
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= back[t, state]
    spans = []
    for t, state in enumerate(path.tolist()):
        if state % 2 == 0:
            continue
        index = state // 2
        if len(spans) == index:
            spans.append([ids[index], t, t + 1])
        else:
            spans[index][2] = t + 1
    return [tuple(span) for span in spans]

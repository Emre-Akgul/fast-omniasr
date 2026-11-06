"""Greedy CTC decoding: collapse repeats before removing blank frames."""
import numpy as np

# The feature extractor emits one frame per 320 samples, and needs 400 for the first.
FRAME_STRIDE = 320
FRAME_WINDOW = 400


def _check_logits(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] == 0:
        raise ValueError("Expected logits [1, frames, vocabulary]")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinity")
    return logits


def greedy_tokens(logits: np.ndarray, blank_id: int = 0) -> list[tuple[int, int, int]]:
    """(token id, first frame, frame after its last) for each token, in order."""
    ids = _check_logits(logits)[0].argmax(axis=-1)
    if ids.size == 0:
        return []
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], ids.size]
    return [(token, start, end)
            for token, start, end in zip(ids[starts].tolist(), starts.tolist(), ends.tolist())
            if token != blank_id]


def greedy_token_ids(logits: np.ndarray, blank_id: int = 0) -> list[int]:
    return [token for token, _, _ in greedy_tokens(logits, blank_id)]

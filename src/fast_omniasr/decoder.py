"""Greedy CTC decoding: collapse repeats before removing blank frames."""
import numpy as np


def greedy_token_ids(logits: np.ndarray, blank_id: int = 0) -> list[int]:
    logits = np.asarray(logits)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[2] == 0:
        raise ValueError("Expected logits [1, frames, vocabulary]")
    if not np.isfinite(logits).all():
        raise ValueError("Logits contain NaN or infinity")
    ids = logits[0].argmax(axis=-1)
    if ids.size == 0:
        return []
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    return ids[ids != blank_id].tolist()

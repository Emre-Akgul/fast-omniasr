import numpy as np
import pytest

from fast_omniasr.decoder import greedy_token_ids


def test_blank_separates_repeated_characters():
    ids = [0, 2, 2, 0, 2, 3, 3, 0]
    logits = np.eye(4, dtype=np.float32)[ids][None, :, :]
    assert greedy_token_ids(logits) == [2, 2, 3]


def test_empty_frames():
    assert greedy_token_ids(np.empty((1, 0, 4))) == []


def test_all_blank():
    assert greedy_token_ids(np.array([[[1, 0], [1, 0]]])) == []


def test_reject_nan():
    with pytest.raises(ValueError):
        greedy_token_ids(np.full((1, 4, 4), np.nan))

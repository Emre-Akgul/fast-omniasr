from typing import Protocol

import numpy as np


class Backend(Protocol):
    def infer(self, waveform: np.ndarray) -> np.ndarray:
        """Normalized float32 [1, samples] -> logits [1, frames, vocabulary]."""
        ...

    # Optional: infer_batch(waveforms) -> list of logits, one [1, frames, vocabulary] array
    # per [1, samples] waveform. OmniASR falls back to infer() per waveform without it.

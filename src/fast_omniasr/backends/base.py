from typing import Protocol

import numpy as np


class Backend(Protocol):
    def infer(self, waveform: np.ndarray) -> np.ndarray:
        """Normalized float32 [1, samples] -> logits [1, frames, vocabulary]."""
        ...

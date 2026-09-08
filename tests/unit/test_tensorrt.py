import pytest

from fast_omniasr.backends.tensorrt import TensorRTBackend


def test_missing_tensorrt_raises_helpful_import_error():
    with pytest.raises(ImportError, match="fast-omniasr\\[tensorrt\\]"):
        TensorRTBackend("nonexistent.engine")

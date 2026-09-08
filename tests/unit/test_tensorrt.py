import builtins

import pytest

from fast_omniasr.backends.tensorrt import TensorRTBackend


def test_missing_tensorrt_raises_helpful_import_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("simulated: tensorrt not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[tensorrt\\]"):
        TensorRTBackend("nonexistent.engine")

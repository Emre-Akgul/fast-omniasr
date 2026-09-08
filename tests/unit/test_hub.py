import builtins

import pytest

from fast_omniasr import OmniASR


def test_missing_huggingface_hub_raises_helpful_import_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated: huggingface_hub not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[hub\\]"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX")


def test_non_onnx_backend_rejected_before_any_download():
    with pytest.raises(ValueError, match="backend='onnx' only"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="tensorrt")

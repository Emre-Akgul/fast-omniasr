import builtins
import hashlib
import json
import shutil
import sys
import types

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


def test_tensorrt_backend_requires_explicit_precision_before_any_download():
    with pytest.raises(ValueError, match="precision='fp32' or precision='fp16'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="tensorrt")


def test_unknown_backend_rejected_before_any_download():
    with pytest.raises(ValueError, match="backend must be 'onnx' or 'tensorrt'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="bogus")


def test_onnx_backend_rejects_unknown_precision_before_any_download():
    with pytest.raises(ValueError, match="precision must be one of fp32, fp16, bf16"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx", precision="int8")


def test_onnx_backend_rejects_bf16_on_cpu_before_any_download():
    with pytest.raises(ValueError, match="needs device='cuda'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx", precision="bf16")


def test_onnx_fp16_downloads_only_its_variant(monkeypatch, tmp_path):
    remote = tmp_path / "remote"
    files = ["model.onnx", "model.onnx.data", "tokenizer.model", "fp16/model.onnx",
             "fp16/model.onnx.data", "bf16/model.onnx", "bf16/model.onnx.data"]
    entries = {}
    for name in files:
        (remote / name).parent.mkdir(parents=True, exist_ok=True)
        (remote / name).write_bytes(name.encode())
        entries[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "size": len(name)}
    (remote / "config.json").write_text(json.dumps({"format": 1, "files": entries}))
    fetched = []

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None):
        fetched.append(filename)
        (local_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, local_dir / filename)
        return str(local_dir / filename)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    loaded = {}
    monkeypatch.setattr(OmniASR, "__init__", lambda self, *args, **kwargs: loaded.update(
        model=args[0], **kwargs
    ))
    OmniASR.from_pretrained(
        "EmreAkgul/omniASR-CTC-300M-v2-ONNX", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert sorted(fetched) == ["config.json", "fp16/model.onnx", "fp16/model.onnx.data", "tokenizer.model"]
    assert loaded["model"].parts[-2:] == ("fp16", "model.onnx")
    assert loaded["precision"] == "fp16"


@pytest.mark.parametrize("parameter", ["min_samples", "opt_samples", "max_samples"])
def test_tensorrt_profile_rejected_for_onnx_backend(parameter):
    with pytest.raises(ValueError, match="only used with backend='tensorrt'"):
        OmniASR.from_pretrained(
            "EmreAkgul/omniASR-CTC-300M-v2-ONNX",
            backend="onnx",
            **{parameter: 8000},
        )

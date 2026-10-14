import hashlib
import json
import sys
import types

import pytest

from fast_omniasr.llm import OmniASRLLM

FILES = [
    "encoder.onnx",
    "encoder.onnx.data",
    "decoder_prefill.onnx",
    "decoder_step.onnx",
    "decoder.weights",
    "decoder_weights.json",
    "tokenizer.model",
]


def fake_hub(monkeypatch, tmp_path, corrupt=None):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    entries = {}
    for name in FILES:
        (snapshot / name).write_bytes(name.encode())
        entries[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "size": len(name)}
    if corrupt:
        (snapshot / corrupt).write_bytes(b"tampered")
    config = {"format_version": 1, "model": "omniASR_LLM_1B_v2", "files": entries}
    (snapshot / "config.json").write_text(json.dumps(config))
    fetched = []

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None):
        fetched.append(filename)
        return str(snapshot / filename)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    return snapshot, fetched


def test_from_pretrained_verifies_every_listed_file(monkeypatch, tmp_path):
    snapshot, fetched = fake_hub(monkeypatch, tmp_path)
    loaded = {}

    def init(self, directory, **kwargs):
        loaded.update(directory=directory, **kwargs)

    monkeypatch.setattr(OmniASRLLM, "__init__", init)
    OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-1B-v2-ONNX", device="cuda", threads=8)
    assert sorted(fetched) == sorted(["config.json", *FILES])
    assert loaded == {"directory": snapshot, "backend": "onnx", "device": "cuda", "threads": 8}


def test_from_pretrained_rejects_checksum_mismatch(monkeypatch, tmp_path):
    fake_hub(monkeypatch, tmp_path, corrupt="decoder.weights")
    with pytest.raises(RuntimeError, match="decoder.weights checksum mismatch"):
        OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-1B-v2-ONNX")


def test_from_pretrained_rejects_unknown_device_before_download():
    with pytest.raises(ValueError, match="device must be cpu or cuda"):
        OmniASRLLM.from_pretrained("EmreAkgul/omniASR-LLM-1B-v2-ONNX", device="tpu")

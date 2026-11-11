import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


provenance = _load("export_provenance", ROOT / "tools" / "provenance.py")
export_onnx = _load("export_onnx", ROOT / "export_onnx.py")


def test_path_sha256_hashes_a_file_like_sha256sum(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"weights")
    assert provenance.path_sha256(path) == hashlib.sha256(b"weights").hexdigest()


def test_path_sha256_covers_every_file_and_name_in_a_checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "model"
    (checkpoint / "shard").mkdir(parents=True)
    (checkpoint / "pp_00.pt").write_bytes(b"a")
    (checkpoint / "shard" / "tp_00.pt").write_bytes(b"b")
    first = provenance.path_sha256(checkpoint)

    (checkpoint / "shard" / "tp_00.pt").write_bytes(b"c")
    assert provenance.path_sha256(checkpoint) != first
    (checkpoint / "shard" / "tp_00.pt").write_bytes(b"b")
    assert provenance.path_sha256(checkpoint) == first
    (checkpoint / "shard" / "tp_00.pt").rename(checkpoint / "shard" / "tp_01.pt")
    assert provenance.path_sha256(checkpoint) != first


def test_path_sha256_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        provenance.path_sha256(tmp_path / "missing")


def test_weights_sha256_ignores_order_but_not_values_names_or_dtypes():
    weights = [("a", np.arange(4, dtype=np.float32)), ("b", np.ones((2, 2), np.float32))]
    digest = provenance.weights_sha256(weights)

    assert provenance.weights_sha256(reversed(weights)) == digest
    changed = [("a", np.arange(4, dtype=np.float32) + 1), weights[1]]
    assert provenance.weights_sha256(changed) != digest
    assert provenance.weights_sha256([("c", weights[0][1]), weights[1]]) != digest
    assert provenance.weights_sha256([("a", np.arange(4)), weights[1]]) != digest
    assert provenance.weights_sha256([weights[0], ("b", np.ones(4, np.float32))]) != digest


def _fake_distribution(directory, name, version, direct_url=None):
    info = directory / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


def test_distribution_records_the_vcs_commit_of_a_git_install(tmp_path, monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"
    _fake_distribution(
        tmp_path,
        "fake-upstream-asr",
        "0.2.1.dev0",
        {
            "url": "https://github.com/example/upstream-asr.git",
            "vcs_info": {"vcs": "git", "commit_id": commit},
        },
    )
    _fake_distribution(tmp_path, "fake-released-asr", "0.2.0")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

    assert provenance.distribution("fake-upstream-asr") == {
        "version": "0.2.1.dev0",
        "url": "https://github.com/example/upstream-asr.git",
        "commit": commit,
    }
    assert provenance.distribution("fake-released-asr") == {"version": "0.2.0"}
    assert provenance.distribution("fake-missing-asr") is None


def test_distribution_omits_local_paths(tmp_path, monkeypatch):
    _fake_distribution(
        tmp_path,
        "fake-local-asr",
        "0.3.0",
        {"url": "file:///home/someone/fast-omniasr", "dir_info": {"editable": True}},
    )
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])

    assert provenance.distribution("fake-local-asr") == {"version": "0.3.0", "editable": True}


def test_environment_lists_every_export_package():
    env = provenance.environment()
    assert set(env["packages"]) == set(provenance.EXPORT_PACKAGES)
    assert {"omnilingual-asr", "fairseq2", "torch", "onnx"} <= set(env["packages"])
    assert env["python"].count(".") == 2
    assert env["packages"]["numpy"]["version"] == np.__version__


def test_exporter_records_the_checkout_commit():
    record = provenance.exporter("export_onnx.py")
    assert record["script"] == "export_onnx.py"
    if (ROOT / ".git").exists():
        assert re.fullmatch(r"[0-9a-f]{40}", record["git_commit"])
        assert isinstance(record["git_dirty"], bool)


def test_exporter_outside_a_checkout_records_no_commit(tmp_path):
    record = provenance.exporter("export_onnx.py", root=tmp_path)
    assert record["git_commit"] is None
    assert record["git_dirty"] is None


def test_write_adds_format_and_timestamp(tmp_path):
    path = provenance.write(tmp_path, {"source": {"model_card": "omniASR_CTC_300M_v2"}})
    record = json.loads(path.read_text())
    assert path == tmp_path / "provenance.json"
    assert record["format"] == provenance.PROVENANCE_FORMAT
    assert record["created_at"].endswith("+00:00")
    assert record["source"] == {"model_card": "omniASR_CTC_300M_v2"}
    assert [p.name for p in tmp_path.iterdir()] == ["provenance.json"]


def _export_dir(tmp_path):
    out = tmp_path / "export"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"graph")
    (out / "model.onnx.data").write_bytes(b"external weights")
    (out / "tokenizer.model").write_bytes(b"tokenizer")
    return out


def test_provenance_records_hashes_environment_and_validation(tmp_path):
    out = _export_dir(tmp_path)
    args = SimpleNamespace(
        model="omniASR_CTC_300M_v2", checkpoint=None, output=out / "model.onnx"
    )
    validation = {"rtol": 1e-4, "atol": 1e-3, "probes": [{"input": "noise_400"}]}

    record = export_onnx.build_provenance(
        args,
        weights_sha256="ab" * 32,
        model_path=out / "model.onnx",
        files=[out / "model.onnx", out / "model.onnx.data"],
        validation=validation,
    )

    assert record["exporter"]["script"] == "export_onnx.py"
    assert "omnilingual-asr" in record["environment"]["packages"]
    assert record["source"] == {
        "model_card": "omniASR_CTC_300M_v2",
        "checkpoint": None,
        "weights_sha256": "ab" * 32,
    }
    assert record["tokenizer"] == {
        "file": "tokenizer.model",
        "family": "v2",
        "url": export_onnx.TOKENIZER_URLS["v2"],
        "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
    }
    assert record["files"] == {
        "model.onnx": {"sha256": hashlib.sha256(b"graph").hexdigest(), "size": 5},
        "model.onnx.data": {
            "sha256": hashlib.sha256(b"external weights").hexdigest(),
            "size": 16,
        },
    }
    assert record["validation"] == validation
    json.dumps(record)


def test_provenance_hashes_a_finetuned_checkpoint(tmp_path):
    out = _export_dir(tmp_path)
    checkpoint = tmp_path / "step_100.pt"
    checkpoint.write_bytes(b"fine-tuned")
    args = SimpleNamespace(
        model="omniASR_CTC_1B", checkpoint=checkpoint, output=out / "model.onnx"
    )

    record = export_onnx.build_provenance(
        args,
        weights_sha256="cd" * 32,
        model_path=out / "model.onnx",
        files=[out / "model.onnx"],
        validation={},
    )

    assert record["source"]["checkpoint"] == {
        "name": "step_100.pt",
        "sha256": hashlib.sha256(b"fine-tuned").hexdigest(),
    }
    assert record["tokenizer"]["family"] == "v1"


def _build(model_path, tokenizer_path):
    model_path.write_bytes(b"new graph")
    tokenizer_path.write_bytes(b"new tokenizer")


def test_staged_export_promotes_provenance_describing_the_staged_files(tmp_path):
    output = tmp_path / "export" / "model.onnx"
    seen = []

    def describe(model_path, report):
        seen.append((model_path.parent != output.parent, report))
        return {"files": {"model.onnx": provenance.file_sha256(model_path)}}

    export_onnx.staged_export(output, _build, lambda _: {"probes": []}, describe)

    assert seen == [(True, {"probes": []})]
    record = json.loads((output.parent / "provenance.json").read_text())
    assert record["files"] == {"model.onnx": hashlib.sha256(b"new graph").hexdigest()}
    assert sorted(p.name for p in output.parent.iterdir()) == sorted(
        ["model.onnx", "tokenizer.model", export_onnx.PARITY_REPORT, "provenance.json"]
    )


def test_failed_validation_writes_no_provenance(tmp_path):
    output = tmp_path / "model.onnx"
    described = []

    def validate(_):
        raise ValueError("noise_400: logits differ")

    with pytest.raises(ValueError):
        export_onnx.staged_export(output, _build, validate, lambda *a: described.append(a))

    assert described == []
    assert list(tmp_path.iterdir()) == []


def test_export_constraints_pin_every_export_package():
    pins = {}
    for line in (ROOT / "constraints" / "export.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, version = line.partition("==")
            assert version, f"{line!r} is not an exact pin"
            pins[name.lower().replace("_", "-")] = version
    for name in set(provenance.EXPORT_PACKAGES) - {"fast-omniasr"}:
        assert name in pins, f"{name} is not pinned in constraints/export.txt"
    assert pins["torch"] == "2.8.0"
    assert pins["fairseq2"] == pins["fairseq2n"] == "0.6"
    assert pins["omnilingual-asr"] == "0.2.0"
    assert pins["onnx"] == "1.17.0"

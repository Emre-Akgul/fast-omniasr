import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "tools" / "publish_hub.py"
SPEC = importlib.util.spec_from_file_location("publish_hub", SCRIPT)
publish_hub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish_hub)


def test_repo_names_cover_original_and_v2():
    assert publish_hub.repo_name("omniASR_CTC_1B") == "omniASR-CTC-1B-ONNX"
    assert publish_hub.repo_name("omniASR_CTC_7B_v2") == "omniASR-CTC-7B-v2-ONNX"


def test_prepare_bundle_includes_external_data_and_checksums(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    external = tmp_path / "model.onnx.data"
    tokenizer = tmp_path / "source-tokenizer.model"
    model.write_bytes(b"onnx")
    external.write_bytes(b"weights")
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(publish_hub, "external_data_files", lambda _: [external])

    files = publish_hub.prepare_bundle("omniASR_CTC_1B", model, tokenizer)

    assert {path.name for path in files} == {
        "config.json", "model.onnx", "model.onnx.data", "tokenizer.model"
    }
    config = json.loads((tmp_path / "config.json").read_text())
    assert set(config["files"]) == {"model.onnx", "model.onnx.data", "tokenizer.model"}
    assert config["files"]["model.onnx.data"]["size"] == 7

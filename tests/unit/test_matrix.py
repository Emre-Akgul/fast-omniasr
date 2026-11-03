"""tests/matrix.json stays consistent with the supported models, backends and docs (#31)."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix_tool = _load("test_matrix_tool", ROOT / "tools" / "test_matrix.py")
MATRIX = matrix_tool.load()


def test_every_supported_model_card_is_in_the_matrix():
    from fast_omniasr.llm.models import LLM_MODEL_CARDS

    export = _load("export_onnx", ROOT / "export_onnx.py")
    assert tuple(MATRIX["families"]["ctc"]["cards"]) == tuple(export.CTC_MODEL_CARDS)
    assert tuple(MATRIX["families"]["llm"]["cards"]) == tuple(LLM_MODEL_CARDS)
    for family in MATRIX["families"].values():
        assert family["pinned"] in family["cards"]


def test_pinned_models_match_the_scheduled_validation():
    thresholds = json.loads((ROOT / "benchmarks/fleurs/thresholds.json").read_text(encoding="utf-8"))
    for family, model in thresholds["models"].items():
        card = MATRIX["families"][family]["pinned"]
        repo = model["repo"].split("/")[1]
        expected = card.replace("omniASR_", "omniASR-").replace("_", "-") + "-ONNX"
        assert repo == expected, (family, repo, card)


VALID = {
    ("ctc", "onnx"): {"cpu": {"fp32", "fp16"}, "cuda": {"fp32", "fp16", "bf16"}},
    ("ctc", "tensorrt"): {"cuda": {"fp32", "fp16", "mixed"}},
    ("llm", "onnx"): {"cpu": {"fp32", "fp16"}, "cuda": {"fp32", "fp16", "bf16"}},
    ("llm", "torch"): {"cpu": {"fp32"}},
}


def test_configurations_are_exactly_the_supported_ones():
    listed = {(c["family"], c["backend"], c["device"], c["precision"])
              for c in MATRIX["configurations"]}
    supported = {(family, backend, device, precision)
                 for (family, backend), devices in VALID.items()
                 for device, precisions in devices.items() for precision in precisions}
    assert listed == supported
    assert len(listed) == len(MATRIX["configurations"])  # no duplicates


def test_tiers_are_defined_and_unit_covers_everything():
    tiers = set(MATRIX["tiers"])
    for config in MATRIX["configurations"]:
        for key in ("pinned", "other"):
            assert set(config[key]) <= tiers, config
            assert "unit" in config[key], config
        assert set(config["other"]) <= set(config["pinned"]), config  # pinned is a superset
    for tier in MATRIX["tiers"].values():
        assert tier["when"] and tier["runs"] and tier["command"]


@pytest.mark.parametrize("path", ["tests/integration/test_real_models_cpu.py",
                                  "tests/integration/test_tensorrt_fleurs.py",
                                  "tools/fleurs_eval.py", "tools/evaluate_precisions.py",
                                  "tools/publish_precisions.py", "tools/validate_llm_fairseq2.py"])
def test_referenced_tests_and_tools_exist(path):
    assert path in json.dumps(MATRIX)
    assert (ROOT / path).is_file()


def test_docs_are_generated_from_the_matrix():
    assert matrix_tool.main(["--check"]) == 0

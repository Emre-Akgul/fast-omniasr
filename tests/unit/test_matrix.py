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
    fixtures = json.loads((ROOT / "tests/fixtures/fleurs/manifest.json").read_text(encoding="utf-8"))
    for family, model in thresholds["models"].items():
        card = MATRIX["families"][family]["pinned"]
        expected = card.replace("omniASR_", "omniASR-").replace("_", "-") + "-ONNX"
        assert model["repo"].split("/")[1] == expected, (family, model["repo"], card)
        pinned = fixtures["models"][family]
        assert pinned["repo"] == model["repo"], (family, pinned["repo"], model["repo"])


def _supported():
    """Every (family, backend, device, precision) the code accepts, derived from the code."""
    from fast_omniasr.precision import PRECISIONS, check_precision
    from fast_omniasr.tensorrt_builder import PRECISIONS as ENGINE_PRECISIONS

    supported = set()
    for family in ("ctc", "llm"):
        for device in ("cpu", "cuda"):
            for precision in PRECISIONS:
                try:
                    check_precision(precision, device)
                except ValueError:
                    continue
                supported.add((family, "onnx", device, precision))
    supported |= {("ctc", "tensorrt", "cuda", p) for p in ENGINE_PRECISIONS}
    supported.add(("llm", "torch", "cpu", "fp32"))  # the TorchScript reference: CPU fp32 only
    return supported


def test_configurations_are_exactly_the_supported_ones():
    listed = {(c["family"], c["backend"], c["device"], c["precision"])
              for c in MATRIX["configurations"]}
    assert listed == _supported()
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


def test_every_test_and_tool_the_matrix_names_exists():
    import re

    text = json.dumps(MATRIX)
    paths = set(re.findall(r"(?:tools|tests/integration|tests/unit)/[\w/]+\.py", text))
    assert {"tools/fleurs_eval.py", "tests/integration/test_real_models_cpu.py"} <= paths
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    assert not missing, missing


def test_torch_backend_is_still_cpu_fp32_only():
    from fast_omniasr.llm import OmniASRLLM

    with pytest.raises(ValueError, match="precision='fp32' only"):
        OmniASRLLM(ROOT, backend="torch", precision="fp16")


def test_splice_rejects_missing_or_misordered_markers():
    begin, end = matrix_tool.BEGIN, matrix_tool.END
    assert matrix_tool.splice(f"a{begin}old{end}b", "T") == f"a{begin}\nT{end}b"
    for document in ("no markers", f"{end}x{begin}", f"{begin}{begin}{end}"):
        with pytest.raises(ValueError, match="exactly one"):
            matrix_tool.splice(document, "T")


def test_docs_are_generated_from_the_matrix():
    assert matrix_tool.main(["--check"]) == 0

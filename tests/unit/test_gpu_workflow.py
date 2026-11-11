"""The GPU release validation workflow stays runnable and covers every CUDA configuration (#28)."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "gpu.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")


def _top_level_block(name):
    """The lines of a top-level YAML key's block (the workflow is plain block YAML)."""
    lines = TEXT.splitlines()
    start = lines.index(f"{name}:")
    block = []
    for line in lines[start + 1:]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        block.append(line)
    return block


def test_runs_only_on_release_and_demand():
    """With no hosted GPU runner, a push or PR trigger would queue forever on every change."""
    triggers = {line.strip().rstrip(":") for line in _top_level_block("on")
                if re.match(r"^  \w", line)}
    assert triggers == {"workflow_dispatch", "release"}
    assert "types: [published]" in TEXT


def test_targets_a_self_hosted_gpu_runner():
    assert re.search(r"runs-on: \[self-hosted, linux, x64, gpu\]", TEXT)


def test_missing_runtimes_fail_instead_of_skipping():
    assert re.search(r'OMNIASR_REQUIRE_GPU: "1"', TEXT)
    assert re.search(r'OMNIASR_REAL_MODELS: "1"', TEXT)


def test_every_test_it_runs_exists_and_is_a_gpu_test():
    paths = set(re.findall(r"tests/integration/\w+\.py", TEXT))
    assert {"tests/integration/test_real_models_gpu.py",
            "tests/integration/test_tensorrt_fleurs.py"} <= paths
    for path in paths:
        assert (ROOT / path).is_file(), path
        assert "pytest.mark.cuda" in (ROOT / path).read_text(encoding="utf-8"), path


def test_gpu_test_covers_every_cuda_ctc_configuration_in_the_matrix():
    spec = importlib.util.spec_from_file_location(
        "test_real_models_gpu", ROOT / "tests" / "integration" / "test_real_models_gpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = json.loads((ROOT / "tests" / "matrix.json").read_text(encoding="utf-8"))
    cuda = {(c["backend"], c["precision"]) for c in matrix["configurations"]
            if c["family"] == "ctc" and c["device"] == "cuda"}
    assert set(module.CONFIGURATIONS) == cuda
    assert len(module.CONFIGURATIONS) == len(cuda)

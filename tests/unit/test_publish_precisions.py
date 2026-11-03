import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load():
    spec = importlib.util.spec_from_file_location("publish_precisions", ROOT / "tools/publish_precisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_section_is_inserted_once_and_replaced_on_republish():
    publish = load()
    card = "# Model\n\nIntro.\n\n## Repository files\n\n| a | b |\n\n## License\n\nApache.\n"
    first = publish.update_model_card(card, "## Precision\n\nversion one\n")
    assert first.index("## Precision") < first.index("## Repository files")
    second = publish.update_model_card(first, "## Precision\n\nversion two\n")
    assert second.count("## Precision") == 1
    assert "version two" in second and "version one" not in second
    assert second.endswith("## License\n\nApache.\n")


def test_memory_column_reads_current_and_older_reports():
    publish = load()
    assert publish.memory_gib({"peak_gpu_mib": 2048}) == "2.0 GiB"
    assert publish.memory_gib({"gpu_mib": 1536}) == "1.5 GiB"  # report from before #21
    assert publish.memory_gib({"peak_gpu_mib": None}) == "n/a"


def test_precision_section_accepts_a_current_report():
    publish = load()
    report = {"precisions": {p: {"inputs": 3, "identical_to_fp32": 3, "peak_gpu_mib": mib}
                             for p, mib in [("fp32", 2048), ("fp16", 1024), ("bf16", None)]}}
    section = publish.precision_section(False, "org/model", report)
    assert "2.0 GiB" in section and "1.0 GiB" in section and "n/a" in section

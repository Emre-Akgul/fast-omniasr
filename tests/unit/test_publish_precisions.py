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

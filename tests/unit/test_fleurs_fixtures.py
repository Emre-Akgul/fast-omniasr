"""The committed FLEURS fixtures are intact, attributed, and usable by both runtimes."""

import hashlib
import json
from pathlib import Path

import pytest
import soundfile as sf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fleurs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def test_manifest_pins_dataset_and_models():
    assert MANIFEST["dataset"] == "google/fleurs"
    assert MANIFEST["license"] == "CC-BY-4.0"
    assert len(MANIFEST["revision"]) == 40
    for model in MANIFEST["models"].values():
        assert model["repo"].startswith("EmreAkgul/") and len(model["revision"]) == 40


def test_attribution_is_present():
    readme = (FIXTURES / "README.md").read_text()
    assert "CC BY 4.0" in readme and MANIFEST["revision"] in readme


@pytest.mark.parametrize("clip", MANIFEST["clips"], ids=lambda c: c["fleurs_language"])
def test_clip_matches_manifest(clip):
    path = FIXTURES / clip["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"]
    info = sf.info(path)
    assert info.samplerate == 16000 and info.channels == 1
    assert abs(info.frames / info.samplerate - clip["seconds"]) < 0.01
    assert info.frames / info.samplerate <= 30  # within the LLM runtime's clip limit
    assert clip["reference"].strip()
    assert set(clip["max_cer"]) == {"ctc", "llm"}

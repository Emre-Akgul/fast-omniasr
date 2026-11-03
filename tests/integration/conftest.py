"""Explicit fixtures for the integration tests.

The audio clip, the model it is checked against and the exact output expected are all
declared rather than assumed:

- OMNIASR_TEST_WAV picks the clip; by default the committed English FLEURS clip.
- Expected transcripts come from tests/fixtures/integration_expectations.json, matched by
  the clip's sha256 and a `<repo id>@<commit>` model key (OMNIASR_TEST_MODEL for local
  assets, which have no Hub identity). OMNIASR_TEST_TRANSCRIPT overrides the manifest for
  any other clip. Without either, exact-text checks are skipped, not guessed.
- Vocabulary sizes come from the tokenizer, and TensorRT length checks from the engine's
  own profile, so custom exports and custom profiles are tested on their own terms.
"""

import hashlib
import json
import os
import types
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EXPECTATIONS = FIXTURES / "integration_expectations.json"
DEFAULT_CLIP = "fleurs/en_us_10233995782544396174.flac"
PUBLISHED_CTC = "EmreAkgul/omniASR-CTC-300M-v2-ONNX"


def clip_path() -> Path:
    return Path(os.environ.get("OMNIASR_TEST_WAV") or FIXTURES / DEFAULT_CLIP)


def expected_transcript(clip: Path, model: str | None, variant: str) -> str | None:
    """The exact transcript expected for `clip` from `model` (`<repo id>@<commit>`) run as
    `variant` (e.g. 'onnx-cpu-fp32'), or None when nothing is declared."""
    override = os.environ.get("OMNIASR_TEST_TRANSCRIPT")
    if override is not None:
        return override
    if model is None:
        return None
    digest = hashlib.sha256(Path(clip).read_bytes()).hexdigest()
    manifest = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    for entry in manifest["clips"].values():
        if entry["sha256"] == digest:
            return entry["expected"].get(model, {}).get("transcripts", {}).get(variant)
    return None


def vocabulary_size(tokenizer_path) -> int:
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(tokenizer_path)).get_piece_size()


def frames(samples: int) -> int:
    """Encoder frames for a waveform: the wav2vec2 front end's 400-sample window, 320 hop."""
    return (samples - 400) // 320 + 1


@pytest.fixture
def clip():
    path = clip_path()
    if not path.is_file():
        pytest.skip(f"test clip {path} not found")
    return path


@pytest.fixture
def local_model_key():
    """Model identity for local assets (OMNIASR_TEST_MODEL), which can't be derived."""
    return os.environ.get("OMNIASR_TEST_MODEL")


@pytest.fixture
def expect():
    """Helpers: expect.transcript(clip, model, variant), expect.vocabulary(tokenizer),
    expect.frames(samples), expect.published(snapshot) -> model key."""
    return types.SimpleNamespace(
        transcript=expected_transcript, vocabulary=vocabulary_size, frames=frames,
        published=lambda snapshot: f"{snapshot.repo_id}@{snapshot.commit}",
    )

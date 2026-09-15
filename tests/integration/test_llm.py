"""Opt-in real-checkpoint parity in a fresh, reference-free process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_standalone_llm_parity():
    directory = os.environ.get("FAST_OMNIASR_LLM_DIR")
    if not directory:
        pytest.skip("Set FAST_OMNIASR_LLM_DIR to validated LLM export assets")
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / "tools" / "validate_llm.py"), directory]
    audio = os.environ.get("FAST_OMNIASR_LLM_TEST_WAV")
    if audio:
        command.extend(["--audio", audio])
    subprocess.run(
        command,
        check=True,
        timeout=1800,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_standalone_cached_llm_parity():
    directory = os.environ.get("FAST_OMNIASR_LLM_DIR")
    if not directory or not (Path(directory) / "decoder_cached.pt").exists():
        pytest.skip("Set FAST_OMNIASR_LLM_DIR to an export with decoder_cached.pt")
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, str(root / "tools" / "validate_llm.py"), directory, "--cached"],
        check=True,
        timeout=1800,
    )

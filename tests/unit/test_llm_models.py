import json

import pytest

from fast_omniasr.llm import OmniASRLLM
from fast_omniasr.llm.models import LLM_MODEL_CARDS, repo_name


def test_repo_names_cover_all_supported_cards():
    assert [repo_name(card) for card in LLM_MODEL_CARDS] == [
        "omniASR-LLM-300M-ONNX",
        "omniASR-LLM-1B-ONNX",
        "omniASR-LLM-3B-ONNX",
        "omniASR-LLM-7B-ONNX",
        "omniASR-LLM-300M-v2-ONNX",
        "omniASR-LLM-1B-v2-ONNX",
        "omniASR-LLM-3B-v2-ONNX",
        "omniASR-LLM-7B-v2-ONNX",
    ]
    for unsupported in ["omniASR_LLM_7B_ZS", "omniASR_LLM_Unlimited_300M_v2"]:
        with pytest.raises(ValueError, match="Unsupported"):
            repo_name(unsupported)


@pytest.mark.parametrize("model", ["omniASR_LLM_7B_ZS", "omniASR_LLM_Unlimited_1B_v2"])
def test_runtime_rejects_unsupported_cards(tmp_path, model):
    (tmp_path / "config.json").write_text(json.dumps({"format_version": 1, "model": model}))
    with pytest.raises(ValueError, match="Unsupported"):
        OmniASRLLM(tmp_path, backend="onnx")

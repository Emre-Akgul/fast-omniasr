"""Supported LLM-ASR model cards and their upstream tokenizers.

All eight share the 12-layer, 4096-dimensional LLaMA decoder with eight K/V
heads; they differ in the wav2vec2 encoder and the output vocabulary. The
zero-shot (7B_ZS) and streaming Unlimited cards use other prompt syntaxes and
are not supported.
"""

TOKENIZER_URLS = {
    "omniASR_tokenizer_v1": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
    "omniASR_tokenizer_v1_variant7": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model",
    "omniASR_tokenizer_written_v2": (
        "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model"
    ),
}

LLM_MODEL_TOKENIZERS = {
    "omniASR_LLM_300M": "omniASR_tokenizer_v1",
    "omniASR_LLM_1B": "omniASR_tokenizer_v1",
    "omniASR_LLM_3B": "omniASR_tokenizer_v1",
    "omniASR_LLM_7B": "omniASR_tokenizer_v1_variant7",
    "omniASR_LLM_300M_v2": "omniASR_tokenizer_written_v2",
    "omniASR_LLM_1B_v2": "omniASR_tokenizer_written_v2",
    "omniASR_LLM_3B_v2": "omniASR_tokenizer_written_v2",
    "omniASR_LLM_7B_v2": "omniASR_tokenizer_written_v2",
}

LLM_MODEL_CARDS = tuple(LLM_MODEL_TOKENIZERS)


def repo_name(model_card: str) -> str:
    """omniASR_LLM_1B_v2 -> omniASR-LLM-1B-v2-ONNX."""
    if model_card not in LLM_MODEL_TOKENIZERS:
        raise ValueError(f"Unsupported LLM model card: {model_card}")
    suffix = model_card.removeprefix("omniASR_LLM_").replace("_v2", "-v2")
    return f"omniASR-LLM-{suffix}-ONNX"

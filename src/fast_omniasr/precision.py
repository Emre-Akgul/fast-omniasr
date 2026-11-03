"""Inference precisions for ONNX Runtime graphs.

fp16/bf16 variants store MatMul/Gemm weights in half precision; convolutions, norms,
softmax, residual paths and all graph inputs/outputs stay float32. Each variant lives in a
subdirectory named after its precision ("fp16/", "bf16/") next to the float32 graphs and
shares the tokenizer and other non-graph assets at the top level.
"""

PRECISIONS = ("fp32", "fp16", "bf16")


def check_precision(precision: str, device: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}")
    if precision == "bf16" and device == "cpu":
        raise ValueError(
            "precision='bf16' needs device='cuda': ONNX Runtime's CPU provider has no bf16 "
            "MatMul kernel. Use 'fp16' or 'fp32' on CPU."
        )
    return precision


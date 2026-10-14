"""Small tensor parity tests for the production ONNX lowering; no fairseq2."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_onnx_prefill_step_dynamic_lengths_and_weights(tmp_path):
    from fast_omniasr.llm.onnx_backend import decoder_sessions

    fixtures = load("cache_fixtures", ROOT / "tests/unit/test_llm_cache.py")
    lowering = load("onnx_lowering", ROOT / "tools/llm_onnx_graphs.py")
    torch.manual_seed(17)
    model = fixtures.synthetic_decoder(4)
    for layer in model.layers:
        layer.ffn.gate_proj = layer.ffn.gate
        layer.ffn.inner_proj = layer.ffn.up
        layer.ffn.output_proj = layer.ffn.down
    bos = torch.tensor([[0]])
    with torch.inference_mode():
        context = torch.randn(1, 3, 32)
        _, cache = model.prefill(context, bos)
        traced = torch.jit.trace_module(
            model,
            {"prefill": (context, bos), "decode_step": (bos, cache)},
            strict=False,
            check_trace=False,
        )
        traced.save(str(tmp_path / "oracle.pt"))
        traced = torch.jit.load(str(tmp_path / "oracle.pt"))
        lowering.export_decoder(traced, tmp_path)
        prefill, step, keepalive = decoder_sessions(tmp_path)
        assert len(keepalive) > 0
        for length in [1, 9, 49, 149]:
            context = torch.randn(1, length, 32)
            ref_logits, ref_cache = traced.prefill(context, bos)
            actual = prefill.run(None, {"context": context.numpy(), "bos": bos.numpy()})
            for offset in range(3):
                for a, b in zip(actual, [ref_logits, *ref_cache], strict=True):
                    np.testing.assert_allclose(a, b.numpy(), atol=3e-6, rtol=2e-5)
                assert all(x.shape == (1, length + offset + 1, 4, 8) for x in actual[1:])
                token = ref_logits.argmax(-1).reshape(1, 1)
                ref_logits, ref_cache = traced.decode_step(token, ref_cache)
                names = [value.name for value in step.get_inputs()]
                actual = step.run(None, dict(zip(names, [token.numpy(), *actual[1:]], strict=True)))


def test_weight_norm_folding_matches_float64_and_removes_runtime_norm(tmp_path):
    import onnx
    import onnxruntime as ort

    exporter = load("export_llm_onnx", ROOT / "export_llm_onnx.py")
    torch.manual_seed(3)
    conv = torch.nn.utils.weight_norm(
        torch.nn.Conv1d(32, 32, kernel_size=16, padding=8, groups=4), dim=2
    )
    conv.weight_v.data *= 50.0
    path = tmp_path / "conv.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            conv, (torch.randn(1, 32, 40),), str(path), opset_version=18, dynamo=False,
            input_names=["x"], output_names=["y"], dynamic_axes={"x": {2: "t"}},
        )
    model = onnx.load(str(path))
    assert "ReduceL2" in {node.op_type for node in model.graph.node}
    assert exporter.fold_weight_norm(model) == 1
    assert "ReduceL2" not in {node.op_type for node in model.graph.node}
    assert not {"weight_g", "weight_v"} & {t.name.split(".")[-1] for t in model.graph.initializer}
    x = np.random.default_rng(1).normal(size=(1, 32, 77)).astype(np.float32)
    actual = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]
    g, v = conv.weight_g.detach().double(), conv.weight_v.detach().double()
    weight = g * v / torch.linalg.vector_norm(v, dim=(0, 1), keepdim=True)
    expected = torch.nn.functional.conv1d(
        torch.from_numpy(x).double(), weight, conv.bias.detach().double(), padding=8, groups=4
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

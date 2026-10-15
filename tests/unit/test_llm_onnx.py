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


def traced_decoder(tmp_path, seed=17):
    """A small traced cached decoder (4 K/V heads, head dim 8, hidden 32) as the oracle."""
    fixtures = load("cache_fixtures", ROOT / "tests/unit/test_llm_cache.py")
    torch.manual_seed(seed)
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
    traced.save(str(tmp_path / f"oracle_{seed}.pt"))
    return torch.jit.load(str(tmp_path / f"oracle_{seed}.pt"))


def test_onnx_prefill_step_dynamic_lengths_and_weights(tmp_path):
    from fast_omniasr.llm.onnx_backend import decoder_sessions

    lowering = load("onnx_lowering", ROOT / "tools/llm_onnx_graphs.py")
    traced = traced_decoder(tmp_path)
    bos = torch.tensor([[0]])
    with torch.inference_mode():
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


def test_unified_decoder_matches_reference_with_empty_context_and_cache(tmp_path):
    from fast_omniasr.llm.onnx_backend import decoder_session

    lowering = load("onnx_lowering", ROOT / "tools/llm_onnx_graphs.py")
    traced = traced_decoder(tmp_path)
    bos = torch.tensor([[0]])
    with torch.inference_mode():
        lowering.export_decoder(traced, tmp_path)
        decoder, _keepalive = decoder_session(tmp_path, "decoder.onnx")
        names = [value.name for value in decoder.get_inputs()]
        assert names[:3] == ["context", "tokens", "past_key_0"]
        empty_cache = [np.zeros((1, 0, 4, 8), np.float32)] * (len(names) - 2)
        for length in [1, 9, 49, 149]:
            context = torch.randn(1, length, 32)
            ref_logits, ref_cache = traced.prefill(context, bos)
            feeds = [context.numpy(), bos.numpy(), *empty_cache]
            actual = decoder.run(None, dict(zip(names, feeds, strict=True)))
            for offset in range(3):
                for a, b in zip(actual, [ref_logits, *ref_cache], strict=True):
                    np.testing.assert_allclose(a, b.numpy(), atol=3e-6, rtol=2e-5)
                assert all(x.shape == (1, length + offset + 1, 4, 8) for x in actual[1:])
                token = ref_logits.argmax(-1).reshape(1, 1)
                ref_logits, ref_cache = traced.decode_step(token, ref_cache)
                feeds = [np.zeros((1, 0, 32), np.float32), token.numpy(), *actual[1:]]
                actual = decoder.run(None, dict(zip(names, feeds, strict=True)))


def test_unified_decoder_reuses_existing_weights_and_rejects_mismatch(tmp_path):
    import onnx
    import onnxruntime as ort

    lowering = load("onnx_lowering", ROOT / "tools/llm_onnx_graphs.py")
    traced = traced_decoder(tmp_path)
    with torch.inference_mode():
        lowering.export_decoder(traced, tmp_path)
        written = (tmp_path / "decoder.weights").read_bytes()
        (tmp_path / "decoder.onnx").rename(tmp_path / "fresh.onnx")
        lowering.export_unified_decoder(traced, tmp_path)
        assert (tmp_path / "decoder.weights").read_bytes() == written
        fresh = onnx.load(str(tmp_path / "fresh.onnx"), load_external_data=False)
        added = onnx.load(str(tmp_path / "decoder.onnx"), load_external_data=False)
        assert fresh.graph == added.graph
        assert ort.InferenceSession(str(tmp_path / "decoder.onnx")).get_inputs()[1].name == "tokens"
        (tmp_path / "decoder.onnx").unlink()
        with pytest.raises(ValueError, match="differs from the source"):
            lowering.export_unified_decoder(traced_decoder(tmp_path, seed=18), tmp_path)


def test_weight_norm_folding_matches_float64_and_removes_runtime_norm(tmp_path):
    import onnx
    import onnxruntime as ort

    exporter = load("onnx_fold", ROOT / "tools/onnx_fold.py")
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


def fake_backend(monkeypatch, tmp_path):
    """ONNXLLMBackend with recorded session loads and NumPy stand-in graphs."""
    from types import SimpleNamespace

    from fast_omniasr.llm import onnx_backend

    for name in [
        "encoder.onnx",
        "decoder_prefill.onnx",
        "decoder_step.onnx",
        "decoder.weights",
        "decoder_weights.json",
    ]:
        (tmp_path / name).write_bytes(b"")
    loads = []

    class Graph:
        def __init__(self, inputs, outputs):
            self.inputs, self.outputs = inputs, outputs

        def get_inputs(self):
            return [SimpleNamespace(name=name) for name in self.inputs]

        def run(self, names, feeds, run_options=None):
            return self.outputs()

    def step_outputs():
        return [np.zeros((1, 10), np.float32), *[np.zeros((1, 2, 1, 1), np.float32)] * 24]

    def unified_graph():
        inputs = [
            SimpleNamespace(name="context", shape=[1, "context_len", 8]),
            SimpleNamespace(name="tokens", shape=[1, "new_tokens"]),
            *[
                SimpleNamespace(name=name, shape=[1, "cached_positions", 1, 1])
                for name in onnx_backend.STEP_INPUTS[1:]
            ],
        ]
        graph = Graph([], step_outputs)
        graph.get_inputs = lambda: inputs
        return graph

    def session(path, device="cpu", opts=None, arena=None):
        path = Path(path)
        loads.append(path.name if path.parent == tmp_path else f"{path.parent.name}/{path.name}")
        if path.name == "decoder.onnx":  # half-precision decoders load directly
            return unified_graph()
        return Graph(["audio"], lambda: [np.zeros((1, 4, 8), np.float32)])

    def decoder_sessions(directory, device="cpu", threads=4):
        loads.append("decoder")
        prefill = Graph(["context", "bos"], step_outputs)
        return prefill, Graph(onnx_backend.STEP_INPUTS, step_outputs), []

    def decoder_session(directory, graph, device="cpu", threads=4, arena=None):
        loads.append(graph)
        return unified_graph(), []

    monkeypatch.setattr(onnx_backend, "session", session)
    monkeypatch.setattr(onnx_backend, "decoder_sessions", decoder_sessions)
    monkeypatch.setattr(onnx_backend, "decoder_session", decoder_session)
    return onnx_backend.ONNXLLMBackend, loads


def run_clips(backend, clips=3):
    for _ in range(clips):
        context = backend.encode(np.zeros((1, 1600), np.float32))
        _, cache = backend.prefill(context, np.zeros((1, 1), np.int64))
        backend.decode_step(np.zeros((1, 1), np.int64), cache)


def test_resident_backend_loads_each_session_once(monkeypatch, tmp_path):
    backend_class, loads = fake_backend(monkeypatch, tmp_path)
    run_clips(backend_class(tmp_path, "cpu"))
    assert loads == ["encoder.onnx", "decoder"]


def test_low_memory_backend_phases_sessions_per_clip(monkeypatch, tmp_path):
    backend_class, loads = fake_backend(monkeypatch, tmp_path)
    run_clips(backend_class(tmp_path, "cpu", low_memory=True))
    assert loads == ["encoder.onnx", "decoder"] * 3


def test_unified_backend_uses_one_decoder_session(monkeypatch, tmp_path):
    backend_class, loads = fake_backend(monkeypatch, tmp_path)
    (tmp_path / "decoder.onnx").write_bytes(b"")
    backend = backend_class(tmp_path, "cpu")
    assert backend.unified
    run_clips(backend)
    assert loads == ["encoder.onnx", "decoder.onnx"]


def test_half_precision_backend_loads_its_subdirectory(monkeypatch, tmp_path):
    backend_class, loads = fake_backend(monkeypatch, tmp_path)
    (tmp_path / "fp16").mkdir()
    for name in ["encoder.onnx", "decoder.onnx"]:
        (tmp_path / "fp16" / name).write_bytes(b"")
    run_clips(backend_class(tmp_path, "cpu", precision="fp16"))
    assert loads == ["fp16/encoder.onnx", "fp16/decoder.onnx"]
    with pytest.raises(ValueError, match="needs device='cuda'"):
        backend_class(tmp_path, "cpu", precision="bf16")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        backend_class(tmp_path / "empty", "cpu", precision="fp16")


def small_matmul_conv_model(path):
    """x [1, 8, 20] -> Conv(8->8, k3) -> Transpose -> MatMul(8x6) -> y [1, 20, 6]."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(5)
    nodes = [
        helper.make_node("Conv", ["x", "conv_w"], ["c"], pads=[1, 1]),
        helper.make_node("Transpose", ["c"], ["t"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["t", "proj_w"], ["y"]),
    ]
    initializers = [
        numpy_helper.from_array(rng.normal(size=(8, 8, 3)).astype(np.float32), "conv_w"),
        numpy_helper.from_array(rng.normal(size=(8, 6)).astype(np.float32), "proj_w"),
    ]
    graph = helper.make_graph(
        nodes, "small",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, "t"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "t", 6])],
        initializers,
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9), path)


def test_convert_precision_halves_matmul_and_keeps_conv_float32(tmp_path):
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto

    converter = load("convert_precision", ROOT / "tools/convert_precision.py")
    small_matmul_conv_model(tmp_path / "model.onnx")
    x = np.random.default_rng(6).normal(size=(1, 8, 20)).astype(np.float32)
    expected = ort.InferenceSession(str(tmp_path / "model.onnx")).run(None, {"x": x})[0]
    for precision, dtype in [("fp16", TensorProto.FLOAT16), ("bf16", TensorProto.BFLOAT16)]:
        destination = tmp_path / precision / "model.onnx"
        assert converter.convert_graph(tmp_path / "model.onnx", destination, precision) == 1
        model = onnx.load(str(destination))
        types = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
        assert types == {"conv_w": TensorProto.FLOAT, f"proj_w/{precision}": dtype}
        assert [value.type.tensor_type.elem_type for value in model.graph.output] == [TensorProto.FLOAT]
    # ONNX Runtime's CPU provider runs fp16 MatMul (bf16 has no CPU kernel).
    actual = ort.InferenceSession(str(tmp_path / "fp16" / "model.onnx")).run(None, {"x": x})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)


def test_fold_file_rewrites_per_tensor_external_model_in_place(tmp_path):
    import onnx
    import onnxruntime as ort

    folding = load("onnx_fold", ROOT / "tools/onnx_fold.py")
    torch.manual_seed(4)
    conv = torch.nn.utils.weight_norm(
        torch.nn.Conv1d(32, 32, kernel_size=16, padding=8, groups=4), dim=2
    )
    staged = tmp_path / "staged.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            conv, (torch.randn(1, 32, 40),), str(staged), opset_version=18, dynamo=False,
            input_names=["x"], output_names=["y"], dynamic_axes={"x": {2: "t"}},
        )
    export = tmp_path / "export"
    export.mkdir()
    # Like the large CTC exports: one external file per tensor.
    onnx.save_model(
        onnx.load(str(staged)), str(export / "model.onnx"), save_as_external_data=True,
        all_tensors_to_one_file=False, size_threshold=0,
    )
    before = {path.name: path.read_bytes() for path in export.iterdir() if path.name != "model.onnx"}
    assert folding.fold_file(export / "model.onnx") == 1
    after = {path.name for path in export.iterdir() if path.name != "model.onnx"}
    added = after - before.keys()
    assert len(added) == 1 and next(iter(added)).endswith("_folded")
    assert all((export / name).read_bytes() == data for name, data in before.items())
    model = onnx.load(str(export / "model.onnx"), load_external_data=False)
    assert "ReduceL2" not in {node.op_type for node in model.graph.node}
    x = np.random.default_rng(2).normal(size=(1, 32, 50)).astype(np.float32)
    actual = ort.InferenceSession(str(export / "model.onnx")).run(None, {"x": x})[0]
    g, v = conv.weight_g.detach().double(), conv.weight_v.detach().double()
    weight = g * v / torch.linalg.vector_norm(v, dim=(0, 1), keepdim=True)
    expected = torch.nn.functional.conv1d(
        torch.from_numpy(x).double(), weight, conv.bias.detach().double(), padding=8, groups=4
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

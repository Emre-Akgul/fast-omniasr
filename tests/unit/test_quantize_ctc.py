"""tools/quantize_ctc.py: variant layout, calibration checks, gates and the quantizers."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("quantize_ctc", ROOT / "tools" / "quantize_ctc.py")
quantize_ctc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize_ctc)

GATES = json.loads((ROOT / "benchmarks" / "quantization" / "gates.json").read_text())
MANIFEST = {"languages": {"en_us": {"metric": "wer"}, "cmn_hans_cn": {"metric": "cer"}}}


def test_variant_paths(tmp_path):
    ctc, out = tmp_path / "ctc", tmp_path / "out"
    assert quantize_ctc.variant_path("fp32", ctc, out) == ctc / "model.onnx"
    assert quantize_ctc.variant_path("fp16", ctc, out) == ctc / "fp16/model.onnx"
    assert quantize_ctc.variant_path("int8-static", ctc, out) == out / "int8-static/model.onnx"
    assert quantize_ctc.variant_path("trt-int8", ctc, out) == out / "trt-int8/model.engine"
    with pytest.raises(ValueError, match="unknown variant"):
        quantize_ctc.variant_path("int2", ctc, out)


def test_engines_are_cuda_only_and_variants_unique():
    quantize_ctc.check_device_variants("cuda", ["fp32", "trt-int8"])
    with pytest.raises(ValueError, match="--device cuda"):
        quantize_ctc.check_device_variants("cpu", ["fp32", "trt-int8"])
    with pytest.raises(ValueError, match="twice"):
        quantize_ctc.check_device_variants("cpu", ["fp32", "fp32"])


def test_gates_file_lists_known_variants():
    for device in ("cpu", "cuda"):
        quantize_ctc.check_device_variants(device, GATES["variants"][device])
        assert GATES["speed_reference"][device] in GATES["variants"][device]
        assert set(GATES["shipped"][device]) <= set(GATES["variants"][device])


def _calibration(tmp_path, split="dev", data=b"RIFF"):
    (tmp_path / "en_us_1.wav").write_bytes(data)
    manifest = {"split": split, "clips": [
        {"file": "en_us_1.wav", "sha256": hashlib.sha256(b"RIFF").hexdigest()}]}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(manifest))
    return path


def test_read_calibration_verifies_clips(tmp_path):
    assert quantize_ctc.read_calibration(_calibration(tmp_path)) == [tmp_path / "en_us_1.wav"]


def test_read_calibration_rejects_changed_clips_and_the_test_split(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        quantize_ctc.read_calibration(_calibration(tmp_path, data=b"other"))
    with pytest.raises(ValueError, match="test split"):
        quantize_ctc.read_calibration(_calibration(tmp_path, split="test"))


def test_latency_summary_takes_the_median_pass():
    passes = [[1.0, 1.0], [3.0, 3.0], [2.0, 1.0]]
    summary = quantize_ctc.latency_summary(passes, audio_seconds=30.0)
    assert summary["total_seconds"] == 3.0
    assert summary["pass_totals"] == [2.0, 6.0, 3.0]
    assert summary["rtf"] == 0.1
    assert summary["clip_p50_ms"] == 2000.0  # from the median pass [2.0, 1.0]


def test_per_language_scores_splits_texts_in_order():
    manifest = {"languages": {
        "en_us": {"metric": "wer", "utterances": [{"reference": "a b"}, {"reference": "c"}]},
        "cmn_hans_cn": {"metric": "cer", "utterances": [{"reference": "你好。"}]}}}
    scores = quantize_ctc.per_language_scores(manifest, ["en_us", "cmn_hans_cn"],
                                              {"en_us": 2, "cmn_hans_cn": 1},
                                              ["a b", "x", "你 好"])
    assert scores["en_us"]["wer"] == pytest.approx(100 / 3)
    assert scores["cmn_hans_cn"] == {"wer": None, "cer": 0.0, "n": 1}


def test_quality_gate_allows_the_larger_of_absolute_and_relative_margins():
    baselines = {"en_us": {"baseline_wer": 40.0, "baseline_cer": 4.0},
                 "cmn_hans_cn": {"baseline_wer": None, "baseline_cer": 20.0}}
    passing = {"en_us": {"wer": 42.0, "cer": 4.5}, "cmn_hans_cn": {"wer": None, "cer": 21.0}}
    assert quantize_ctc.quality_failures(passing, baselines, GATES, MANIFEST) == []
    failing = {"en_us": {"wer": 42.1, "cer": 4.6}, "cmn_hans_cn": {"wer": None, "cer": 21.1}}
    failures = quantize_ctc.quality_failures(failing, baselines, GATES, MANIFEST)
    assert [f.split(":")[0] for f in failures] == ["en_us", "en_us", "cmn_hans_cn"]
    assert "WER 42.10 > 42.00" in failures[0]


def test_quality_gate_fails_languages_without_a_baseline():
    scores = {"en_us": {"wer": 1.0, "cer": 1.0}}
    assert quantize_ctc.quality_failures(scores, {}, GATES, MANIFEST) == [
        "en_us: no fp32 baseline"]


def _measured(seconds, rss, gpu=None):
    return {"latency": {"total_seconds": seconds}, "peak_rss_mib": rss, "peak_gpu_mib": gpu}


def test_efficiency_gate_needs_speed_or_memory():
    reference = _measured(10.0, 3000, 2000)
    assert quantize_ctc.efficiency_check(_measured(8.0, 2900), reference, GATES, "cpu")[0]
    assert quantize_ctc.efficiency_check(_measured(10.0, 2000), reference, GATES, "cpu")[0]
    passed, why = quantize_ctc.efficiency_check(_measured(9.0, 2900), reference, GATES, "cpu")
    assert not passed and "speedup 1.111x (needs >= 1.2)" in why
    # On CUDA the memory that counts is GPU memory, not host RSS.
    assert not quantize_ctc.efficiency_check(_measured(10.0, 100, 1900), reference, GATES,
                                             "cuda")[0]
    assert not quantize_ctc.efficiency_check(_measured(1.0, 1), None, GATES, "cpu")[0]


def _matmul_conv_graph(path):
    """audio [1, samples] -> Conv (kept float) -> MatMul with a weight (quantized) ->
    self-similarity MatMuls of two activations (kept float, like attention) -> logits."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    nodes = [
        helper.make_node("Unsqueeze", ["audio", "axis"], ["x3"]),
        helper.make_node("Conv", ["x3", "conv_w"], ["conv"], strides=[4], kernel_shape=[4],
                         name="/model/feature_extractor/conv"),
        helper.make_node("Transpose", ["conv"], ["frames"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["frames", "proj_w"], ["proj"], name="/model/proj"),
        helper.make_node("Transpose", ["proj"], ["proj_t"], perm=[0, 2, 1]),
        helper.make_node("MatMul", ["proj", "proj_t"], ["scores"], name="/model/attn/MatMul"),
        helper.make_node("MatMul", ["scores", "proj"], ["logits"], name="/model/attn/MatMul_1"),
    ]
    initializers = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis"),
        numpy_helper.from_array(rng.standard_normal((64, 1, 4)).astype(np.float32), "conv_w"),
        numpy_helper.from_array(rng.standard_normal((64, 32)).astype(np.float32), "proj_w"),
    ]
    graph = helper.make_graph(
        nodes, "tiny", [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "n"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, "f", 32])],
        initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18),
                                                    helper.make_opsetid("com.microsoft", 1)])
    model.ir_version = 9
    path.parent.mkdir(parents=True)
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("variant, op", [("int8-dynamic", "DynamicQuantizeLinear"),
                                         ("int8-weight-only", "MatMulNBits"),
                                         ("int4-weight-only", "MatMulNBits")])
def test_quantizers_touch_matmuls_only(tmp_path, variant, op):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.quantization")
    _matmul_conv_graph(tmp_path / "ctc" / "model.onnx")
    path = quantize_ctc.quantize(variant, tmp_path / "ctc", tmp_path / "out", None)
    nodes = onnx.load(path).graph.node
    ops = [node.op_type for node in nodes]
    assert op in ops and "Conv" in ops
    # Only the activation-by-activation MatMuls are left in float.
    assert sorted(n.name for n in nodes if n.op_type == "MatMul") == [
        "/model/attn/MatMul", "/model/attn/MatMul_1"]
    audio = np.random.default_rng(1).standard_normal((1, 400)).astype(np.float32)
    reference = ort.InferenceSession(str(tmp_path / "ctc" / "model.onnx")).run(
        None, {"audio": audio})[0]
    quantized = ort.InferenceSession(str(path)).run(None, {"audio": audio})[0]
    assert quantized.shape == reference.shape
    assert np.abs(quantized - reference).max() < 0.1 * np.abs(reference).max()
    assert quantize_ctc.file_bytes(path) == path.stat().st_size


def test_static_quantization_calibrates_and_emits_symmetric_qdq(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime.quantization")
    sf = pytest.importorskip("soundfile")
    from onnx import numpy_helper

    _matmul_conv_graph(tmp_path / "ctc" / "model.onnx")
    clip = tmp_path / "calib" / "en_us_1.wav"
    clip.parent.mkdir()
    sf.write(clip, 0.1 * np.random.default_rng(2).standard_normal(1600), 16000)
    path = quantize_ctc.quantize("int8-static", tmp_path / "ctc", tmp_path / "out", [clip])
    model = onnx.load(path)
    assert {"QuantizeLinear", "DequantizeLinear", "Conv"} <= {n.op_type
                                                               for n in model.graph.node}
    produced_by = {o: n.op_type for n in model.graph.node for o in n.output}
    node = {n.name: n for n in model.graph.node}
    # The linear layer reads quantized weight and input; the attention products don't.
    assert [produced_by[i] for i in node["/model/proj"].input] == ["DequantizeLinear"] * 2
    assert produced_by[node["/model/attn/MatMul_1"].input[0]] == "MatMul"  # scores: float
    zero_points = [numpy_helper.to_array(t) for t in model.graph.initializer
                   if t.name.endswith("_zero_point")]
    assert zero_points and all(z.dtype == np.int8 and not z.any() for z in zero_points)
    with pytest.raises(ValueError, match="calibration"):
        quantize_ctc.quantize("int8-static", tmp_path / "ctc", tmp_path / "again", None)

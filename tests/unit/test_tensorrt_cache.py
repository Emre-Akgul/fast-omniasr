import builtins
import json
import threading
import time

import pytest

from fast_omniasr import tensorrt_cache
from fast_omniasr.tensorrt_cache import get_or_build_engine

FAKE_ENV = {
    "tensorrt_version": "10.16.1.11",
    "compute_capability": "8.6",
    "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
}


@pytest.fixture
def counting_builder(monkeypatch):
    calls = []

    def fake_build_engine(onnx_path, output_path, *, precision, min_samples, opt_samples,
                           max_samples):
        calls.append(precision)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"FAKE ENGINE {precision}".encode())
        return output_path

    monkeypatch.setattr(tensorrt_cache, "build_engine", fake_build_engine)
    monkeypatch.setattr(tensorrt_cache, "_local_environment", lambda: dict(FAKE_ENV))
    return calls


# --- minimal protobuf encoder for building tiny, valid ONNX ModelProto bytes ---

def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, payload):
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _external_tensor(name, location):
    entries = b"".join(
        _field(13, _field(1, key) + _field(2, value))
        for key, value in (("location", location), ("offset", "0"), ("length", "4"))
    )
    # dims=1, data_type=FLOAT, name, raw_data (ignored), external_data..., data_location=EXTERNAL
    return (_field(1, 1) + _field(2, 1) + _field(8, name) + _field(9, b"\x00" * 16)
            + entries + _field(14, 1))


def _model(*, doc="", initializers=(), sparse=(), node_tensors=(), subgraph_tensors=(),
           function_tensors=()):
    graph = b"".join(_field(5, t) for t in initializers)
    graph += b"".join(_field(15, _field(1, t) + _field(2, idx)) for t, idx in sparse)
    for tensor in node_tensors:  # Constant node: attribute { name: "value", t: tensor }
        graph += _field(1, _field(4, "Constant") + _field(5, _field(1, "value") + _field(5, tensor)))
    if subgraph_tensors:  # If node with a then_branch subgraph holding initializers
        inner = b"".join(_field(5, t) for t in subgraph_tensors)
        graph += _field(1, _field(4, "If") + _field(5, _field(1, "then_branch") + _field(6, inner)))
    model = _field(1, 8) + (_field(6, doc) if doc else b"") + _field(7, graph)
    for tensor in function_tensors:
        node = _field(4, "Constant") + _field(5, _field(1, "value") + _field(5, tensor))
        model += _field(25, _field(1, "fn") + _field(7, node))
    return model


@pytest.fixture
def onnx_a(tmp_path):
    path = tmp_path / "a.onnx"
    path.write_bytes(_model(doc="model A"))
    return path


@pytest.fixture
def onnx_b(tmp_path):
    path = tmp_path / "b.onnx"
    path.write_bytes(_model(doc="model B, totally different"))
    return path


def test_empty_cache_builds(counting_builder, onnx_a, tmp_path):
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=tmp_path / "cache")
    assert len(counting_builder) == 1
    assert engine_path.exists()
    metadata = json.loads((engine_path.parent / "metadata.json").read_text())
    assert metadata["profile"] == {"min": 400, "opt": 80000, "max": 480000}


def test_second_call_reuses_cache(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    second = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert first == second
    assert len(counting_builder) == 1


def test_onnx_hash_change_triggers_rebuild(counting_builder, onnx_a, onnx_b, tmp_path):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    second = get_or_build_engine(onnx_b, precision="fp16", cache_dir=cache_dir)
    assert first != second
    assert len(counting_builder) == 2


def test_precision_change_produces_distinct_entries(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    fp16 = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    fp32 = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache_dir)
    assert fp16 != fp32
    assert len(counting_builder) == 2
    assert fp16.exists() and fp32.exists()  # neither call clobbered the other's entry


def test_profile_change_produces_distinct_entries(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    default = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache_dir)
    custom = get_or_build_engine(
        onnx_a, precision="fp32", min_samples=8000, cache_dir=cache_dir
    )
    assert default != custom
    assert len(counting_builder) == 2


def test_incompatible_metadata_triggers_rebuild(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    metadata_path = engine_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tensorrt_version"] = "1.0.0.0"  # simulate a stale/incompatible record
    metadata_path.write_text(json.dumps(metadata))

    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert len(counting_builder) == 2


def test_missing_or_corrupted_metadata_triggers_rebuild(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    engine_path = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    (engine_path.parent / "metadata.json").write_text("not valid json{")

    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    assert len(counting_builder) == 2


def test_invalid_precision_raises(onnx_a, tmp_path):
    with pytest.raises(ValueError, match="precision"):
        get_or_build_engine(onnx_a, precision="int8", cache_dir=tmp_path / "cache")


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ((399, 80000, 480000), "at least 400"),
        ((16000, 8000, 480000), "min_samples <= opt_samples <= max_samples"),
        ((400, 80000, 79999), "min_samples <= opt_samples <= max_samples"),
        ((400.0, 80000, 480000), "must be integers"),
    ],
)
def test_invalid_profile_raises_before_querying_gpu(onnx_a, tmp_path, profile, message):
    with pytest.raises(ValueError, match=message):
        get_or_build_engine(
            onnx_a,
            precision="fp32",
            min_samples=profile[0],
            opt_samples=profile[1],
            max_samples=profile[2],
            cache_dir=tmp_path / "cache",
        )


def test_concurrent_builds_of_same_key_build_only_once(counting_builder, onnx_a, tmp_path,
                                                         monkeypatch):
    cache_dir = tmp_path / "cache"
    real_build = tensorrt_cache.build_engine

    def slow_build(*args, **kwargs):
        time.sleep(0.2)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(tensorrt_cache, "build_engine", slow_build)
    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(
                get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
            )
        )
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(counting_builder) == 1
    assert len(set(results)) == 1


def test_missing_tensorrt_raises_helpful_import_error(monkeypatch, onnx_a, tmp_path):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "tensorrt":
            raise ImportError("simulated: tensorrt not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[tensorrt\\]"):
        get_or_build_engine(onnx_a, precision="fp16", cache_dir=tmp_path / "cache")


# --- external-data weights are part of the cache identity (issue #10) ---

@pytest.fixture
def external_model(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "model.onnx.data").write_bytes(b"weights v1")
    (model_dir / "weights" / "nested.bin").write_bytes(b"nested v1")
    path = model_dir / "model.onnx"
    path.write_bytes(_model(initializers=[
        _external_tensor("w0", "model.onnx.data"),
        _external_tensor("w1", "weights/nested.bin"),
    ]))
    return path


def test_changing_only_external_weights_triggers_rebuild(counting_builder, external_model,
                                                         tmp_path):
    cache_dir = tmp_path / "cache"
    graph_bytes = external_model.read_bytes()
    first = get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir)
    assert get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir) == first
    assert len(counting_builder) == 1

    (external_model.parent / "model.onnx.data").write_bytes(b"weights v2")
    assert external_model.read_bytes() == graph_bytes  # graph untouched
    second = get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir)
    assert second != first
    assert len(counting_builder) == 2

    (external_model.parent / "weights" / "nested.bin").write_bytes(b"nested v2")
    third = get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir)
    assert third not in (first, second)
    assert len(counting_builder) == 3


def test_metadata_records_external_weight_hashes(counting_builder, external_model, tmp_path):
    engine_path = get_or_build_engine(external_model, precision="fp32",
                                      cache_dir=tmp_path / "cache")
    metadata = json.loads((engine_path.parent / "metadata.json").read_text())
    assert sorted(metadata["external_data_sha256"]) == ["model.onnx.data", "weights/nested.bin"]


def test_unreferenced_sibling_files_do_not_affect_cache(counting_builder, external_model,
                                                         tmp_path):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir)
    # e.g. another precision variant or the tokenizer downloaded into the same snapshot
    (external_model.parent / "model_fp16.onnx").write_bytes(_model(doc="other variant"))
    (external_model.parent / "tokenizer.model").write_bytes(b"tokenizer")
    assert get_or_build_engine(external_model, precision="fp32", cache_dir=cache_dir) == first
    assert len(counting_builder) == 1


def test_missing_external_weights_raise_before_build(counting_builder, external_model, tmp_path):
    (external_model.parent / "weights" / "nested.bin").unlink()
    with pytest.raises(FileNotFoundError, match="weights/nested.bin"):
        get_or_build_engine(external_model, precision="fp32", cache_dir=tmp_path / "cache")
    assert counting_builder == []


def test_external_locations_found_in_every_tensor_container(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(_model(
        initializers=[_external_tensor("a", "init.bin")],
        sparse=[(_external_tensor("v", "sparse_values.bin"),
                 _external_tensor("i", "sparse_indices.bin"))],
        node_tensors=[_external_tensor("c", "constant.bin")],
        subgraph_tensors=[_external_tensor("s", "subgraph.bin")],
        function_tensors=[_external_tensor("f", "function.bin")],
    ))
    assert tensorrt_cache._external_data_locations(path) == sorted([
        "init.bin", "sparse_values.bin", "sparse_indices.bin", "constant.bin",
        "subgraph.bin", "function.bin",
    ])


def test_inline_weights_have_no_external_locations(onnx_a):
    assert tensorrt_cache._external_data_locations(onnx_a) == []


@pytest.mark.parametrize("payload", [b"", b"not an onnx model", _model(doc="x")[:-3]])
def test_invalid_onnx_raises_clear_error(payload, tmp_path):
    path = tmp_path / "bad.onnx"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="bad.onnx"):
        tensorrt_cache._external_data_locations(path)


def test_external_locations_match_onnx_package(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import helper, numpy_helper

    weights = [numpy_helper.from_array(np.full((64, 64), i, np.float32), f"w{i}")
               for i in range(3)]
    then_graph = helper.make_graph(
        [helper.make_node("Constant", [], ["t"],
                          value=numpy_helper.from_array(np.ones(512, np.float32), "k"))],
        "then", [], [helper.make_tensor_value_info("t", onnx.TensorProto.FLOAT, [512])],
    )
    graph = helper.make_graph(
        [helper.make_node("Add", ["w0", "w1"], ["s"]),
         helper.make_node("If", ["cond"], ["t"], then_branch=then_graph, else_branch=then_graph)],
        "g", [helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("s", onnx.TensorProto.FLOAT, [64, 64])],
        initializer=weights,
    )
    model = helper.make_model(graph)
    path = tmp_path / "real.onnx"
    onnx.save_model(model, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=False, size_threshold=0)
    reloaded = onnx.load(str(path), load_external_data=False)
    expected = set()

    def collect(g):
        tensors = list(g.initializer)
        for node in g.node:
            for attr in node.attribute:
                tensors += [attr.t] if attr.HasField("t") else []
                tensors += list(attr.tensors)
                for sub in ([attr.g] if attr.HasField("g") else []) + list(attr.graphs):
                    collect(sub)
        for tensor in tensors:
            expected.update(e.value for e in tensor.external_data if e.key == "location")

    collect(reloaded.graph)
    assert expected  # the fixture really uses external data
    assert tensorrt_cache._external_data_locations(path) == sorted(expected)

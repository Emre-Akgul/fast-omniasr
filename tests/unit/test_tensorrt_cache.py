import builtins
import json
import threading
import time
import types

import pytest

from fast_omniasr import tensorrt_cache
from fast_omniasr.tensorrt_cache import get_or_build_engine

FAKE_ENV = {
    "tensorrt_version": "10.16.1.11",
    "tensorrt_cuda": "cu12",
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
           function_tensors=(), function_default_tensors=()):
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
    for tensor in function_default_tensors:  # FunctionProto.attribute_proto default value
        model += _field(25, _field(1, "fn") + _field(11, _field(1, "value") + _field(5, tensor)))
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
        function_default_tensors=[_external_tensor("d", "function_default.bin")],
    ))
    assert tensorrt_cache._external_data_locations(path) == sorted([
        "init.bin", "sparse_values.bin", "sparse_indices.bin", "constant.bin",
        "subgraph.bin", "function.bin", "function_default.bin",
    ])


def test_pathologically_nested_graph_raises_value_error(tmp_path):
    graph = b""
    for _ in range(2000):  # If -> then_branch -> If -> ... far beyond any real model
        graph = _field(1, _field(4, "If") + _field(5, _field(1, "then_branch") + _field(6, graph)))
    path = tmp_path / "deep.onnx"
    path.write_bytes(_field(1, 8) + _field(7, graph))
    with pytest.raises(ValueError, match="deep.onnx"):
        tensorrt_cache._external_data_locations(path)


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


# --- the fingerprint describes the device actually in use, plus CUDA (issue #11) ---

class _Prop:
    def __init__(self, name, major, minor):
        self.name, self.major, self.minor = name, major, minor


def _fake_cuda(monkeypatch, *, current=1, runtime=12080, fail=None,
               wheels=("tensorrt_cu12", "tensorrt_cu12_libs")):
    import sys
    import types

    devices = {0: _Prop(b"Example GPU A\x00\x00", 7, 5), 1: _Prop("Example GPU B", 8, 0)}
    queried = []

    def result(name, value):
        return (1 if fail == name else 0, value)

    def get_props(device):
        queried.append(device)
        return result("props", devices[device])

    cudart = types.SimpleNamespace(
        cudaGetDevice=lambda: result("device", current),
        cudaGetDeviceProperties=get_props,
        cudaRuntimeGetVersion=lambda: result("runtime", runtime),
    )
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = cudart
    cuda = types.ModuleType("cuda")
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "tensorrt", types.SimpleNamespace(__version__="10.16.1.11"))
    mapping = {}
    for wheel in wheels:
        package = "tensorrt_libs" if wheel.endswith("_libs") else "tensorrt"
        mapping.setdefault(package, []).append(wheel)
    monkeypatch.setattr(tensorrt_cache.metadata, "packages_distributions", lambda: mapping)
    return queried


def test_environment_queries_the_current_device_not_device_zero(monkeypatch):
    queried = _fake_cuda(monkeypatch, current=1)
    env = tensorrt_cache._local_environment()
    assert queried == [1]
    assert env == {
        "tensorrt_version": "10.16.1.11",
        "tensorrt_cuda": "cu12",
        "compute_capability": "8.0",
        "device_name": "Example GPU B",
    }


def test_environment_decodes_nul_padded_device_name(monkeypatch):
    _fake_cuda(monkeypatch, current=0)
    env = tensorrt_cache._local_environment()
    assert env["device_name"] == "Example GPU A"
    assert env["compute_capability"] == "7.5"


@pytest.mark.parametrize("wheels, expected", [
    (("tensorrt_cu13", "tensorrt_cu13_libs"), "cu13"),
    (("tensorrt-cu12",), "cu12"),
    (("tensorrt_cu13_libs",), "cu13"),
])
def test_tensorrt_cuda_comes_from_the_tensorrt_wheel(monkeypatch, wheels, expected):
    _fake_cuda(monkeypatch, runtime=13040, wheels=wheels)  # cuda-python version is ignored
    assert tensorrt_cache._local_environment()["tensorrt_cuda"] == expected


def test_tensorrt_cuda_falls_back_to_cuda_python_major(monkeypatch):
    _fake_cuda(monkeypatch, runtime=13040, wheels=("tensorrt",))  # e.g. tarball install
    assert tensorrt_cache._local_environment()["tensorrt_cuda"] == "cuda-python-13"


@pytest.mark.parametrize("fail, message", [
    ("device", "current CUDA device"),
    ("props", "properties of CUDA device 1"),
])
def test_environment_query_failures_are_reported(monkeypatch, fail, message):
    _fake_cuda(monkeypatch, fail=fail)
    with pytest.raises(RuntimeError, match=message):
        tensorrt_cache._local_environment()


def test_runtime_query_failure_reported_in_fallback(monkeypatch):
    _fake_cuda(monkeypatch, fail="runtime", wheels=())
    with pytest.raises(RuntimeError, match="CUDA runtime version"):
        tensorrt_cache._local_environment()


def _fake_builder(monkeypatch):
    builds = []

    def fake_build_engine(onnx_path, output_path, **kwargs):
        builds.append(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"engine")

    monkeypatch.setattr(tensorrt_cache, "build_engine", fake_build_engine)
    return builds


def test_switching_device_or_tensorrt_cuda_uses_a_different_engine(monkeypatch, onnx_a,
                                                                   tmp_path):
    builds = _fake_builder(monkeypatch)
    cache = tmp_path / "cache"
    _fake_cuda(monkeypatch, current=0)
    on_gpu_a = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache)
    _fake_cuda(monkeypatch, current=1)
    on_gpu_b = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache)
    _fake_cuda(monkeypatch, current=1, wheels=("tensorrt_cu13", "tensorrt_cu13_libs"))
    on_gpu_b_cu13 = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache)
    _fake_cuda(monkeypatch, current=1)
    again = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache)
    assert len({on_gpu_a, on_gpu_b, on_gpu_b_cu13}) == 3
    assert again == on_gpu_b
    assert len(builds) == 3


def test_cuda_python_upgrade_does_not_rebuild(monkeypatch, onnx_a, tmp_path):
    builds = _fake_builder(monkeypatch)
    cache = tmp_path / "cache"
    _fake_cuda(monkeypatch, runtime=13040)
    first = get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache)
    _fake_cuda(monkeypatch, runtime=13050)
    assert get_or_build_engine(onnx_a, precision="fp32", cache_dir=cache) == first
    assert len(builds) == 1


def test_mixed_precision_has_its_own_cache_entry(counting_builder, onnx_a, tmp_path):
    cache_dir = tmp_path / "cache"
    fp16 = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache_dir)
    mixed = get_or_build_engine(onnx_a, precision="mixed", cache_dir=cache_dir)
    assert fp16 != mixed and counting_builder == ["fp16", "mixed"]
    assert get_or_build_engine(onnx_a, precision="mixed", cache_dir=cache_dir) == mixed
    metadata = json.loads((mixed.parent / "metadata.json").read_text())
    assert metadata["mixed_recipe"]["fp32_names"] == ["/model/pos_encoder/conv/Conv"]


def test_changing_the_mixed_recipe_rebuilds(counting_builder, onnx_a, tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    first = get_or_build_engine(onnx_a, precision="mixed", cache_dir=cache_dir)
    monkeypatch.setattr(tensorrt_cache, "MIXED_RECIPE",
                        {"fp32_prefixes": ["/model/feature_extractor/"], "fp32_names": []})
    assert get_or_build_engine(onnx_a, precision="mixed", cache_dir=cache_dir) != first
    assert counting_builder == ["mixed", "mixed"]


# --- cached engines are integrity-checked before reuse (issue #15) ---

def _engine_and_metadata(path):
    return path, json.loads((path.parent / "metadata.json").read_text())


def test_metadata_records_the_engine_checksum(counting_builder, onnx_a, tmp_path):
    import hashlib

    engine, metadata = _engine_and_metadata(
        get_or_build_engine(onnx_a, precision="fp16", cache_dir=tmp_path / "c"))
    data = engine.read_bytes()
    assert metadata["engine"] == {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.mark.parametrize("damage", [
    lambda p: p.write_bytes(b"FAKE ENGINE fp1X"),           # same size, different bytes
    lambda p: p.write_bytes(p.read_bytes()[:5]),             # truncated
])
def test_corrupted_engine_is_rebuilt_with_a_warning(counting_builder, onnx_a, tmp_path, damage):
    cache = tmp_path / "c"
    engine = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    damage(engine)
    with pytest.warns(UserWarning, match="does not match its recorded checksum"):
        assert get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache) == engine
    assert counting_builder == ["fp16", "fp16"]
    assert engine.read_bytes() == b"FAKE ENGINE fp16"
    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)  # healthy again: reused
    assert len(counting_builder) == 2


def test_entries_cached_before_checksums_are_adopted_not_rebuilt(counting_builder, onnx_a,
                                                                 tmp_path):
    cache = tmp_path / "c"
    engine = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    metadata_path = engine.parent / "metadata.json"
    legacy = json.loads(metadata_path.read_text())
    del legacy["engine"]
    metadata_path.write_text(json.dumps(legacy))
    assert get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache) == engine
    assert counting_builder == ["fp16"]
    assert "engine" in json.loads(metadata_path.read_text())


def test_rebuild_forces_a_new_engine(counting_builder, onnx_a, tmp_path):
    cache = tmp_path / "c"
    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache, rebuild=True)
    assert counting_builder == ["fp16", "fp16"]


def test_failed_build_leaves_no_record_that_vouches_for_it(counting_builder, onnx_a,
                                                           tmp_path, monkeypatch):
    cache = tmp_path / "c"
    engine = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)

    def crashing_build(onnx_path, output_path, **kwargs):
        output_path.write_bytes(b"partial")
        raise RuntimeError("builder crashed")

    monkeypatch.setattr(tensorrt_cache, "build_engine", crashing_build)
    with pytest.raises(RuntimeError, match="builder crashed"):
        get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache, rebuild=True)
    assert not (engine.parent / "metadata.json").exists()


def _from_pretrained_with(monkeypatch, tmp_path, failures):
    """OmniASR.from_pretrained(backend='tensorrt') with fake download, cache and backend;
    the backend raises the queued errors in order, then succeeds."""
    from fast_omniasr import OmniASR, hub, model

    monkeypatch.setattr(hub, "download_verified", lambda *a, **k: (
        {}, {"model.onnx": tmp_path / "m.onnx", "tokenizer.model": tmp_path / "t.model"}))
    rebuilds = []

    def fake_cache(onnx_path, **kwargs):
        rebuilds.append({k: kwargs[k] for k in ("rebuild", "discard_sha256") if k in kwargs})
        return tmp_path / "e.engine"

    monkeypatch.setattr(tensorrt_cache, "get_or_build_engine", fake_cache)
    real_init = model.OmniASR.__init__

    def fake_init(self, *args, **kwargs):
        if failures:
            raise failures.pop(0)
        self.backend = types.SimpleNamespace(closed=False, close=lambda: None)

    monkeypatch.setattr(model.OmniASR, "__init__", fake_init)
    try:
        import warnings as warnings_module

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            result = OmniASR.from_pretrained("o/r", backend="tensorrt", precision="fp16")
        return result, rebuilds, caught
    finally:
        monkeypatch.setattr(model.OmniASR, "__init__", real_init)


def _deserialize_error():
    from fast_omniasr.backends.tensorrt import EngineDeserializationError

    return EngineDeserializationError("Failed to deserialize TensorRT engine")


def test_undeserializable_engine_is_rebuilt_once(monkeypatch, tmp_path):
    _, rebuilds, caught = _from_pretrained_with(monkeypatch, tmp_path, [_deserialize_error()])
    assert rebuilds == [{}, {"rebuild": True}]  # no recorded checksum for this engine
    assert any("could not be deserialized" in str(w.message) and "transient" in str(w.message)
               for w in caught)


def test_rebuild_targets_the_engine_that_failed(monkeypatch, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"engine": {"sha256": "bad"}}))
    _, rebuilds, _ = _from_pretrained_with(monkeypatch, tmp_path, [_deserialize_error()])
    assert rebuilds == [{}, {"discard_sha256": "bad"}]


@pytest.mark.parametrize("make_failures", [
    lambda: [_deserialize_error(), _deserialize_error()],              # rebuilt engine also bad
    lambda: [RuntimeError("Failed to create TensorRT execution context")],  # not a bad file
    lambda: [RuntimeError("something about deserialize, but not the engine")],
])
def test_other_or_repeated_failures_are_raised(monkeypatch, tmp_path, make_failures):
    with pytest.raises(RuntimeError):
        _from_pretrained_with(monkeypatch, tmp_path, make_failures())


def test_discard_reuses_an_engine_another_process_already_rebuilt(counting_builder, onnx_a,
                                                                  tmp_path):
    from fast_omniasr.tensorrt_cache import recorded_engine_sha256

    cache = tmp_path / "c"
    engine = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    failed = recorded_engine_sha256(engine)
    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache, discard_sha256=failed)
    assert len(counting_builder) == 2  # it was still the failed engine: rebuilt
    engine.write_bytes(b"rebuilt by another process")
    metadata = json.loads((engine.parent / "metadata.json").read_text())
    metadata["engine"] = tensorrt_cache._engine_record(engine)
    (engine.parent / "metadata.json").write_text(json.dumps(metadata))
    get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache, discard_sha256=failed)
    assert len(counting_builder) == 2  # someone else already replaced it: reused


@pytest.mark.parametrize("record", ["not a dict", {"size": "x"}, {"sha256": "y"}])
def test_malformed_engine_record_triggers_a_rebuild(counting_builder, onnx_a, tmp_path, record):
    cache = tmp_path / "c"
    engine = get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    metadata = json.loads((engine.parent / "metadata.json").read_text())
    metadata["engine"] = record
    (engine.parent / "metadata.json").write_text(json.dumps(metadata))
    with pytest.warns(UserWarning, match="recorded checksum"):
        get_or_build_engine(onnx_a, precision="fp16", cache_dir=cache)
    assert len(counting_builder) == 2

"""build_engine precision recipes, exercised with a fake tensorrt module (no GPU)."""

import os
import stat
import sys
import types
from pathlib import Path

import pytest

from fast_omniasr import tensorrt_builder
from fast_omniasr.tensorrt_builder import TensorRTBuildOptions

F32, F16, I64 = "float32", "float16", "int64"


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


class Layer:
    def __init__(self, name, kind="CONVOLUTION", outputs=(F32,)):
        self.name, self.type = name, kind
        self.outputs = [Tensor(d) for d in outputs]
        self.precision = None
        self.output_types = {}

    @property
    def num_outputs(self):
        return len(self.outputs)

    def get_output(self, index):
        return self.outputs[index]

    def set_output_type(self, index, dtype):
        self.output_types[index] = dtype


def ctc_like_layers():
    return [
        Layer("/model/feature_extractor/layers/layers.0/conv/Conv"),
        Layer("/model/feature_extractor/layers/layers.0/layer_norm/LayerNormalization",
              "NORMALIZATION"),
        Layer("/model/feature_extractor/layers/layers.6/conv/Conv"),
        Layer("/model/feature_extractor/Constant_output_0", "CONSTANT"),
        Layer("/model/feature_extractor/Shape", "SHAPE", (I64,)),
        Layer("/model/feature_extractor/Gather", "GATHER", (I64,)),
        Layer("/model/pos_encoder/conv/Conv"),
        Layer("/model/pos_encoder/activation/Erf", "UNARY"),
        Layer("/model/encoder/layers.0/self_attn/MatMul", "MATRIX_MULTIPLY"),
        Layer("/model/encoder/layers.0/self_attn_layer_norm/LayerNormalization",
              "NORMALIZATION"),
        Layer("/model/final_proj/MatMul", "MATRIX_MULTIPLY"),
    ]


@pytest.fixture
def fake_trt(monkeypatch):
    built = {}

    class Network:
        def __init__(self):
            self.layers = built.setdefault("layers", ctc_like_layers())

        @property
        def num_layers(self):
            return len(self.layers)

        def get_layer(self, index):
            return self.layers[index]

    class Profile:
        def set_shape(self, name, *shapes):
            self.shapes = [tuple(s) for s in shapes]

        def get_shape(self, name):
            return self.shapes

    class TimingCache:
        def serialize(self):
            return b"measurements"

    class Config:
        builder_optimization_level = 3
        timing_cache = None

        def __init__(self):
            self.flags = set()
            built["config"] = self

        def set_flag(self, flag):
            self.flags.add(flag)

        def clear_flag(self, flag):
            self.flags.discard(flag)

        def set_memory_pool_limit(self, pool, size):
            self.workspace = size

        def create_timing_cache(self, serialized):
            built["cache_read"] = serialized
            return TimingCache()

        def set_timing_cache(self, cache, ignore_mismatch):
            built["ignore_mismatch"] = ignore_mismatch
            accepted = built.get("accept_timing_cache", True)
            self.timing_cache = cache if accepted else None
            return accepted

        def get_timing_cache(self):
            return self.timing_cache

        def add_optimization_profile(self, profile):
            pass

    class Builder:
        def __init__(self, logger):
            pass

        def create_network(self, flags):
            return Network()

        def create_builder_config(self):
            return Config()

        def create_optimization_profile(self):
            return Profile()

        def build_serialized_network(self, network, config):
            return built.get("engine", b"engine")

    class Parser:
        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            return True

    built["config_type"], built["timing_cache_type"] = Config, TimingCache
    trt = types.SimpleNamespace(
        Logger=type("Logger", (), {"INFO": 0, "__init__": lambda self, level: None}),
        Builder=Builder, OnnxParser=Parser,
        BuilderFlag=types.SimpleNamespace(TF32="TF32", FP16="FP16",
                                          OBEY_PRECISION_CONSTRAINTS="OBEY"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="WORKSPACE"),
        LayerType=types.SimpleNamespace(CONSTANT="CONSTANT", SHAPE="SHAPE"),
        float32=F32, float16=F16,
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    return built


def _build(tmp_path, precision):
    return tensorrt_builder.build_engine(tmp_path / "m.onnx", tmp_path / "out.engine",
                                         precision=precision)


def test_fp16_sets_only_the_fp16_flag(fake_trt, tmp_path):
    _build(tmp_path, "fp16")
    assert fake_trt["config"].flags == {"FP16"}
    assert all(layer.precision is None for layer in fake_trt["layers"])


def test_fp32_sets_no_reduced_precision_flags(fake_trt, tmp_path):
    _build(tmp_path, "fp32")
    assert fake_trt["config"].flags == set()


def test_mixed_pins_exactly_the_convolutional_front_end(fake_trt, tmp_path):
    out = _build(tmp_path, "mixed")
    assert out.read_bytes() == b"engine"
    assert fake_trt["config"].flags == {"FP16", "OBEY"}
    pinned = {layer.name for layer in fake_trt["layers"] if layer.precision == F32}
    assert pinned == {
        "/model/feature_extractor/layers/layers.0/conv/Conv",
        "/model/feature_extractor/layers/layers.0/layer_norm/LayerNormalization",
        "/model/feature_extractor/layers/layers.6/conv/Conv",
        "/model/pos_encoder/conv/Conv",
    }  # constants, shape/integer layers, pos activation and the encoder stay unconstrained
    for layer in fake_trt["layers"]:
        assert layer.output_types == ({0: F32} if layer.name in pinned else {})


def test_mixed_refuses_graphs_without_the_front_end(fake_trt, tmp_path):
    fake_trt["layers"] = [Layer("/model/encoder/layers.0/self_attn/MatMul")]
    with pytest.raises(ValueError, match="could not find the OmniASR front-end"):
        _build(tmp_path, "mixed")
    assert not (tmp_path / "out.engine").exists()


def test_mixed_refuses_a_partially_matching_graph(fake_trt, tmp_path):
    # e.g. a re-export renamed the positional conv: don't silently pin only half.
    fake_trt["layers"] = [Layer("/model/feature_extractor/layers/layers.0/conv/Conv"),
                          Layer("/model/pos_encoder/conv/conv/Conv")]
    with pytest.raises(ValueError, match="pos_encoder/conv/Conv"):
        _build(tmp_path, "mixed")


def test_unknown_precision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="'fp32', 'fp16' or 'mixed'"):
        _build(tmp_path, "int8")


def _build_with(tmp_path, **options):
    return tensorrt_builder.build_engine(tmp_path / "m.onnx", tmp_path / "out.engine",
                                         options=TensorRTBuildOptions(**options))


def test_defaults_match_the_original_recipe(fake_trt, tmp_path):
    tensorrt_builder.build_engine(tmp_path / "m.onnx", tmp_path / "out.engine")
    config = fake_trt["config"]
    assert config.workspace == 1024**3
    assert config.builder_optimization_level == 3  # left at TensorRT's own default
    assert "cache_read" not in fake_trt


def test_build_options_reach_the_builder(fake_trt, tmp_path):
    _build_with(tmp_path, workspace_bytes=2**31, optimization_level=5)
    assert fake_trt["config"].workspace == 2**31
    assert fake_trt["config"].builder_optimization_level == 5


def test_optimization_level_zero_is_applied(fake_trt, tmp_path):
    _build_with(tmp_path, optimization_level=0)
    assert fake_trt["config"].builder_optimization_level == 0


def test_timing_cache_starts_empty_and_is_written_atomically(fake_trt, tmp_path):
    path = tmp_path / "nested" / "timing.bin"
    _build_with(tmp_path, timing_cache=path)
    assert fake_trt["cache_read"] == b""
    assert path.read_bytes() == b"measurements"
    assert [entry.name for entry in path.parent.iterdir()] == ["timing.bin"]


def test_timing_cache_is_reused_and_replaced(fake_trt, tmp_path):
    path = tmp_path / "timing.bin"
    path.write_bytes(b"earlier")
    _build_with(tmp_path, timing_cache=str(path))
    assert fake_trt["cache_read"] == b"earlier"
    assert fake_trt["ignore_mismatch"] is False  # another device's timings aren't trusted
    assert path.read_bytes() == b"measurements"


def test_rejected_timing_cache_warns_and_still_builds(fake_trt, tmp_path):
    fake_trt["accept_timing_cache"] = False
    with pytest.warns(UserWarning, match="rejected the timing cache"):
        engine = _build_with(tmp_path, timing_cache=tmp_path / "timing.bin")
    assert engine.read_bytes() == b"engine"


def test_unreadable_timing_cache_warns_and_starts_empty(fake_trt, tmp_path, monkeypatch):
    path = tmp_path / "timing.bin"
    path.write_bytes(b"earlier")
    real_read = Path.read_bytes

    def denied(self):
        if self == path:
            raise PermissionError("denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", denied)
    with pytest.warns(UserWarning, match="Could not read the timing cache"):
        engine = _build_with(tmp_path, timing_cache=path)
    assert fake_trt["cache_read"] == b""
    assert engine.read_bytes() == b"engine"


def test_timing_cache_save_failure_keeps_the_finished_engine(fake_trt, tmp_path, monkeypatch):
    path = tmp_path / "timing.bin"
    path.write_bytes(b"earlier")
    real_replace = os.replace

    def replace(source, target):
        if Path(target) == path:
            raise PermissionError("denied")
        real_replace(source, target)

    monkeypatch.setattr(tensorrt_builder.os, "replace", replace)
    with pytest.warns(UserWarning, match="Could not save the timing cache"):
        engine = _build_with(tmp_path, timing_cache=path)
    assert engine.read_bytes() == b"engine"
    assert path.read_bytes() == b"earlier"  # replaced whole or not at all
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.engine", "timing.bin"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_timing_cache_gets_normal_file_permissions(fake_trt, tmp_path):
    path = tmp_path / "timing.bin"
    previous = os.umask(0o002)
    try:
        _build_with(tmp_path, timing_cache=path)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_failed_build_leaves_the_timing_cache_alone(fake_trt, tmp_path):
    path = tmp_path / "timing.bin"
    path.write_bytes(b"earlier")
    fake_trt["engine"] = None
    with pytest.raises(RuntimeError, match="build failed"):
        _build_with(tmp_path, timing_cache=path)
    assert path.read_bytes() == b"earlier"


@pytest.mark.parametrize("options,message", [
    ({"workspace_bytes": 0}, "workspace_bytes"),
    ({"workspace_bytes": -1}, "workspace_bytes"),
    ({"workspace_bytes": 1.5}, "workspace_bytes"),
    ({"workspace_bytes": True}, "workspace_bytes"),
    ({"workspace_bytes": 2**63}, "workspace_bytes"),
    ({"optimization_level": -1}, "optimization_level"),
    ({"optimization_level": 6}, "optimization_level"),
    ({"optimization_level": True}, "optimization_level"),
    ({"optimization_level": "3"}, "optimization_level"),
    ({"timing_cache": ""}, "empty path"),
])
def test_invalid_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        TensorRTBuildOptions(**options)


def test_timing_cache_at_a_directory_warns_and_still_builds(fake_trt, tmp_path):
    with pytest.warns(UserWarning, match="timing cache"):
        engine = _build_with(tmp_path, timing_cache=tmp_path)
    assert engine.read_bytes() == b"engine"


@pytest.mark.parametrize("name", ["create_timing_cache", "set_timing_cache", "get_timing_cache"])
def test_tensorrt_errors_in_the_timing_cache_never_stop_the_build(fake_trt, tmp_path, monkeypatch,
                                                                  name):
    config = fake_trt["config_type"]

    def fail(self, *args):
        raise RuntimeError("TensorRT failure")

    monkeypatch.setattr(config, name, fail)
    with pytest.warns(UserWarning, match="timing cache"):
        engine = _build_with(tmp_path, timing_cache=tmp_path / "timing.bin")
    assert engine.read_bytes() == b"engine"


def test_serialize_failure_keeps_the_engine_and_leaves_no_temp_file(fake_trt, tmp_path, monkeypatch):
    def fail(self):
        raise RuntimeError("TensorRT failure")

    monkeypatch.setattr(fake_trt["timing_cache_type"], "serialize", fail)
    with pytest.warns(UserWarning, match="Could not save"):
        engine = _build_with(tmp_path, timing_cache=tmp_path / "timing.bin")
    assert engine.read_bytes() == b"engine"
    assert [p.name for p in tmp_path.iterdir()] == ["out.engine"]


def test_constructing_options_does_not_touch_the_filesystem(monkeypatch):
    def fail(self):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "is_dir", fail)
    monkeypatch.setattr(Path, "exists", fail)
    assert TensorRTBuildOptions(timing_cache="x" * 300).timing_cache.name == "x" * 300


def test_options_are_normalized():
    options = TensorRTBuildOptions(workspace_bytes=np_int(2**30), timing_cache="timing.bin")
    assert type(options.workspace_bytes) is int
    assert options.timing_cache.name == "timing.bin"
    assert TensorRTBuildOptions().identity() == {}


def np_int(value):
    import numpy as np

    return np.int64(value)

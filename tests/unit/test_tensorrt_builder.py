"""build_engine precision recipes, exercised with a fake tensorrt module (no GPU)."""

import sys
import types

import pytest

from fast_omniasr import tensorrt_builder

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

    class Config:
        def __init__(self):
            self.flags = set()
            built["config"] = self

        def set_flag(self, flag):
            self.flags.add(flag)

        def clear_flag(self, flag):
            self.flags.discard(flag)

        def set_memory_pool_limit(self, pool, size):
            pass

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
            return b"engine"

    class Parser:
        def __init__(self, network, logger):
            pass

        def parse_from_file(self, path):
            return True

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
    with pytest.raises(ValueError, match="precision='mixed' found none"):
        _build(tmp_path, "mixed")
    assert not (tmp_path / "out.engine").exists()


def test_unknown_precision_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="'fp32', 'fp16' or 'mixed'"):
        _build(tmp_path, "int8")

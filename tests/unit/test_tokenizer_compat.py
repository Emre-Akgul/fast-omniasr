"""A CTC graph must be loaded with the tokenizer it was exported with (issue #8)."""

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import onnx_bytes as ob
import pytest

from fast_omniasr import model as model_module
from fast_omniasr._onnx_deps import model_metadata
from fast_omniasr.tokenizer import (
    TOKENIZER_SHA256_KEY,
    VOCABULARY_SIZE_KEY,
    Tokenizer,
    check_compatible,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tokenizers(tmp_path_factory):
    """Two different SentencePiece models with the same vocabulary size, and one larger."""
    import sentencepiece as spm

    root = tmp_path_factory.mktemp("spm")
    corpus = root / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog\n"
                      "pack my box with five dozen liquor jugs\n" * 50)
    paths = {}
    for name, size, kind in [("a", 36, "unigram"), ("b", 36, "bpe"), ("big", 42, "unigram")]:
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / name),
                                       vocab_size=size, model_type=kind, minloglevel=2)
        paths[name] = root / f"{name}.model"
    assert Tokenizer(paths["a"]).vocabulary_size == Tokenizer(paths["b"]).vocabulary_size
    return paths


def _graph(tmp_path, metadata=None):
    path = tmp_path / "model.onnx"
    path.write_bytes(ob.model(ob.graph([ob.tensor("w", raw=b"\0" * 64)]), metadata))
    return path


def _stamped(tmp_path, tokenizer_path):
    tokenizer = Tokenizer(tokenizer_path)
    return _graph(tmp_path, {
        TOKENIZER_SHA256_KEY: hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        VOCABULARY_SIZE_KEY: str(tokenizer.vocabulary_size),
        "producer.note": "kept",
    })


def test_model_metadata_reads_metadata_props(tmp_path):
    path = _graph(tmp_path, {"a": "1", "ключ": "значение"})
    assert model_metadata(path) == {"a": "1", "ключ": "значение"}
    assert model_metadata(_graph(tmp_path)) == {}


def test_matching_pair_passes(tmp_path, tokenizers):
    tokenizer = Tokenizer(tokenizers["a"])
    check_compatible(tokenizer, vocabulary_size=tokenizer.vocabulary_size,
                     graph_path=_stamped(tmp_path, tokenizers["a"]))


def test_output_size_mismatch_is_rejected(tokenizers):
    tokenizer = Tokenizer(tokenizers["big"])
    with pytest.raises(ValueError, match="outputs 36 classes but the tokenizer big.model has"):
        check_compatible(tokenizer, vocabulary_size=36)


def test_same_size_but_different_tokenizer_is_rejected(tmp_path, tokenizers):
    graph = _stamped(tmp_path, tokenizers["a"])
    other = Tokenizer(tokenizers["b"])  # same vocabulary size: only the hash tells them apart
    with pytest.raises(ValueError, match="b.model is not the tokenizer model.onnx was exported"):
        check_compatible(other, vocabulary_size=other.vocabulary_size, graph_path=graph)


def test_recorded_vocabulary_size_is_checked(tmp_path, tokenizers):
    graph = _graph(tmp_path, {VOCABULARY_SIZE_KEY: "999"})
    with pytest.raises(ValueError, match="exported for a 999-entry tokenizer"):
        check_compatible(Tokenizer(tokenizers["a"]), graph_path=graph)


def test_graphs_without_metadata_get_the_size_check_only(tmp_path, tokenizers):
    tokenizer = Tokenizer(tokenizers["b"])
    check_compatible(tokenizer, vocabulary_size=tokenizer.vocabulary_size,
                     graph_path=_graph(tmp_path))


def test_omniasr_rejects_a_mismatched_tokenizer_and_closes_the_backend(
        tmp_path, tokenizers, monkeypatch):
    closed = []

    class Backend:
        def __init__(self, path, **kwargs):
            self.vocabulary_size = Tokenizer(tokenizers["a"]).vocabulary_size

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "fast_omniasr.backends.onnx",
                        types.SimpleNamespace(ONNXBackend=Backend))
    graph = _stamped(tmp_path, tokenizers["a"])
    with pytest.raises(ValueError, match="not the tokenizer"):
        model_module.OmniASR(graph, tokenizers["b"])
    assert closed == [True]
    model_module.OmniASR(graph, tokenizers["a"])  # the right tokenizer loads


def test_tensorrt_from_pretrained_checks_before_building(tmp_path, tokenizers, monkeypatch):
    from fast_omniasr import hub, tensorrt_cache

    graph = _stamped(tmp_path, tokenizers["a"])
    monkeypatch.setattr(hub, "download_verified", lambda *a, **k: (
        {}, {"model.onnx": graph, "tokenizer.model": tokenizers["b"]}))
    builds = []
    monkeypatch.setattr(tensorrt_cache, "get_or_build_engine", lambda *a, **k: builds.append(1))
    with pytest.raises(ValueError, match="not the tokenizer"):
        model_module.OmniASR.from_pretrained("owner/repo", backend="tensorrt", precision="fp32")
    assert builds == []


def _stamp_tool():
    spec = importlib.util.spec_from_file_location("stamp_tokenizer",
                                                  ROOT / "tools" / "stamp_tokenizer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stamp_tool_writes_metadata_and_keeps_external_data(tmp_path, tokenizers):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    size = Tokenizer(tokenizers["a"]).vocabulary_size
    weight = numpy_helper.from_array(np.ones((4, size), np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["audio", "w"], ["logits"])], "g",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, size])], [weight])
    path = tmp_path / "model.onnx"
    model = helper.make_model(graph)
    model.metadata_props.add(key="producer.note", value="kept")
    onnx.save_model(model, path, save_as_external_data=True, location="model.onnx.data",
                    size_threshold=0)
    data = (tmp_path / "model.onnx.data").read_bytes()

    tool = _stamp_tool()
    tool.stamp(path, tokenizers["a"])
    tool.stamp(path, tokenizers["a"])  # idempotent: keys replaced, not duplicated
    metadata = model_metadata(path)
    assert metadata["producer.note"] == "kept"
    assert metadata[VOCABULARY_SIZE_KEY] == str(size)
    assert metadata[TOKENIZER_SHA256_KEY] == hashlib.sha256(
        tokenizers["a"].read_bytes()).hexdigest()
    assert len(onnx.load(str(path), load_external_data=False).metadata_props) == 3
    assert (tmp_path / "model.onnx.data").read_bytes() == data
    check_compatible(Tokenizer(tokenizers["a"]), graph_path=path)
    with pytest.raises(ValueError, match="outputs .* classes but the tokenizer has"):
        tool.stamp(path, tokenizers["big"])

"""OmniASRLLM validates its artifact on load (issue #7)."""

import json
import sys
import types

import numpy as np
import onnx_bytes as ob
import pytest

from fast_omniasr._onnx_deps import graph_signature
from fast_omniasr.llm import OmniASRLLM
from fast_omniasr.llm.validation import InvalidArtifactError

HIDDEN, HEADS, HEAD_DIM, LAYERS = 8, 2, 4, 12
CACHE = [f"past_{kind}_{i}" for i in range(LAYERS) for kind in ("key", "value")]
PRESENT = [name.replace("past_", "present_") for name in CACHE]


@pytest.fixture(scope="module")
def tokenizer_file(tmp_path_factory):
    import sentencepiece as spm

    root = tmp_path_factory.mktemp("spm")
    corpus = root / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog\n" * 50)
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(root / "t"),
                                   vocab_size=30, minloglevel=2)
    return root / "t.model"


def _encoder(audio="audio", context_dims=(1, "p", HIDDEN)):
    return ob.signature_model([(audio, "float", (1, "samples"))],
                              [("context", "float", context_dims)])


def _decoder(vocab, *, tokens_type="int64", hidden=HIDDEN, heads=HEADS, drop=None,
             names=None):
    cache = [n for n in (names or CACHE) if n != drop]
    inputs = [("context", "float", (1, "c", hidden)), ("tokens", tokens_type, (1, "n")),
              *[(n, "float", (1, "t", heads, HEAD_DIM)) for n in cache]]
    outputs = [("logits", "float", (1, vocab)),
               *[(n, "float", (1, "t", heads, HEAD_DIM)) for n in PRESENT]]
    return ob.signature_model(inputs, outputs)


@pytest.fixture
def artifact(tmp_path, tokenizer_file):
    """A minimal valid fp32 ONNX LLM artifact (graphs carry signatures only)."""
    import sentencepiece as spm

    vocab = spm.SentencePieceProcessor(model_file=str(tokenizer_file)).get_piece_size()
    (tmp_path / "tokenizer.model").write_bytes(tokenizer_file.read_bytes())
    config = {
        "format_version": 1, "model": "omniASR_LLM_300M_v2", "bos_idx": 0, "eos_idx": 2,
        "pad_idx": 1, "vocab_size": vocab, "max_seq_len": 8192, "lid_marker_idx": vocab,
        "onnx": {"layers": LAYERS, "kv_heads": HEADS, "head_dim": HEAD_DIM},
        "languages": {"mapping": "languages.json", "embeddings": "lang_embeddings.npy",
                      "count": 2},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "encoder.onnx").write_bytes(_encoder())
    (tmp_path / "decoder.onnx").write_bytes(_decoder(vocab))
    np.save(tmp_path / "lang_embeddings.npy", np.zeros((3, HIDDEN), np.float32))
    (tmp_path / "languages.json").write_text(json.dumps({"eng_latn": 1, "tur_latn": 2}))
    (tmp_path / "decoder.weights").write_bytes(b"\0" * 64)
    (tmp_path / "decoder_weights.json").write_text(json.dumps(
        {"w": {"dtype": "float32", "shape": [4, 4], "offset": 0}}))
    return tmp_path


@pytest.fixture
def loaded(monkeypatch):
    """Record backend construction instead of creating ONNX Runtime sessions."""
    created = []
    # A stand-in module: validation must not need onnxruntime (not installed in every CI job).
    monkeypatch.setitem(sys.modules, "fast_omniasr.llm.onnx_backend", types.SimpleNamespace(
        ONNXLLMBackend=lambda *args, **kwargs: created.append(args) or object()))
    return created


def _edit_config(artifact, **changes):
    config = json.loads((artifact / "config.json").read_text())
    for key, value in changes.items():
        section, _, field = key.partition("__")
        if field:
            config[section][field] = value
        elif value is None:
            config.pop(key)
        else:
            config[key] = value
    (artifact / "config.json").write_text(json.dumps(config))


def test_valid_artifact_loads(artifact, loaded):
    OmniASRLLM(artifact, backend="onnx")
    assert len(loaded) == 1


def test_signature_reader(artifact):
    inputs, outputs = graph_signature(artifact / "decoder.onnx")
    assert inputs[0] == ("context", "float", (1, "c", HIDDEN))
    assert inputs[1] == ("tokens", "int64", (1, "n"))
    assert outputs[0][0] == "logits" and len(outputs) == 1 + 2 * LAYERS


@pytest.mark.parametrize("changes, message", [
    ({"bos_idx": 5000}, r"'bos_idx' is 5000, outside \[0, "),
    ({"eos_idx": None}, "'eos_idx' must be an integer, got None"),
    ({"eos_idx": 0}, "'bos_idx' and 'eos_idx' must differ"),
    ({"pad_idx": -1}, "'pad_idx' is -1"),
    ({"vocab_size": True}, "'vocab_size' must be an integer"),
    ({"max_seq_len": 2}, "'max_seq_len' is 2, too small for a 30 s clip"),
    ({"max_seq_len": 1000}, "'max_seq_len' is 1000, too small for a 30 s clip"),
    ({"lid_marker_idx": -1}, "'lid_marker_idx' is -1"),
    ({"format_version": 2}, "unsupported format_version 2"),
    ({"model": "omniASR_CTC_300M"}, "unsupported model 'omniASR_CTC_300M'"),
    ({"onnx__layers": 24}, "onnx.layers is 24; this runtime supports 12-layer"),
    ({"onnx__kv_heads": 0}, "'kv_heads' is 0"),
    ({"languages__mapping": 3}, "languages.mapping must be a file name"),
])
def test_invalid_config_is_rejected_before_loading(artifact, loaded, changes, message):
    _edit_config(artifact, **changes)
    with pytest.raises(InvalidArtifactError, match=message):
        OmniASRLLM(artifact, backend="onnx")
    assert loaded == []


def test_invalid_json_names_the_file(artifact, loaded):
    (artifact / "config.json").write_text("{not json")
    with pytest.raises(InvalidArtifactError, match="config.json: not valid JSON"):
        OmniASRLLM(artifact, backend="onnx")


def test_tokenizer_must_match_vocab_size(artifact, loaded):
    _edit_config(artifact, vocab_size=999)
    with pytest.raises(InvalidArtifactError, match="tokenizer.model: it has .* entries but "
                                                   "config.json declares vocab_size 999"):
        OmniASRLLM(artifact, backend="onnx")


def _vocab(artifact):
    return json.loads((artifact / "config.json").read_text())["vocab_size"]


@pytest.mark.parametrize("graph, build, message", [
    ("encoder.onnx", lambda v: _encoder(audio="wav"), r"encoder.onnx: inputs .*missing \['audio'\]"),
    ("encoder.onnx", lambda v: _encoder(context_dims=(1, HIDDEN)), "'context' has rank 2"),
    ("decoder.onnx", lambda v: _decoder(v, drop="past_value_11"),
     r"decoder.onnx: inputs .*missing \['past_value_11'\]"),
    ("decoder.onnx", lambda v: _decoder(v, tokens_type="int32"),
     "input 'tokens' is int32, expected int64"),
    ("decoder.onnx", lambda v: _decoder(v + 1), "'logits' has vocabulary size"),
    ("decoder.onnx", lambda v: _decoder(v, hidden=16),
     "'context' has hidden size 16, expected 8"),
    ("decoder.onnx", lambda v: _decoder(v, heads=4), "has kv_heads 4, expected 2"),
    ("decoder.onnx", lambda v: _decoder(v, names=list(reversed(CACHE))), "wrong order"),
    ("decoder.onnx", lambda v: b"garbage", r"decoder.onnx: is not a valid ONNX model"),
    ("decoder.onnx", lambda v: _decoder(v, hidden="h"),
     r"input 'context' needs static sizes on axes \[2\]"),
    ("decoder.onnx", lambda v: _decoder(v, heads="heads"),
     r"input 'past_key_0' needs static sizes on axes \[2, 3\]"),
])
def test_graph_contracts_are_checked_on_load(artifact, loaded, graph, build, message):
    (artifact / graph).write_bytes(build(_vocab(artifact)))
    with pytest.raises(InvalidArtifactError, match=message):
        OmniASRLLM(artifact, backend="onnx")
    assert loaded == []


def test_missing_graph_is_named(artifact, loaded):
    (artifact / "encoder.onnx").unlink()
    with pytest.raises(InvalidArtifactError, match="encoder.onnx: the graph is missing"):
        OmniASRLLM(artifact, backend="onnx")


def test_legacy_prefill_step_layout_is_accepted(artifact, loaded):
    vocab = _vocab(artifact)
    (artifact / "decoder.onnx").unlink()
    cache_values = [(n, "float", (1, "t", HEADS, HEAD_DIM)) for n in CACHE]
    (artifact / "decoder_prefill.onnx").write_bytes(ob.signature_model(
        [("context", "float", (1, "c", HIDDEN)), ("bos", "int64", (1, 1))],
        [("logits", "float", (1, vocab)), *cache_values]))
    (artifact / "decoder_step.onnx").write_bytes(ob.signature_model(
        [("token", "int64", (1, 1)), *cache_values],
        [("logits", "float", (1, vocab)),
         *[(n, "float", (1, "t", HEADS, HEAD_DIM)) for n in PRESENT]]))
    OmniASRLLM(artifact, backend="onnx")
    assert len(loaded) == 1


def test_half_precision_variant_is_validated(artifact, loaded):
    (artifact / "fp16").mkdir()
    (artifact / "fp16" / "encoder.onnx").write_bytes(_encoder())
    (artifact / "fp16" / "decoder.onnx").write_bytes(_decoder(_vocab(artifact), heads=4))
    with pytest.raises(InvalidArtifactError, match="fp16/decoder.onnx: .*kv_heads 4"):
        OmniASRLLM(artifact, backend="onnx", precision="fp16")


@pytest.mark.parametrize("damage, message", [
    (lambda a: (a / "languages.json").write_text(json.dumps({"eng_latn": 1, "x": 7})),
     r"languages.json: 'x' maps to 7, not a row of the 3-row"),
    (lambda a: (a / "languages.json").write_text(json.dumps({"eng_latn": 1})),
     "maps 1 languages but config.json declares languages.count 2"),
    (lambda a: np.save(a / "lang_embeddings.npy", np.zeros((3, 5), np.float32)),
     "rows have 5 values but the decoder's hidden size is 8"),
    (lambda a: (a / "lang_embeddings.npy").write_bytes(b"nope"),
     "lang_embeddings.npy: cannot read the language embeddings"),
    (lambda a: (a / "languages.json").write_text("[]"),
     r"expected a non-empty \{language code: row\} object"),
])
def test_language_tables_are_checked_on_load(artifact, loaded, damage, message):
    damage(artifact)
    with pytest.raises(InvalidArtifactError, match=message):
        OmniASRLLM(artifact, backend="onnx")


@pytest.mark.parametrize("entry, message", [
    ({"dtype": "float32", "shape": [8, 8], "offset": 0}, "past the end of decoder.weights"),
    ({"dtype": "float32", "shape": [4], "offset": -4}, "negative offset"),
    ({"shape": [4], "offset": 0}, "needs dtype, shape and offset"),
])
def test_decoder_weight_manifest_is_checked(artifact, loaded, entry, message):
    (artifact / "decoder_weights.json").write_text(json.dumps({"w": entry}))
    with pytest.raises(InvalidArtifactError, match=message):
        OmniASRLLM(artifact, backend="onnx")


def test_invalid_artifact_error_is_a_value_error():
    assert issubclass(InvalidArtifactError, ValueError)


def test_lid_marker_may_index_past_the_output_vocabulary(artifact, loaded):
    # It indexes the input embedding table, which can exceed the output vocabulary.
    _edit_config(artifact, lid_marker_idx=_vocab(artifact) + 5)
    OmniASRLLM(artifact, backend="onnx")
    assert len(loaded) == 1


def test_outputs_without_declared_shapes_are_not_misreported(artifact, loaded):
    vocab = _vocab(artifact)
    inputs, outputs = graph_signature(artifact / "decoder.onnx")
    assert graph_signature(artifact / "decoder.onnx")[1][0][2] == (1, vocab)
    body = ob._field(2, "g")  # the same graph, with the logits output's shape dropped
    body += b"".join(ob._field(11, ob.value_info(*v)) for v in inputs)
    logits = ob._field(1, "logits") + ob._field(2, ob._field(1, ob._field(1, 1)))
    body += ob._field(12, logits)
    body += b"".join(ob._field(12, ob.value_info(*v)) for v in outputs[1:])
    (artifact / "decoder.onnx").write_bytes(ob.model(body))
    assert graph_signature(artifact / "decoder.onnx")[1][0] == ("logits", "float", None)
    OmniASRLLM(artifact, backend="onnx")

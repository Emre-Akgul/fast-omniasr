"""Load-time validation of exported LLM artifacts.

OmniASRLLM checks config.json, the tokenizer, the language tables and (for the ONNX
backend) every graph's input/output contract before any session is created, so a broken
or mismatched export fails on load with a message naming the file at fault instead of
deep inside the first transcription.
"""

import json
from pathlib import Path

import numpy as np

from .models import LLM_MODEL_CARDS

# The ONNX runtime feeds and reads the K/V cache of exactly this many decoder layers.
SUPPORTED_LAYERS = 12
_MIN_CONTEXT = 4  # bos + at least one generated token + upstream's 3-position margin


class InvalidArtifactError(ValueError):
    """An exported LLM artifact is missing, malformed or inconsistent."""


def _fail(where, message):
    raise InvalidArtifactError(f"Invalid LLM artifact {where}: {message}")


def _integer(config, key, where, *, low=0, high=None, required=True):
    value = config.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        _fail(where, f"'{key}' must be an integer, got {value!r}")
    if value < low or (high is not None and value >= high):
        bounds = f"[{low}, {high})" if high is not None else f">= {low}"
        _fail(where, f"'{key}' is {value}, outside {bounds}")
    return value


def validate_config(config: dict, directory: Path) -> None:
    """Check the fields the runtime relies on: format, model, token ids, vocabulary and
    context sizes, and (when present) the ONNX decoder layout and language tables."""
    where = directory / "config.json"
    if not isinstance(config, dict):
        _fail(where, "the top level must be a JSON object")
    if config.get("format_version") != 1:
        _fail(where, f"unsupported format_version {config.get('format_version')!r} "
                     "(this runtime reads format_version 1)")
    if config.get("model") not in LLM_MODEL_CARDS:
        _fail(where, f"unsupported model {config.get('model')!r}; expected one of "
                     f"{', '.join(LLM_MODEL_CARDS)}")
    vocab = _integer(config, "vocab_size", where, low=1)
    ids = {key: _integer(config, key, where, high=vocab) for key in ("bos_idx", "eos_idx")}
    for key in ("pad_idx", "lid_marker_idx"):
        ids[key] = _integer(config, key, where, high=vocab, required=False)
    if ids["bos_idx"] == ids["eos_idx"]:
        _fail(where, "'bos_idx' and 'eos_idx' must differ")
    _integer(config, "max_seq_len", where, low=_MIN_CONTEXT)
    onnx = config.get("onnx")
    if onnx is not None:
        if not isinstance(onnx, dict):
            _fail(where, "'onnx' must be an object")
        layers = _integer(onnx, "layers", where / "onnx", low=1)
        if layers != SUPPORTED_LAYERS:
            _fail(where, f"onnx.layers is {layers}; this runtime supports "
                         f"{SUPPORTED_LAYERS}-layer decoders")
        _integer(onnx, "kv_heads", where / "onnx", low=1)
        _integer(onnx, "head_dim", where / "onnx", low=1)
    languages = config.get("languages")
    if languages is not None:
        if not isinstance(languages, dict):
            _fail(where, "'languages' must be an object")
        for key in ("mapping", "embeddings"):
            if not isinstance(languages.get(key), str):
                _fail(where, f"languages.{key} must be a file name")
        _integer(languages, "count", where / "languages", low=1, required=False)


def validate_tokenizer(tokenizer, config: dict, directory: Path) -> None:
    if tokenizer.vocabulary_size != config["vocab_size"]:
        _fail(directory / "tokenizer.model",
              f"it has {tokenizer.vocabulary_size} entries but config.json declares "
              f"vocab_size {config['vocab_size']}")


def validate_languages(config: dict, directory: Path, hidden: int | None) -> None:
    """The language mapping must index rows of the embedding table, whose width must be
    the decoder's hidden size. Only the .npy header is read."""
    languages = config.get("languages")
    if languages is None:
        return
    mapping_path = directory / languages["mapping"]
    embeddings_path = directory / languages["embeddings"]
    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(mapping_path, f"cannot read the language mapping ({exc})")
    try:
        table = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        _fail(embeddings_path, f"cannot read the language embeddings ({exc})")
    if table.ndim != 2 or table.shape[0] < 1:
        _fail(embeddings_path, f"expected a [languages, hidden] table, got {table.shape}")
    if hidden is not None and table.shape[1] != hidden:
        _fail(embeddings_path, f"rows have {table.shape[1]} values but the decoder's "
                               f"hidden size is {hidden}")
    if not isinstance(mapping, dict) or not mapping:
        _fail(mapping_path, "expected a non-empty {language code: row} object")
    count = languages.get("count")  # mapped languages; the table may add a no-language row
    if count is not None and count != len(mapping):
        _fail(mapping_path, f"maps {len(mapping)} languages but config.json declares "
                            f"languages.count {count}")
    for code, row in mapping.items():
        if not isinstance(row, int) or isinstance(row, bool) or not 0 <= row < table.shape[0]:
            _fail(mapping_path, f"{code!r} maps to {row!r}, not a row of the "
                                f"{table.shape[0]}-row embedding table")


def _check_io(path, values, expected, kind):
    """`expected` is [(name, element type, rank)]; names and order must match exactly."""
    names = [name for name, _, _ in values]
    wanted = [name for name, _, _ in expected]
    if names != wanted:
        missing = [n for n in wanted if n not in names]
        extra = [n for n in names if n not in wanted]
        detail = (f"missing {missing}" if missing else "") + (
            f"{'; ' if missing else ''}unexpected {extra}" if extra else "")
        _fail(path, f"{kind}s do not match the runtime's contract "
                    f"({detail or 'wrong order'})")
    for (name, element, dims), (_, want_element, rank) in zip(values, expected):
        if element != want_element:
            _fail(path, f"{kind} {name!r} is {element}, expected {want_element}")
        if len(dims) != rank:
            _fail(path, f"{kind} {name!r} has rank {len(dims)}, expected {rank}")


def _fixed(path, name, dims, axis, expected, what, source):
    value = dims[axis]
    if isinstance(value, int) and value != expected:
        _fail(path, f"{name!r} has {what} {value}, expected {expected} ({source})")


def validate_graphs(directory: Path, precision: str, config: dict) -> int:
    """Check the encoder and decoder graphs the ONNX backend will load against the
    runtime's input/output contract and config.json; return the decoder hidden size."""
    from .._onnx_deps import graph_signature

    graphs = directory if precision == "fp32" else directory / precision
    layout = config.get("onnx") or {}
    layers = layout.get("layers", SUPPORTED_LAYERS)
    heads, head_dim = layout.get("kv_heads"), layout.get("head_dim")
    vocab = config["vocab_size"]
    cache = [f"past_{kind}_{i}" for i in range(layers) for kind in ("key", "value")]
    present = [name.replace("past_", "present_") for name in cache]

    def signature(path):
        if not path.is_file():
            _fail(path, "the graph is missing")
        try:
            return graph_signature(path)
        except ValueError as exc:
            _fail(path, str(exc))

    encoder = graphs / "encoder.onnx"
    inputs, outputs = signature(encoder)
    _check_io(encoder, inputs, [("audio", "float", 2)], "input")
    _check_io(encoder, outputs, [("context", "float", 3)], "output")
    hidden = outputs[0][2][2]

    unified = graphs / "decoder.onnx"
    if unified.is_file() or precision != "fp32":
        decoders = {unified: [("context", "float", 3), ("tokens", "int64", 2)]}
    else:
        decoders = {graphs / "decoder_prefill.onnx": [("context", "float", 3),
                                                      ("bos", "int64", 2)],
                    graphs / "decoder_step.onnx": [("token", "int64", 2)]}
    for path, leading in decoders.items():
        inputs, outputs = signature(path)
        takes_cache = path.name != "decoder_prefill.onnx"
        _check_io(path, inputs, leading + ([(n, "float", 4) for n in cache]
                                           if takes_cache else []), "input")
        # The prefill graph names its cache outputs past_* (the step graph's inputs).
        produced = present if takes_cache else cache
        _check_io(path, outputs, [("logits", "float", 2)] + [(n, "float", 4) for n in produced],
                  "output")
        _fixed(path, "logits", outputs[0][2], 1, vocab, "vocabulary size", "config vocab_size")
        if leading[0][0] == "context":
            dims = inputs[0][2]
            if isinstance(hidden, int):
                _fixed(path, "context", dims, 2, hidden, "hidden size", "the encoder's output")
            elif isinstance(dims[2], int):
                hidden = dims[2]
        for name, _, dims in [*(inputs if takes_cache else []), *outputs[1:]]:
            if name.startswith(("past_", "present_")):
                if heads is not None:
                    _fixed(path, name, dims, 2, heads, "kv_heads", "config onnx.kv_heads")
                if head_dim is not None:
                    _fixed(path, name, dims, 3, head_dim, "head_dim", "config onnx.head_dim")
    if precision == "fp32":
        _validate_weight_manifest(directory)
    return hidden if isinstance(hidden, int) else None


def _validate_weight_manifest(directory: Path) -> None:
    """decoder_weights.json entries must describe arrays inside decoder.weights."""
    manifest = directory / "decoder_weights.json"
    weights = directory / "decoder.weights"
    if not manifest.is_file() or not weights.is_file():
        return  # the backend reports missing files itself
    try:
        entries = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(manifest, f"cannot read the weight manifest ({exc})")
    if not isinstance(entries, dict) or not entries:
        _fail(manifest, "expected a non-empty {initializer: entry} object")
    size = weights.stat().st_size
    for name, entry in entries.items():
        try:
            dtype = np.dtype(entry["dtype"])
            shape = tuple(int(d) for d in entry["shape"])
            offset = int(entry["offset"])
        except (KeyError, TypeError, ValueError):
            _fail(manifest, f"{name!r} needs dtype, shape and offset")
        if offset < 0 or any(d < 0 for d in shape):
            _fail(manifest, f"{name!r} has a negative offset or dimension")
        end = offset + dtype.itemsize * int(np.prod(shape, dtype=np.int64))
        if end > size:
            _fail(manifest, f"{name!r} ends at byte {end}, past the end of "
                            f"decoder.weights ({size} bytes)")

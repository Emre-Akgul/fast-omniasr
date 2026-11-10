"""Look up or build a locally-cached TensorRT engine for a given ONNX model + config.

TensorRT engines are not portable across GPUs, TensorRT versions or CUDA versions, so the
cache key covers everything that can invalidate a previously-built engine: the ONNX content
(the graph file plus every external-data weights file it references), the precision and
optimization profile requested, the build environment (see `_local_environment`), and this
module's own build-recipe version.
"""
import hashlib
import json
import mmap
import os
import re
import warnings
from importlib import metadata
from pathlib import Path

from ._locking import file_lock
from .tensorrt_builder import (
    BUILDER_CONFIG_VERSION,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_OPT_SAMPLES,
    MIXED_RECIPE,
    PRECISIONS,
    TensorRTBuildOptions,
    build_engine,
    check_build_options,
    validate_profile,
)

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "fast-omniasr" / "tensorrt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Protobuf field numbers from onnx.proto3 that can (transitively) hold a TensorProto.
_MODEL_GRAPH, _MODEL_FUNCTIONS = 7, 25
_GRAPH_NODE, _GRAPH_INITIALIZER, _GRAPH_SPARSE_INITIALIZER = 1, 5, 15
_FUNCTION_NODE, _FUNCTION_ATTRIBUTE_PROTO = 7, 11
_NODE_ATTRIBUTE = 5
_ATTR_T, _ATTR_G, _ATTR_TENSORS, _ATTR_GRAPHS = 5, 6, 10, 11
_ATTR_SPARSE_TENSOR, _ATTR_SPARSE_TENSORS = 22, 23
_SPARSE_VALUES, _SPARSE_INDICES = 1, 2
_TENSOR_EXTERNAL_DATA = 13
_ENTRY_KEY, _ENTRY_VALUE = 1, 2


def _varint(buf, pos: int, end: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if pos >= end:
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _length_delimited_fields(buf, span: tuple[int, int], wanted: frozenset):
    """Yield (field_number, (start, end)) for the wanted length-delimited fields of the
    protobuf message at buf[span], skipping everything else (including large raw tensor
    data). Works on offsets so a memory-mapped file is never copied."""
    pos, end = span
    while pos < end:
        tag, pos = _varint(buf, pos, end)
        field, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            _, pos = _varint(buf, pos, end)
        elif wire_type == 1:
            pos += 8
        elif wire_type == 5:
            pos += 4
        elif wire_type == 2:
            length, pos = _varint(buf, pos, end)
            if pos + length > end:
                raise ValueError("truncated length-delimited field")
            if field in wanted:
                yield field, (pos, pos + length)
            pos += length
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
    if pos != end:
        raise ValueError("truncated protobuf message")


def _tensor_locations(buf, tensor, out: set) -> None:
    for _, entry in _length_delimited_fields(buf, tensor, frozenset({_TENSOR_EXTERNAL_DATA})):
        fields = {
            field: bytes(buf[start:stop])
            for field, (start, stop) in _length_delimited_fields(
                buf, entry, frozenset({_ENTRY_KEY, _ENTRY_VALUE})
            )
        }
        if fields.get(_ENTRY_KEY) == b"location":
            out.add(fields.get(_ENTRY_VALUE, b"").decode())


def _sparse_locations(buf, sparse, out: set) -> None:
    wanted = frozenset({_SPARSE_VALUES, _SPARSE_INDICES})
    for _, tensor in _length_delimited_fields(buf, sparse, wanted):
        _tensor_locations(buf, tensor, out)


def _attribute_locations(buf, attribute, out: set) -> None:
    wanted = frozenset({_ATTR_T, _ATTR_G, _ATTR_TENSORS, _ATTR_GRAPHS,
                        _ATTR_SPARSE_TENSOR, _ATTR_SPARSE_TENSORS})
    for field, value in _length_delimited_fields(buf, attribute, wanted):
        if field in (_ATTR_T, _ATTR_TENSORS):
            _tensor_locations(buf, value, out)
        elif field in (_ATTR_G, _ATTR_GRAPHS):
            _graph_locations(buf, value, out)
        else:
            _sparse_locations(buf, value, out)


def _node_locations(buf, node, out: set) -> None:
    for _, attribute in _length_delimited_fields(buf, node, frozenset({_NODE_ATTRIBUTE})):
        _attribute_locations(buf, attribute, out)


def _function_locations(buf, function, out: set) -> None:
    # Nodes, plus default attribute values (FunctionProto.attribute_proto).
    wanted = frozenset({_FUNCTION_NODE, _FUNCTION_ATTRIBUTE_PROTO})
    for field, value in _length_delimited_fields(buf, function, wanted):
        if field == _FUNCTION_NODE:
            _node_locations(buf, value, out)
        else:
            _attribute_locations(buf, value, out)


def _graph_locations(buf, graph, out: set) -> None:
    wanted = frozenset({_GRAPH_NODE, _GRAPH_INITIALIZER, _GRAPH_SPARSE_INITIALIZER})
    for field, value in _length_delimited_fields(buf, graph, wanted):
        if field == _GRAPH_INITIALIZER:
            _tensor_locations(buf, value, out)
        elif field == _GRAPH_SPARSE_INITIALIZER:
            _sparse_locations(buf, value, out)
        else:
            _node_locations(buf, value, out)


def _external_data_locations(onnx_path: Path) -> list[str]:
    """Every external-data `location` referenced by an ONNX model, without needing the
    `onnx` package (it is not a runtime dependency) and without loading tensor data."""
    if onnx_path.stat().st_size == 0:
        raise ValueError(f"{onnx_path} is not a valid ONNX model: empty file")
    locations: set[str] = set()
    with onnx_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        wanted = frozenset({_MODEL_GRAPH, _MODEL_FUNCTIONS})
        try:
            for field, value in _length_delimited_fields(mm, (0, len(mm)), wanted):
                if field == _MODEL_GRAPH:
                    _graph_locations(mm, value, locations)
                else:
                    _function_locations(mm, value, locations)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            # RecursionError: pathologically nested subgraphs (protobuf/onnx cap at 100).
            raise ValueError(f"{onnx_path} is not a valid ONNX model: {exc}") from exc
    return sorted(locations)


def _external_data_fingerprint(onnx_path: Path) -> dict[str, str]:
    """sha256 of every external-data file the graph references, keyed by its location.
    Resolved like ONNX/TensorRT do: relative to the graph's directory."""
    fingerprint = {}
    for location in _external_data_locations(onnx_path):
        data_path = onnx_path.parent / location
        if not data_path.is_file():
            raise FileNotFoundError(
                f"{onnx_path.name} references external data {location!r}, "
                f"which is missing from {onnx_path.parent}"
            )
        fingerprint[location] = _sha256_file(data_path)
    return fingerprint


_TENSORRT_WHEEL = re.compile(r"tensorrt[-_]cu(\d+)(?:[-_].*)?")


def _tensorrt_cuda_flavor() -> str | None:
    """The CUDA major TensorRT itself was built for, e.g. "cu13" from the installed
    `tensorrt_cu13` / `tensorrt_cu13_libs` wheel. TensorRT links the CUDA runtime
    statically, so this -- not the cuda-python version -- is the CUDA its engines use.
    None when TensorRT wasn't installed from a pip wheel (tarball, system package)."""
    try:
        distributions = metadata.packages_distributions()
    except Exception:  # noqa: BLE001 -- metadata is best effort; caller falls back
        return None
    for package in ("tensorrt_libs", "tensorrt"):
        for name in distributions.get(package, []):
            match = _TENSORRT_WHEEL.fullmatch(name.lower())
            if match:
                return f"cu{match.group(1)}"
    return None


def _local_environment() -> dict:
    """Environment fields that decide whether a cached engine can be reused.

    Queried for the *current* CUDA device -- the one TensorRT builds and deserializes
    engines on in this thread (`OmniASR.from_pretrained(device_index=...)` selects it for
    the build, cudaSetDevice or CUDA_VISIBLE_DEVICES otherwise) -- not device 0.

    In the key:
      - `tensorrt_version`
      - `tensorrt_cuda`: the CUDA major TensorRT was built for, from its pip wheel name
        ("cu12", "cu13"). Without wheel metadata it falls back to the CUDA major of the
        cuda-python runtime ("cuda-python-13"); minor versions are deliberately left out
        so upgrading cuda-python doesn't rebuild every engine.
      - the GPU's `compute_capability` and `device_name`.
    Not in the key: the device ordinal, so identical GPUs share engines, and the driver
    version, since newer drivers run existing engines (a too-old driver fails loudly).
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "Install fast-omniasr[tensorrt] and a compatible TensorRT installation"
        ) from exc
    try:
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[tensorrt] for the cuda-python dependency") from exc

    def query(what, result):
        error, value = result
        if int(error) != 0:
            raise RuntimeError(f"Failed to query the {what}: {error}")
        return value

    device = query("current CUDA device", cudart.cudaGetDevice())
    prop = query(f"properties of CUDA device {device}", cudart.cudaGetDeviceProperties(device))
    tensorrt_cuda = _tensorrt_cuda_flavor()
    if tensorrt_cuda is None:
        runtime_version = query("CUDA runtime version", cudart.cudaRuntimeGetVersion())
        tensorrt_cuda = f"cuda-python-{runtime_version // 1000}"
    name = prop.name.decode() if isinstance(prop.name, bytes) else prop.name
    return {
        "tensorrt_version": trt.__version__,
        "tensorrt_cuda": tensorrt_cuda,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "device_name": name.rstrip("\x00"),
    }


def get_or_build_engine(onnx_path: str | Path, *, precision: str,
                         min_samples: int = DEFAULT_MIN_SAMPLES,
                         opt_samples: int = DEFAULT_OPT_SAMPLES,
                         max_samples: int = DEFAULT_MAX_SAMPLES,
                         options: TensorRTBuildOptions | None = None,
                         cache_dir: str | Path | None = None,
                         rebuild: bool = False,
                         discard_sha256: str | None = None) -> Path:
    """Path of a cached engine for this ONNX model and config, building it if needed.

    The engine's size and sha256 are recorded beside the cache metadata (not in the key)
    and checked before reuse (size first, then a full hash: about a second per GB); a
    corrupted or replaced engine is rebuilt with a warning. `rebuild=True` always
    rebuilds. `discard_sha256` rebuilds only if the cached engine is still that one, so
    when several processes fail to load the same engine, the first rebuild is reused by
    the others instead of being thrown away.
    """
    if precision not in PRECISIONS:
        raise ValueError("precision must be 'fp32', 'fp16' or 'mixed'")
    min_samples, opt_samples, max_samples = validate_profile(
        min_samples, opt_samples, max_samples
    )
    options = check_build_options(options)
    onnx_path = Path(onnx_path)
    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_ROOT
    metadata = {
        "source_sha256": _sha256_file(onnx_path),
        "external_data_sha256": _external_data_fingerprint(onnx_path),
        "precision": precision,
        **({"mixed_recipe": MIXED_RECIPE} if precision == "mixed" else {}),
        "profile": {"min": min_samples, "opt": opt_samples, "max": max_samples},
        **({"build_options": options.identity()} if options.identity() else {}),
        "tf32": False,
        "builder_config_version": BUILDER_CONFIG_VERSION,
        **_local_environment(),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    entry_dir = root / key
    engine_path = entry_dir / "model.engine"
    metadata_path = entry_dir / "metadata.json"

    with file_lock(root / f"{key}.lock"):
        if not rebuild and engine_path.exists() and metadata_path.exists():
            try:
                cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict):
                recorded = cached.pop("engine", None)
                discarded = (discard_sha256 is not None and isinstance(recorded, dict)
                             and recorded.get("sha256") == discard_sha256)
                if cached == metadata and not discarded:
                    if recorded is None:
                        # Cached before checksums were recorded: adopt on first use (an
                        # engine that then fails to deserialize is still rebuilt).
                        _write_metadata(metadata_path,
                                        {**metadata, "engine": _engine_record(engine_path)})
                        return engine_path
                    if _engine_matches(engine_path, recorded):
                        return engine_path
                    warnings.warn(f"Cached TensorRT engine {engine_path} does not match its "
                                  "recorded checksum (corrupted or replaced); rebuilding",
                                  stacklevel=2)
        # A crash mid-build must not leave a record that vouches for a partial engine.
        metadata_path.unlink(missing_ok=True)
        build_engine(onnx_path, engine_path, precision=precision, min_samples=min_samples,
                     opt_samples=opt_samples, max_samples=max_samples, options=options)
        _write_metadata(metadata_path, {**metadata, "engine": _engine_record(engine_path)})
    return engine_path


def _engine_record(engine_path: Path) -> dict:
    return {"size": engine_path.stat().st_size, "sha256": _sha256_file(engine_path)}


def _engine_matches(engine_path: Path, recorded) -> bool:
    if not isinstance(recorded, dict) or recorded.get("size") != engine_path.stat().st_size:
        return False  # malformed record or truncated/extended file: no need to hash
    return recorded.get("sha256") == _sha256_file(engine_path)


def recorded_engine_sha256(engine_path: str | Path) -> str | None:
    """The sha256 the cache recorded for this engine, if any."""
    try:
        record = json.loads((Path(engine_path).parent / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    engine = record.get("engine") if isinstance(record, dict) else None
    return engine.get("sha256") if isinstance(engine, dict) else None


def _write_metadata(metadata_path: Path, record: dict) -> None:
    tmp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
    tmp_metadata_path.write_text(json.dumps(record, indent=2))
    os.replace(tmp_metadata_path, metadata_path)

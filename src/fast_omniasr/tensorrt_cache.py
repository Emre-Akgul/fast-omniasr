"""Look up or build a locally-cached TensorRT engine for a given ONNX model + config.

TensorRT engines are not portable across GPUs, TensorRT versions or CUDA versions, so the
cache key covers everything that can invalidate a previously-built engine: the ONNX content
(the graph file plus every external-data weights file it references), the precision and
optimization profile requested, the build environment (see `_local_environment`), and this
module's own build-recipe version.
"""
import contextlib
import hashlib
import json
import mmap
import os
from pathlib import Path

from .tensorrt_builder import (
    BUILDER_CONFIG_VERSION,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_OPT_SAMPLES,
    build_engine,
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


def _local_environment() -> dict:
    """Environment fields that decide whether a cached engine can be reused.

    Queried for the *current* CUDA device -- the one TensorRT builds and deserializes
    engines on (selectable with `cudaSetDevice` or CUDA_VISIBLE_DEVICES) -- not device 0.

    In the key: TensorRT version, CUDA runtime version (major.minor), and the GPU's
    compute capability and name. Deliberately not in the key: the device ordinal, so
    identical GPUs share engines, and the driver version, since newer drivers run
    existing engines and including it would force a rebuild on every driver update.
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
    runtime_version = query("CUDA runtime version", cudart.cudaRuntimeGetVersion())
    name = prop.name.decode() if isinstance(prop.name, bytes) else prop.name
    return {
        "tensorrt_version": trt.__version__,
        "cuda_runtime_version": f"{runtime_version // 1000}.{runtime_version % 1000 // 10}",
        "compute_capability": f"{prop.major}.{prop.minor}",
        "device_name": name.rstrip("\x00"),
    }


@contextlib.contextmanager
def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock_path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_or_build_engine(onnx_path: str | Path, *, precision: str,
                         min_samples: int = DEFAULT_MIN_SAMPLES,
                         opt_samples: int = DEFAULT_OPT_SAMPLES,
                         max_samples: int = DEFAULT_MAX_SAMPLES,
                         cache_dir: str | Path | None = None) -> Path:
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be 'fp32' or 'fp16'")
    min_samples, opt_samples, max_samples = validate_profile(
        min_samples, opt_samples, max_samples
    )
    onnx_path = Path(onnx_path)
    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_ROOT
    metadata = {
        "source_sha256": _sha256_file(onnx_path),
        "external_data_sha256": _external_data_fingerprint(onnx_path),
        "precision": precision,
        "profile": {"min": min_samples, "opt": opt_samples, "max": max_samples},
        "tf32": False,
        "builder_config_version": BUILDER_CONFIG_VERSION,
        **_local_environment(),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    entry_dir = root / key
    engine_path = entry_dir / "model.engine"
    metadata_path = entry_dir / "metadata.json"

    with _locked(root / f"{key}.lock"):
        if engine_path.exists() and metadata_path.exists():
            try:
                cached = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = None
            if cached == metadata:
                return engine_path
        build_engine(onnx_path, engine_path, precision=precision, min_samples=min_samples,
                     opt_samples=opt_samples, max_samples=max_samples)
        tmp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
        tmp_metadata_path.write_text(json.dumps(metadata, indent=2))
        os.replace(tmp_metadata_path, metadata_path)
    return engine_path

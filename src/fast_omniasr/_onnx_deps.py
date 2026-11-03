"""Read what an ONNX graph references (external-data files) and its metadata_props,
without the onnx package.

The graph is memory-mapped and its protobuf fields are walked directly; inline weight
bytes are skipped, never read, so multi-GB graphs are scanned in milliseconds. Every
place a tensor can live is visited: graph initializers, sparse initializers, node
attribute tensors, and subgraphs (If/Loop/Scan bodies) and functions recursively.
"""

import mmap
import posixpath
from pathlib import Path

# onnx.proto field numbers
_MODEL_GRAPH, _MODEL_METADATA, _MODEL_FUNCTIONS = 7, 14, 25
_FUNCTION_NODES, _FUNCTION_ATTRIBUTES = 7, 11  # node, attribute_proto (default values)
_GRAPH_NODES, _GRAPH_INITIALIZERS, _GRAPH_SPARSE_INITIALIZERS = 1, 5, 15
_NODE_ATTRIBUTES = 5
_ATTRIBUTE_TENSORS = (5, 10)  # t, tensors
_ATTRIBUTE_GRAPHS = (6, 11)  # g, graphs
_ATTRIBUTE_SPARSE = (22, 23)  # sparse_tensor, sparse_tensors
_SPARSE_TENSORS = (1, 2)  # values, indices
_TENSOR_EXTERNAL_DATA, _TENSOR_DATA_LOCATION = 13, 14
_EXTERNAL = 1


_MAX_DEPTH = 100  # protobuf's own nesting limit


class _Malformed(ValueError):
    pass


def _span(value):
    if not isinstance(value, tuple):
        raise _Malformed("a message field has the wrong wire type")
    return value


def _varint(buf, pos, end):
    result = shift = 0
    while True:
        if pos >= end or shift > 63:
            raise _Malformed("truncated or invalid varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf, start, end):
    """Yield (field, value) for varints and (field, (start, end)) for length-delimited
    fields; fixed-width fields are skipped."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos, end)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos, end)
            yield field, value
        elif wire == 2:
            length, pos = _varint(buf, pos, end)
            if pos + length > end:
                raise _Malformed("field runs past the end of its message")
            yield field, (pos, pos + length)
            pos += length
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            raise _Malformed(f"unsupported protobuf wire type {wire}")
    if pos != end:
        raise _Malformed("message runs past its end")


class _Walker:
    def __init__(self, buf):
        self.buf = buf
        self.locations = set()
        self.depth = 0

    def model(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field == _MODEL_GRAPH:
                self.graph(*_span(span))
            elif field == _MODEL_FUNCTIONS:
                for inner, value in _fields(self.buf, *_span(span)):
                    if inner == _FUNCTION_NODES:
                        self.node(*_span(value))
                    elif inner == _FUNCTION_ATTRIBUTES:
                        self.attribute(*_span(value))

    def graph(self, start, end):
        self.depth += 1
        if self.depth > _MAX_DEPTH:
            raise _Malformed("subgraphs are nested too deeply")
        for field, span in _fields(self.buf, start, end):
            if field == _GRAPH_NODES:
                self.node(*_span(span))
            elif field == _GRAPH_INITIALIZERS:
                self.tensor(*_span(span))
            elif field == _GRAPH_SPARSE_INITIALIZERS:
                self.sparse(*_span(span))
        self.depth -= 1

    def node(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field == _NODE_ATTRIBUTES:
                self.attribute(*_span(span))

    def attribute(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field in _ATTRIBUTE_TENSORS:
                self.tensor(*_span(span))
            elif field in _ATTRIBUTE_GRAPHS:
                self.graph(*_span(span))
            elif field in _ATTRIBUTE_SPARSE:
                self.sparse(*_span(span))

    def sparse(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field in _SPARSE_TENSORS:
                self.tensor(*_span(span))

    def tensor(self, start, end):
        external, location = False, None
        for field, value in _fields(self.buf, start, end):
            if field == _TENSOR_DATA_LOCATION:
                external = value == _EXTERNAL
            elif field == _TENSOR_EXTERNAL_DATA:
                entry = dict(_fields(self.buf, *_span(value)))
                key, val = entry.get(1), entry.get(2)
                if key and val and not (isinstance(key, tuple) and isinstance(val, tuple)):
                    raise _Malformed("an external-data entry has the wrong wire type")
                if key and bytes(self.buf[key[0]:key[1]]) == b"location" and val:
                    location = bytes(self.buf[val[0]:val[1]]).decode("utf-8")
        if external:
            if not location:
                raise _Malformed("an external tensor has no location")
            self.locations.add(location)


def external_data_locations(path: str | Path) -> set[str]:
    """The `location` strings of every external tensor in the ONNX graph at `path`."""
    path = Path(path)
    with path.open("rb") as f:
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty, not an ONNX model")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            walker = _Walker(buf)
            try:
                walker.model(0, len(buf))
            except (_Malformed, UnicodeDecodeError, TypeError) as exc:
                raise ValueError(f"{path} is not a valid ONNX model ({exc})") from exc
            return walker.locations


def graph_dependencies(graph_name: str, path: str | Path) -> set[str]:
    """External-data files of the graph `graph_name` (a repo-relative POSIX name, read
    from `path`), as repo-relative names.

    Like ONNX Runtime, a location must stay inside the graph's own directory: absolute
    paths, `..` out of that directory, and Windows-style paths (backslashes, drive
    letters, which ONNX Runtime on Linux would read literally) are rejected."""
    directory = posixpath.dirname(graph_name)
    names = set()
    for location in external_data_locations(path):
        relative = posixpath.normpath(location)
        if (not location or "\\" in location or location.startswith("/")
                or (len(location) > 1 and location[1] == ":")
                or relative == ".." or relative.startswith("../")):
            raise ValueError(f"{graph_name} has an unsafe external-data location: {location}")
        names.add(posixpath.join(directory, relative) if directory else relative)
    return names


def model_metadata(path: str | Path) -> dict[str, str]:
    """The model's metadata_props as {key: value}."""
    path = Path(path)
    with path.open("rb") as f:
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty, not an ONNX model")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            metadata = {}
            try:
                for field, span in _fields(buf, 0, len(buf)):
                    if field == _MODEL_METADATA:
                        entry = dict(_fields(buf, *_span(span)))
                        key, value = _span(entry.get(1, (0, 0))), _span(entry.get(2, (0, 0)))
                        metadata[bytes(buf[key[0]:key[1]]).decode("utf-8")] = bytes(
                            buf[value[0]:value[1]]).decode("utf-8")
            except (_Malformed, UnicodeDecodeError) as exc:
                raise ValueError(f"{path} is not a valid ONNX model ({exc})") from exc
            return metadata


# GraphProto input/output -> ValueInfoProto(name, type) -> TypeProto.tensor_type ->
# (elem_type, shape) -> TensorShapeProto.dim -> (dim_value | dim_param)
_GRAPH_INPUTS, _GRAPH_OUTPUTS = 11, 12
_VALUE_NAME, _VALUE_TYPE = 1, 2
_TYPE_TENSOR = 1
_TENSOR_ELEM_TYPE, _TENSOR_SHAPE = 1, 2
_SHAPE_DIM = 1
_DIM_VALUE, _DIM_PARAM = 1, 2

ELEMENT_TYPES = {1: "float", 6: "int32", 7: "int64", 10: "float16", 16: "bfloat16"}


def _text(buf, span):
    start, end = _span(span)
    return bytes(buf[start:end]).decode("utf-8")


def _value_info(buf, start, end):
    """(name, element type name, dims) with dims as ints or symbolic strings (None when
    neither is given); dims is None when the graph declares no shape, and the type is
    None for non-tensor values."""
    name, element, dims = "", None, None
    for field, value in _fields(buf, start, end):
        if field == _VALUE_NAME:
            name = _text(buf, value)
        elif field == _VALUE_TYPE:
            for inner, tensor in _fields(buf, *_span(value)):
                if inner != _TYPE_TENSOR:
                    continue
                for key, item in _fields(buf, *_span(tensor)):
                    if key == _TENSOR_ELEM_TYPE:
                        element = ELEMENT_TYPES.get(item, f"type{item}")
                    elif key == _TENSOR_SHAPE:
                        dims = []
                        for kind, dim in _fields(buf, *_span(item)):
                            if kind != _SHAPE_DIM:
                                continue
                            size = None
                            for part, raw in _fields(buf, *_span(dim)):
                                if part == _DIM_VALUE:
                                    size = raw
                                elif part == _DIM_PARAM:
                                    size = _text(buf, raw)
                            dims.append(size)
    return name, element, None if dims is None else tuple(dims)


def graph_signature(path: str | Path):
    """The main graph's inputs and outputs as lists of (name, element type, dims), read
    without the onnx package or an inference session."""
    path = Path(path)
    with path.open("rb") as f:
        if path.stat().st_size == 0:
            raise ValueError(f"{path} is empty, not an ONNX model")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            inputs, outputs, initializers = [], [], set()
            try:
                for field, span in _fields(buf, 0, len(buf)):
                    if field != _MODEL_GRAPH:
                        continue
                    for inner, value in _fields(buf, *_span(span)):
                        if inner == _GRAPH_INPUTS:
                            inputs.append(_value_info(buf, *_span(value)))
                        elif inner == _GRAPH_OUTPUTS:
                            outputs.append(_value_info(buf, *_span(value)))
                        elif inner == _GRAPH_INITIALIZERS:
                            for key, name in _fields(buf, *_span(value)):
                                if key == 8:  # TensorProto.name
                                    initializers.add(_text(buf, name))
            except (_Malformed, UnicodeDecodeError) as exc:
                raise ValueError(f"{path} is not a valid ONNX model ({exc})") from exc
            # Initializers listed as graph inputs (older IRs) aren't runtime inputs.
            return [i for i in inputs if i[0] not in initializers], outputs

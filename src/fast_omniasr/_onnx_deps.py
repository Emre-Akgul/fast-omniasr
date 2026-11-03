"""Find the external-data files an ONNX graph references, without the onnx package.

The graph is memory-mapped and its protobuf fields are walked directly; inline weight
bytes are skipped, never read, so multi-GB graphs are scanned in milliseconds. Every
place a tensor can live is visited: graph initializers, sparse initializers, node
attribute tensors, and subgraphs (If/Loop/Scan bodies) and functions recursively.
"""

import mmap
import posixpath
from pathlib import Path

# onnx.proto field numbers
_MODEL_GRAPH, _MODEL_FUNCTIONS = 7, 25
_FUNCTION_NODES = 7
_GRAPH_NODES, _GRAPH_INITIALIZERS, _GRAPH_SPARSE_INITIALIZERS = 1, 5, 15
_NODE_ATTRIBUTES = 5
_ATTRIBUTE_TENSORS = (5, 10)  # t, tensors
_ATTRIBUTE_GRAPHS = (6, 11)  # g, graphs
_ATTRIBUTE_SPARSE = (22, 23)  # sparse_tensor, sparse_tensors
_SPARSE_TENSORS = (1, 2)  # values, indices
_TENSOR_EXTERNAL_DATA, _TENSOR_DATA_LOCATION = 13, 14
_EXTERNAL = 1


class _Malformed(ValueError):
    pass


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

    def model(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field == _MODEL_GRAPH:
                self.graph(*span)
            elif field == _MODEL_FUNCTIONS:
                for inner, node in _fields(self.buf, *span):
                    if inner == _FUNCTION_NODES:
                        self.node(*node)

    def graph(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field == _GRAPH_NODES:
                self.node(*span)
            elif field == _GRAPH_INITIALIZERS:
                self.tensor(*span)
            elif field == _GRAPH_SPARSE_INITIALIZERS:
                self.sparse(*span)

    def node(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field == _NODE_ATTRIBUTES:
                self.attribute(*span)

    def attribute(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field in _ATTRIBUTE_TENSORS:
                self.tensor(*span)
            elif field in _ATTRIBUTE_GRAPHS:
                self.graph(*span)
            elif field in _ATTRIBUTE_SPARSE:
                self.sparse(*span)

    def sparse(self, start, end):
        for field, span in _fields(self.buf, start, end):
            if field in _SPARSE_TENSORS:
                self.tensor(*span)

    def tensor(self, start, end):
        external, location = False, None
        for field, value in _fields(self.buf, start, end):
            if field == _TENSOR_DATA_LOCATION:
                external = value == _EXTERNAL
            elif field == _TENSOR_EXTERNAL_DATA:
                entry = dict(_fields(self.buf, *value))
                key, val = entry.get(1), entry.get(2)
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
            except (_Malformed, UnicodeDecodeError) as exc:
                raise ValueError(f"{path} is not a valid ONNX model ({exc})") from exc
            return walker.locations


def graph_dependencies(graph_name: str, path: str | Path) -> set[str]:
    """External-data files of the graph `graph_name` (a repo-relative POSIX name, read
    from `path`), as repo-relative names; locations that escape the repo are rejected."""
    directory = posixpath.dirname(graph_name)
    names = set()
    for location in external_data_locations(path):
        name = posixpath.normpath(posixpath.join(directory, location.replace("\\", "/")))
        if location.startswith(("/", "\\")) or name == ".." or name.startswith("../"):
            raise ValueError(f"{graph_name} has an unsafe external-data location: {location}")
        names.add(name)
    return names

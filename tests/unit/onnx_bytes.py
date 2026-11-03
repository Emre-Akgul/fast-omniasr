"""Build tiny ONNX model files by hand (no onnx package), for dependency-closure tests."""


def _varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, payload):
    if isinstance(payload, int):
        return _varint(number << 3) + _varint(payload)
    if isinstance(payload, str):
        payload = payload.encode()
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def tensor(name, location=None, raw=b""):
    """A TensorProto: inline `raw` bytes, or external data at `location`."""
    body = _field(1, 1) + _field(2, 1) + _field(8, name)  # dims, FLOAT, name
    if location is None:
        return body + _field(9, raw)
    entry = _field(1, "location") + _field(2, location)
    offset = _field(1, "offset") + _field(2, "0")
    return body + _field(13, entry) + _field(13, offset) + _field(14, 1)


def graph(initializers=(), nodes=(), sparse=()):
    body = b"".join(_field(1, node) for node in nodes)
    body += _field(2, "g") + b"".join(_field(5, t) for t in initializers)
    body += b"".join(_field(15, _field(1, values) + _field(2, indices))
                     for values, indices in sparse)
    return body


def constant_node(value_tensor):
    """A Constant node whose value attribute holds `value_tensor`."""
    attribute = _field(1, "value") + _field(5, value_tensor) + _field(20, 4)  # type TENSOR
    return _field(2, "out") + _field(4, "Constant") + _field(5, attribute)


def if_node(then_graph):
    """An If node whose then_branch subgraph is `then_graph`."""
    attribute = _field(1, "then_branch") + _field(6, then_graph) + _field(20, 5)  # GRAPH
    return _field(1, "cond") + _field(4, "If") + _field(5, attribute)


def model(graph_bytes, metadata=None):
    props = b"".join(_field(14, _field(1, key) + _field(2, value))
                     for key, value in (metadata or {}).items())
    return (_field(1, 8) + _field(2, "test") + _field(7, graph_bytes) + _field(8, _field(2, 17))
            + props)


def model_with_external(*locations, inline=b""):
    """A model whose initializers live at `locations` (plus one inline initializer)."""
    tensors = [tensor(f"w{i}", location) for i, location in enumerate(locations)]
    return model(graph([tensor("inline", raw=inline), *tensors]))


def graph_file(name, listed):
    """File content for a fake repo: a graph referencing `<name>.data` when that is listed
    too, otherwise the name itself as bytes."""
    if not name.endswith(".onnx"):
        return name.encode()
    data = name.rsplit("/", 1)[-1] + ".data"
    sibling = name + ".data"
    return model_with_external(*([data] if sibling in listed else []), inline=name.encode())

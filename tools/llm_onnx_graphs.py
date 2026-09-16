"""Lower the frozen cached decoder to standard ONNX ops and shared external weights.

No source-framework imports: the caller supplies the loaded Day-2 ScriptModule.
RoPE is lowered to real arithmetic; K/V layout and causal semantics are unchanged.
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper


class Weights:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.file = (self.directory / "decoder.weights").open("wb")
        self.entries = {}

    def add(self, name, tensor, transpose=False):
        if name not in self.entries:
            array = tensor.detach().cpu().numpy()
            if transpose:
                array = array.T
            array = np.ascontiguousarray(array, dtype=np.float32)
            padding = (-self.file.tell()) % 64
            self.file.write(b"\0" * padding)
            offset = self.file.tell()
            self.file.write(array.tobytes())
            self.entries[name] = {
                "shape": list(array.shape),
                "offset": offset,
                "length": array.nbytes,
                "dtype": "float32",
            }
        entry = self.entries[name]
        tensor = T(name=name, data_type=T.FLOAT, dims=entry["shape"], data_location=T.EXTERNAL)
        for key, value in {
            "location": "decoder.weights",
            "offset": entry["offset"],
            "length": entry["length"],
        }.items():
            tensor.external_data.add(key=key, value=str(value))
        return tensor

    def close(self):
        self.file.close()
        (self.directory / "decoder_weights.json").write_text(
            json.dumps(self.entries, indent=2) + "\n"
        )


class Graph:
    def __init__(self, weights):
        self.weights = weights
        self.nodes = []
        self.initializers = {}
        self.serial = 0

    def op(self, kind, *inputs, **attributes):
        name = f"{kind}_{self.serial}"
        self.serial += 1
        self.nodes.append(h.make_node(kind, list(inputs), [name], name=name, **attributes))
        return name

    def constant(self, value, dtype=np.int64):
        name = f"constant_{self.serial}"
        self.serial += 1
        self.initializers[name] = numpy_helper.from_array(np.asarray(value, dtype=dtype), name)
        return name

    def weight(self, name, value, transpose=False):
        self.initializers[name] = self.weights.add(name, value, transpose)
        return name

    def linear(self, x, module, name):
        out = self.op("MatMul", x, self.weight(name + ".weight", module.weight, True))
        bias = getattr(module, "bias", None)
        if bias is not None:
            out = self.op("Add", out, self.weight(name + ".bias", bias))
        return out

    def norm(self, x, module, name):
        nodes = [n for n in module.forward.inlined_graph.nodes() if n.kind() == "aten::rms_norm"]
        if len(nodes) != 1:
            raise ValueError("Expected the Day-2 RMSNorm operator")
        eps = list(nodes[0].inputs())[3].toIValue()
        square = self.op("Mul", x, x)
        mean = self.op("ReduceMean", square, self.constant([-1]), keepdims=1)
        inv = self.op(
            "Reciprocal", self.op("Sqrt", self.op("Add", mean, self.constant(eps, np.float32)))
        )
        return self.op("Mul", self.op("Mul", x, inv), self.weight(name + ".weight", module.weight))

    def rope(self, value, positions, frequencies):
        cos = self.op("Gather", self.weight("rope.cos", frequencies[..., 0]), positions, axis=0)
        sin = self.op("Gather", self.weight("rope.sin", frequencies[..., 1]), positions, axis=0)
        cos = self.op("Unsqueeze", cos, self.constant([0, 2]))
        sin = self.op("Unsqueeze", sin, self.constant([0, 2]))
        even = self.op(
            "Slice",
            value,
            self.constant([0]),
            self.constant([2**63 - 1]),
            self.constant([3]),
            self.constant([2]),
        )
        odd = self.op(
            "Slice",
            value,
            self.constant([1]),
            self.constant([2**63 - 1]),
            self.constant([3]),
            self.constant([2]),
        )
        real = self.op("Sub", self.op("Mul", even, cos), self.op("Mul", odd, sin))
        imaginary = self.op("Add", self.op("Mul", even, sin), self.op("Mul", odd, cos))
        pairs = self.op(
            "Concat",
            self.op("Unsqueeze", real, self.constant([-1])),
            self.op("Unsqueeze", imaginary, self.constant([-1])),
            axis=-1,
        )
        return self.op("Reshape", pairs, self.op("Shape", value))


def cache_names(prefix, layers=12):
    return [f"{prefix}_{kind}_{i}" for i in range(layers) for kind in ("key", "value")]


def export_decoder(model, directory):
    layers = list(model.layers.children())
    hidden = model.embedding.weight.shape[1]
    dim = layers[0].freqs.shape[1] * 2
    heads = layers[0].q.weight.shape[0] // dim
    if any(
        layer.k.weight.shape[0] != heads * dim or layer.v.weight.shape[0] != heads * dim
        for layer in layers
    ):
        raise ValueError("This ONNX lowering supports multi-head attention only")
    directory = Path(directory)
    weights = Weights(directory)
    try:
        for prefill in (True, False):
            g = Graph(weights)
            token = "bos" if prefill else "token"
            inputs = [h.make_tensor_value_info(token, T.INT64, [1, 1])]
            embedded = g.op(
                "Gather", g.weight("embedding.weight", model.embedding.weight), token, axis=0
            )
            if prefill:
                inputs.insert(
                    0, h.make_tensor_value_info("context", T.FLOAT, [1, "context_len", hidden])
                )
                x = g.op("Concat", "context", embedded, axis=1)
                offset = g.constant(0)
            else:
                x = embedded
                inputs.extend(
                    h.make_tensor_value_info(n, T.FLOAT, [1, "cached_positions", heads, dim])
                    for n in cache_names("past", len(layers))
                )
                offset = g.op("Gather", g.op("Shape", "past_key_0"), g.constant(1), axis=0)
            length = g.op("Gather", g.op("Shape", x), g.constant(1), axis=0)
            positions = g.op("Range", offset, g.op("Add", offset, length), g.constant(1))
            head_shape = g.constant([1, -1, heads, dim])
            outputs = []
            output_names = cache_names("past" if prefill else "present", len(layers))
            for i, layer in enumerate(layers):
                name = f"layers.{i}"
                frequencies = layer.freqs
                if not np.array_equal(frequencies.numpy(), layers[0].freqs.numpy()):
                    raise ValueError("Decoder layers must share RoPE frequencies")
                residual = x
                normalized = g.norm(x, layer.norm, name + ".norm")
                q = g.rope(
                    g.op("Reshape", g.linear(normalized, layer.q, name + ".q"), head_shape),
                    positions,
                    frequencies,
                )
                k = g.rope(
                    g.op("Reshape", g.linear(normalized, layer.k, name + ".k"), head_shape),
                    positions,
                    frequencies,
                )
                v = g.op("Reshape", g.linear(normalized, layer.v, name + ".v"), head_shape)
                if not prefill:
                    k = g.op("Concat", f"past_key_{i}", k, axis=1)
                    v = g.op("Concat", f"past_value_{i}", v, axis=1)
                for value, out_name in zip((k, v), output_names[2 * i : 2 * i + 2]):
                    g.nodes.append(h.make_node("Identity", [value], [out_name]))
                    outputs.append(
                        h.make_tensor_value_info(
                            out_name,
                            T.FLOAT,
                            [
                                1,
                                "prefill_positions" if prefill else "present_positions",
                                heads,
                                dim,
                            ],
                        )
                    )
                q = g.op("Transpose", q, perm=[0, 2, 1, 3])
                k = g.op("Transpose", k, perm=[0, 2, 3, 1])
                v = g.op("Transpose", v, perm=[0, 2, 1, 3])
                scores = g.op("Mul", g.op("MatMul", q, k), g.constant(dim**-0.5, np.float32))
                if prefill:
                    # True for a future key; all queries see their diagonal.
                    future = g.op(
                        "Greater",
                        g.op("Unsqueeze", positions, g.constant([0])),
                        g.op("Unsqueeze", positions, g.constant([1])),
                    )
                    scores = g.op("Where", future, g.constant(float("-inf"), np.float32), scores)
                attention = g.op("MatMul", g.op("Softmax", scores, axis=-1), v)
                attention = g.op(
                    "Reshape",
                    g.op("Transpose", attention, perm=[0, 2, 1, 3]),
                    g.constant([1, -1, hidden]),
                )
                x = g.op("Add", residual, g.linear(attention, layer.out, name + ".out"))
                ffn_input = g.norm(x, layer.ffn_norm, name + ".ffn_norm")
                gate = g.linear(ffn_input, layer.ffn.gate_proj, name + ".ffn.gate_proj")
                activated = g.op("Mul", gate, g.op("Sigmoid", gate))
                inner = g.linear(ffn_input, layer.ffn.inner_proj, name + ".ffn.inner_proj")
                x = g.op(
                    "Add",
                    x,
                    g.linear(
                        g.op("Mul", inner, activated),
                        layer.ffn.output_proj,
                        name + ".ffn.output_proj",
                    ),
                )
            x = g.norm(x, model.norm, "norm")
            x = g.op("Gather", x, g.constant(-1), axis=1)
            logits = g.linear(x, model.proj, "proj")
            g.nodes.append(h.make_node("Identity", [logits], ["logits"]))
            outputs.insert(
                0, h.make_tensor_value_info("logits", T.FLOAT, [1, model.proj.weight.shape[0]])
            )
            graph = h.make_graph(
                g.nodes,
                "decoder_prefill" if prefill else "decoder_step",
                inputs,
                outputs,
                list(g.initializers.values()),
            )
            artifact = h.make_model(graph, opset_imports=[h.make_opsetid("", 18)], ir_version=9)
            path = directory / f"{graph.name}.onnx"
            onnx.save_model(artifact, path)
            weights.file.flush()
            onnx.checker.check_model(str(path))
            print(f"Exported {path}", flush=True)
    finally:
        weights.close()

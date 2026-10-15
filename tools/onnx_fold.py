"""Fold runtime weight normalization into one float64-computed weight.

The wav2vec2 positional convolution is exported as weight_g * weight_v / ||weight_v||,
recomputed on every call. ONNX Runtime's float32 ReduceL2 over each 65,536-element norm
puts ~6e-6 relative error into the convolution weights, which made the 3B encoder 12-37x
less accurate than PyTorch float32 against a float64 reference. Folding computes the
weight once in float64; the result matches or beats PyTorch's own float32 accuracy.

Used by export_onnx.py (CTC) and export_llm_onnx.py (LLM encoders), and as a tool that
fixes an already exported graph in place, writing only the folded tensor and the graph:

  python tools/onnx_fold.py /path/to/model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, external_data_helper, numpy_helper


def _array(tensor, base_dir):
    return numpy_helper.to_array(tensor, base_dir=str(base_dir) if base_dir else "")


def fold_weight_norm(model, base_dir=None, external=False):
    """Fold every ReduceL2 -> Div -> Mul weight-norm pattern over constant tensors.

    With `base_dir`, tensors stored as external data are read from that directory. With
    `external=True` each folded weight is written there as its own external file (named
    after the tensor), so a large on-disk model needs no other weight rewritten.
    Returns the number of folded weights.
    """
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    constants = {
        node.output[0]: node.attribute[0].t
        for node in graph.node
        if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value"
    }
    consumers = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    def constant(name):
        tensor = initializers.get(name)
        if tensor is None:
            tensor = constants.get(name)
        return None if tensor is None else _array(tensor, base_dir)

    folded, removed = 0, set()
    for norm in [node for node in graph.node if node.op_type == "ReduceL2"]:
        v_name = norm.input[0]
        divs = [n for n in consumers.get(norm.output[0], []) if n.op_type == "Div"]
        if v_name not in initializers or len(divs) != 1 or divs[0].input[0] != v_name:
            continue
        muls = [n for n in consumers.get(divs[0].output[0], []) if n.op_type == "Mul"]
        if len(muls) != 1:
            continue
        mul = muls[0]
        g = constant(mul.input[1])
        if g is None:
            continue
        v = _array(initializers[v_name], base_dir).astype(np.float64)
        axes = tuple(int(axis) for axis in constant(norm.input[1]))
        weight = (g.astype(np.float64) * v / np.sqrt((v * v).sum(axis=axes, keepdims=True)))
        name = v_name.removesuffix("_v") + "_folded" if v_name.endswith("_v") else v_name + "_folded"
        tensor = numpy_helper.from_array(weight.astype(np.float32), name)
        if external:
            (Path(base_dir) / name).write_bytes(tensor.raw_data)
            external_data_helper.set_external_data(tensor, location=name)
            tensor.ClearField("raw_data")
            tensor.data_location = TensorProto.EXTERNAL
        graph.initializer.append(tensor)
        for node in graph.node:
            for i, value in enumerate(node.input):
                if value == mul.output[0]:
                    node.input[i] = name
        removed.update(id(node) for node in (mul, divs[0], norm))
        folded += 1
    kept = [node for node in graph.node if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(kept)
    used = {value for node in graph.node for value in node.input}
    for tensor in [t for t in graph.initializer if t.name not in used]:
        graph.initializer.remove(tensor)
    return folded


def fold_file(path, output=None):
    """Fold an exported graph on disk. Large models keep their external tensor files."""
    path = Path(path)
    output = Path(output or path)
    model = onnx.load(str(path), load_external_data=False)
    external = any(t.data_location == TensorProto.EXTERNAL for t in model.graph.initializer)
    if external and output.parent.resolve() != path.parent.resolve():
        raise ValueError("Write a graph with external data next to its data files")
    if not external:
        model = onnx.load(str(path))
    folded = fold_weight_norm(model, base_dir=path.parent, external=external)
    onnx.save_model(model, str(output))
    onnx.checker.check_model(str(output))
    return folded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, help="Default: rewrite the model in place")
    args = parser.parse_args()
    folded = fold_file(args.model, args.output)
    print(f"{args.model}: folded {folded} weight-normalized tensor(s)")
    if folded != 1:
        raise SystemExit("Expected exactly one weight-normalized positional convolution")


if __name__ == "__main__":
    main()

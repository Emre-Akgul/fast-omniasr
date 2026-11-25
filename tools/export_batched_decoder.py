"""Derive a batched decoder from a unified ONNX export, sharing its external weights."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def export(source, destination, half_cache=False):
    source, destination = Path(source), Path(destination)
    if destination.resolve() == source.resolve():
        raise ValueError("Choose a separate output file to preserve the original decoder")
    if destination.parent.resolve() != source.parent.resolve():
        raise ValueError("Write the graph beside its existing external weight files")
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    if [v.name for v in graph.input[:2]] != ["context", "tokens"]:
        raise ValueError("Expected a unified decoder")
    cache_inputs = [v for v in graph.input if v.name.startswith("past_")]
    layers, remainder = divmod(len(cache_inputs), 2)
    if not layers or remainder:
        raise ValueError("Expected key/value cache inputs for each decoder layer")
    causal_masks = sum(n.op_type == "Greater" for n in graph.node)
    cache_joins = sum(n.op_type == "Concat" and n.input[0].startswith("past_")
                      for n in graph.node)
    if causal_masks != layers or cache_joins != 2 * layers:
        raise ValueError("Unsupported attention mask or cache layout")
    for value in [*graph.input, *graph.output]:
        tensor = value.type.tensor_type
        tensor.shape.dim[0].dim_param = "batch"
        if half_cache and value.name.startswith(("past_", "present_")):
            tensor.elem_type = TensorProto.FLOAT16

    # A zero in an ONNX Reshape copies the corresponding input dimension.
    for tensor in graph.initializer:
        if tensor.external_data or tensor.data_type != TensorProto.INT64:
            continue
        shape = numpy_helper.to_array(tensor).copy()
        if shape.ndim == 1 and shape.size in (3, 4) and list(shape[:2]) == [1, -1]:
            shape[0] = 0
            tensor.CopyFrom(numpy_helper.from_array(shape, tensor.name))

    # Left-padding masks invalid keys. Keep the diagonal visible on padded
    # queries to prevent NaNs from propagating through later layers.
    graph.input.append(helper.make_tensor_value_info("key_padding", TensorProto.BOOL,
                                                    ["batch", 1, 1, "total_positions"]))
    graph.input.append(helper.make_tensor_value_info("position_ids", TensorProto.INT64,
                                                    ["batch", "new_positions"]))
    graph.initializer.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "head_axis"))
    rope_values = set()
    nodes = []
    for node in graph.node:
        if node.op_type == "Gather" and node.input[0] in ("rope.cos", "rope.sin"):
            node.input[1] = "position_ids"
            rope_values.add(node.output[0])
        elif node.op_type == "Unsqueeze" and node.input[0] in rope_values:
            node.input[1] = "head_axis"
        elif node.op_type == "Greater":
            future = node.output[0]
            key, query = node.input
            nodes.append(node)
            node.output[0] = future + "_causal"
            nodes.extend([
                helper.make_node("Equal", [key, query], [future + "_diagonal"]),
                helper.make_node("Not", [future + "_diagonal"], [future + "_offdiag"]),
                helper.make_node("And", ["key_padding", future + "_offdiag"], [future + "_pad"]),
                helper.make_node("Or", [future + "_causal", future + "_pad"], [future]),
            ])
            continue
        elif half_cache and node.op_type == "Concat" and node.input[0].startswith("past_"):
            # Concatenate in half precision so the growing cache copy is half the size.
            fresh, full = node.input[1], node.output[0]
            nodes.append(helper.make_node("Cast", [fresh], [fresh + "_half"], to=TensorProto.FLOAT16))
            node.input[1] = fresh + "_half"
            node.output[0] = full + "_half"
            nodes.extend([node, helper.make_node("Cast", [full + "_half"], [full],
                                                 to=TensorProto.FLOAT)])
            continue
        elif half_cache and node.op_type == "Identity" and node.output[0].startswith("present_"):
            node.input[0] += "_half"
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.ClearField("value_info")
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--half-cache", action="store_true")
    args = parser.parse_args()
    export(args.source, args.destination, args.half_cache)

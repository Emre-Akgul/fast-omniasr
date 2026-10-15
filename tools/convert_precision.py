"""Write fp16/bf16 variants of exported ONNX graphs next to the float32 ones.

Only MatMul/Gemm weights become half precision: each such op casts its activation in and
its result back to float32. Convolutions stay float32 (fp16 convolutions lost accuracy in
the CTC models, and ONNX allows bf16 Conv only from opset 22), as do norms, softmax,
residual paths, K/V caches and every graph input/output. The weights, which dominate
memory and per-token bandwidth, halve.

  python tools/convert_precision.py /path/to/llm-or-ctc-export --precision fp16 bf16

An LLM export (encoder.onnx + decoder.onnx) gets <dir>/<precision>/encoder.onnx and
decoder.onnx; a CTC export (model.onnx) gets <dir>/<precision>/model.onnx. Each graph
stores its weights in one `<graph>.data` file beside it.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HALF = {"fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}
CONVERTED_OPS = {"MatMul", "Gemm"}


def half_tensor(name, array, precision):
    array = np.asarray(array, np.float32)
    if precision == "fp16":
        return numpy_helper.from_array(array.astype(np.float16), name)
    bits = array.view(np.uint32)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)  # nearest even
    return helper.make_tensor(name, TensorProto.BFLOAT16, array.shape, rounded.tobytes(), raw=True)


def convert_model(model, precision):
    """Rewrite constant-weight MatMul/Gemm nodes in place; return how many changed."""
    half = HALF[precision]
    graph = model.graph
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    nodes, converted, replaced = [], 0, set()
    for node in graph.node:
        weight = node.input[1] if node.op_type in CONVERTED_OPS and len(node.input) > 1 else None
        if weight not in initializers or initializers[weight].data_type != TensorProto.FLOAT:
            nodes.append(node)
            continue
        name = node.name or node.output[0]
        cast_in = f"{name}/{precision}_input"
        nodes.append(helper.make_node("Cast", [node.input[0]], [cast_in], to=half, name=cast_in))
        inputs = [cast_in]
        for value in node.input[1:]:
            tensor = initializers.get(value)
            if tensor is not None and tensor.data_type == TensorProto.FLOAT:
                renamed = f"{value}/{precision}"
                if renamed not in initializers:
                    converted_tensor = half_tensor(renamed, numpy_helper.to_array(tensor), precision)
                    graph.initializer.append(converted_tensor)
                    initializers[renamed] = converted_tensor
                replaced.add(value)
                inputs.append(renamed)
            else:
                inputs.append(value)
        half_output = f"{name}/{precision}_output"
        rewritten = helper.make_node(node.op_type, inputs, [half_output], name=name)
        rewritten.attribute.extend(node.attribute)
        nodes.append(rewritten)
        nodes.append(
            helper.make_node(
                "Cast", [half_output], [node.output[0]], to=TensorProto.FLOAT,
                name=f"{name}/float_output",
            )
        )
        converted += 1
    del graph.node[:]
    graph.node.extend(nodes)
    used = {value for node in graph.node for value in node.input}
    for tensor in [t for t in graph.initializer if t.name in replaced and t.name not in used]:
        graph.initializer.remove(tensor)
    return converted


def convert_graph(source, destination, precision):
    source, destination = Path(source), Path(destination)
    model = onnx.load(str(source))
    converted = convert_model(model, precision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = f"{destination.name}.data"
    (destination.parent / data).unlink(missing_ok=True)
    # Small constants stay inline: shape inference needs e.g. reduction axes as raw data.
    onnx.save_model(
        model, str(destination), save_as_external_data=True, all_tensors_to_one_file=True,
        location=data, size_threshold=1024,
    )
    onnx.checker.check_model(str(destination))
    print(f"{source} -> {destination} ({precision}, {converted} ops)", flush=True)
    return converted


def export_graphs(directory):
    directory = Path(directory)
    if (directory / "encoder.onnx").is_file():
        if not (directory / "decoder.onnx").is_file():
            raise FileNotFoundError("LLM exports need decoder.onnx (export_llm_onnx.py --stage unified)")
        return ["encoder.onnx", "decoder.onnx"]
    if (directory / "model.onnx").is_file():
        return ["model.onnx"]
    raise FileNotFoundError(f"No LLM (encoder.onnx) or CTC (model.onnx) graphs in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--precision", nargs="+", choices=sorted(HALF), required=True)
    args = parser.parse_args()
    for precision in args.precision:
        for graph in export_graphs(args.directory):
            convert_graph(args.directory / graph, args.directory / precision / graph, precision)


if __name__ == "__main__":
    main()

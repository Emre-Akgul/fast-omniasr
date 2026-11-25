"""Build the small weight-free ONNX graphs used to drop completed decoder batch rows."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def export(directory):
    for name, dtype in [("fp16", TensorProto.FLOAT16), ("fp32", TensorProto.FLOAT)]:
        rows = helper.make_tensor_value_info("rows", TensorProto.INT64, ["remaining"])
        inputs = [rows]
        outputs, nodes = [], []
        for n in range(24):
            dims = ["positions", "heads", "head_dim"]
            source, destination = f"cache_{n}", f"selected_{n}"
            inputs.append(helper.make_tensor_value_info(source, dtype, ["batch", *dims]))
            outputs.append(helper.make_tensor_value_info(destination, dtype, ["remaining", *dims]))
            nodes.append(helper.make_node("Gather", [source, "rows"], [destination], axis=0))
        graph = helper.make_graph(nodes, "select_cache_rows", inputs, outputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=9)
        onnx.checker.check_model(model)
        onnx.save(model, directory / f"llm_cache_gather_{name}.onnx")


if __name__ == "__main__":
    export(Path(__file__).resolve().parents[1] / "src/fast_omniasr/assets")

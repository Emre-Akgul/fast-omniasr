import onnx_bytes as ob
import pytest

from fast_omniasr._onnx_deps import external_data_locations, graph_dependencies


def _write(tmp_path, content, name="model.onnx"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_inline_only_graph_has_no_dependencies(tmp_path):
    path = _write(tmp_path, ob.model_with_external(inline=b"\x00" * 100_000))
    assert external_data_locations(path) == set()


def test_initializers_in_nested_and_shared_files(tmp_path):
    path = _write(tmp_path, ob.model_with_external(
        "model.onnx.data", "weights/encoder.bin", "weights/encoder.bin", "./weights/deep/x.bin"))
    assert external_data_locations(path) == {
        "model.onnx.data", "weights/encoder.bin", "./weights/deep/x.bin"}
    assert graph_dependencies("fp16/model.onnx", path) == {
        "fp16/model.onnx.data", "fp16/weights/encoder.bin", "fp16/weights/deep/x.bin"}


def test_tensors_in_attributes_subgraphs_and_sparse_initializers(tmp_path):
    then_branch = ob.graph([ob.tensor("sub", "sub.bin")],
                           nodes=[ob.constant_node(ob.tensor("deep", "deep_const.bin"))])
    graph = ob.graph(
        nodes=[ob.constant_node(ob.tensor("c", "const.bin")), ob.if_node(then_branch)],
        sparse=[(ob.tensor("values", "values.bin"), ob.tensor("indices", "indices.bin"))],
    )
    path = _write(tmp_path, ob.model(graph))
    assert external_data_locations(path) == {
        "const.bin", "sub.bin", "deep_const.bin", "values.bin", "indices.bin"}


@pytest.mark.parametrize("location", ["../outside.bin", "a/../../outside.bin", "/etc/passwd",
                                      "\\\\server\\share.bin"])
def test_locations_escaping_the_repo_are_rejected(tmp_path, location):
    path = _write(tmp_path, ob.model_with_external(location))
    with pytest.raises(ValueError, match="unsafe external-data location"):
        graph_dependencies("model.onnx", path)


def test_location_inside_a_subdirectory_may_use_dotdot_within_the_repo(tmp_path):
    path = _write(tmp_path, ob.model_with_external("../shared/weights.bin"))
    assert graph_dependencies("fp16/model.onnx", path) == {"shared/weights.bin"}


@pytest.mark.parametrize("content", [b"not an onnx model", b"\x3a\xff\xff\xff\xff\x0f", b""])
def test_malformed_files_raise_value_error(tmp_path, content):
    with pytest.raises(ValueError, match="model.onnx"):
        external_data_locations(_write(tmp_path, content))


def test_matches_the_onnx_package(tmp_path):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import helper, numpy_helper

    weights = [numpy_helper.from_array(np.full((64, 64), i, np.float32), f"w{i}")
               for i in range(3)]
    small = numpy_helper.from_array(np.ones(2, np.float32), "small")
    nodes = [helper.make_node("Add", ["x", f"w{i}"], [f"y{i}"]) for i in range(3)]
    graph = helper.make_graph(nodes, "g", [helper.make_tensor_value_info(
        "x", onnx.TensorProto.FLOAT, [64, 64])], [helper.make_tensor_value_info(
            "y0", onnx.TensorProto.FLOAT, [64, 64])], [*weights, small])
    model = helper.make_model(graph)
    # Write each weight to its own file (two nested, one beside the graph), as onnx does.
    (tmp_path / "weights").mkdir()
    for i, tensor in enumerate(model.graph.initializer[:3]):
        onnx.external_data_helper.set_external_data(
            tensor, f"weights/part{i}.bin" if i < 2 else "model.onnx.data")
    onnx.external_data_helper.write_external_data_tensors(model, str(tmp_path))
    onnx.save_model(model, tmp_path / "model.onnx")
    reference = set()
    loaded = onnx.load(str(tmp_path / "model.onnx"), load_external_data=False)
    for tensor in loaded.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            reference.add({e.key: e.value for e in tensor.external_data}["location"])
    assert reference == {"weights/part0.bin", "weights/part1.bin", "model.onnx.data"}
    assert (tmp_path / "weights" / "part1.bin").stat().st_size == 64 * 64 * 4
    assert external_data_locations(tmp_path / "model.onnx") == reference

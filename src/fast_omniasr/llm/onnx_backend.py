"""ORT session construction with shared, explicit decoder initializers (no Torch)."""

import gc
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def options(threads=4):
    result = ort.SessionOptions()
    result.intra_op_num_threads = threads
    # Decoder cache shapes grow every call. A shape-specific memory pattern can
    # retain stale CUDA allocations until this 5 GiB model exhausts a 6 GiB GPU.
    result.enable_mem_pattern = False
    # Avoid separate packed copies of the 5 GiB weights in prefill and step.
    result.add_session_config_entry("session.disable_prepacking", "1")
    return result


def providers(device):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    name = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if device == "cuda":
        ort.preload_dlls(directory="")
    if name not in ort.get_available_providers():
        raise RuntimeError(f"Required provider {name} is unavailable")
    return (
        [
            (
                name,
                {
                    "use_tf32": "0",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ]
        if device == "cuda"
        else [name]
    )


def run_options(device):
    if device != "cuda":
        return None
    result = ort.RunOptions()
    result.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    return result


def session(path, device="cpu", opts=None):
    result = ort.InferenceSession(
        str(path), sess_options=opts or options(), providers=providers(device)
    )
    # This controls session.run() error recovery. It does not prevent ORT from
    # assigning graph-shape plumbing to the CPU EP when CUDA is requested.
    result.disable_fallback()
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if result.get_providers()[0] != expected:
        raise RuntimeError(f"Required provider {expected} did not initialize")
    return result


def _decoder_options(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "decoder_weights.json").read_text())
    opts = options()
    keepalive = []
    for name, entry in metadata.items():
        array = np.memmap(
            directory / "decoder.weights",
            dtype=entry["dtype"],
            mode="r",
            offset=entry["offset"],
            shape=tuple(entry["shape"]),
        )
        value = ort.OrtValue.ortvalue_from_numpy(array)
        opts.add_initializer(name, value)
        keepalive.append((array, value))
    return opts, keepalive


def decoder_session(directory, graph):
    """Load one CPU decoder phase with memory-mapped shared initializers."""
    directory = Path(directory)
    opts, keepalive = _decoder_options(directory)
    return session(directory / graph, "cpu", opts), keepalive


def decoder_sessions(directory, device="cpu"):
    directory = Path(directory)
    providers(device)
    if device != "cpu":
        raise ValueError("Shared decoder sessions are CPU-only; phase CUDA sessions instead")
    opts, keepalive = _decoder_options(directory)
    prefill = session(directory / "decoder_prefill.onnx", device, opts)
    step = session(directory / "decoder_step.onnx", device, opts)
    return prefill, step, keepalive


class ONNXLLMBackend:
    """Batch-one float32 graphs with caller-owned NumPy K/V tensors.

    Encoder and decoder sessions are loaded in separate phases so the same
    runtime can fit the full model on a 6 GiB GPU. Decoder weights are shared
    between prefill and step; cache transfers remain explicit CPU copies.
    """

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory)
        self.device = device
        providers(device)
        self.prefill_session = self.step_session = None
        self.keepalive = []
        self.run_options = run_options(device)
        for name in [
            "encoder.onnx",
            "decoder_prefill.onnx",
            "decoder_step.onnx",
            "decoder.weights",
            "decoder_weights.json",
        ]:
            if not (self.directory / name).is_file():
                raise FileNotFoundError(self.directory / name)

    def close(self):
        self.prefill_session = self.step_session = None
        self.keepalive = []
        gc.collect()

    def encode(self, audio):
        self.close()
        encoder = session(self.directory / "encoder.onnx", self.device)
        context = encoder.run(
            ["context"], {"audio": np.ascontiguousarray(audio)}, self.run_options
        )[0]
        del encoder
        gc.collect()
        return context

    def initialize_decoder(self):
        if self.prefill_session is None:
            if self.device == "cpu":
                self.prefill_session, self.step_session, self.keepalive = decoder_sessions(
                    self.directory, self.device
                )
            else:
                self.prefill_session = session(
                    self.directory / "decoder_prefill.onnx", self.device
                )
            expected_prefill = ["context", "bos"]
            expected_step = ["token"] + [
                f"past_{kind}_{i}" for i in range(12) for kind in ("key", "value")
            ]
            if [v.name for v in self.prefill_session.get_inputs()] != expected_prefill:
                raise ValueError("Invalid ONNX prefill input contract")
            if self.step_session is not None and [
                v.name for v in self.step_session.get_inputs()
            ] != expected_step:
                raise ValueError("Invalid ONNX step input contract")

    def prefill(self, context, bos):
        self.initialize_decoder()
        outputs = self.prefill_session.run(
            None, {"context": context, "bos": bos}, self.run_options
        )
        if self.device == "cuda":
            # Detach NumPy outputs from ORT-owned values before destroying the
            # prefill session and reclaiming its CUDA allocator.
            outputs = [np.array(value, copy=True) for value in outputs]
            self.prefill_session = None
            gc.collect()
            self.step_session = session(self.directory / "decoder_step.onnx", self.device)
            expected = ["token"] + [
                f"past_{kind}_{i}" for i in range(12) for kind in ("key", "value")
            ]
            if [value.name for value in self.step_session.get_inputs()] != expected:
                raise ValueError("Invalid ONNX step input contract")
        return outputs[0], outputs[1:]

    def decode_step(self, token, cache):
        if token.shape != (1, 1) or token.dtype != np.int64 or len(cache) != 24:
            raise ValueError("Expected one int64 token and 24 K/V tensors")
        if self.step_session is None:
            if self.device == "cuda":
                raise RuntimeError("prefill() must initialize the CUDA step session")
            self.initialize_decoder()
        names = [v.name for v in self.step_session.get_inputs()]
        outputs = self.step_session.run(
            None,
            dict(zip(names, [token, *cache], strict=True)),
            self.run_options,
        )
        return outputs[0], outputs[1:]

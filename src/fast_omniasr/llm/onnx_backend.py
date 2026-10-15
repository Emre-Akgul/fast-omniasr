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


def providers(device, arena="kSameAsRequested"):
    """`arena` is the CUDA arena extend strategy. kSameAsRequested suits phased, shrinking
    sessions; resident sessions use kNextPowerOfTwo so freed cache chunks coalesce and are
    reused as the cache grows instead of stranding one exact-size chunk per length."""
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
                    "arena_extend_strategy": arena,
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


def session(path, device="cpu", opts=None, arena="kSameAsRequested"):
    result = ort.InferenceSession(
        str(path), sess_options=opts or options(), providers=providers(device, arena)
    )
    # This controls session.run() error recovery. It does not prevent ORT from
    # assigning graph-shape plumbing to the CPU EP when CUDA is requested.
    result.disable_fallback()
    expected = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    if result.get_providers()[0] != expected:
        raise RuntimeError(f"Required provider {expected} did not initialize")
    return result


def _decoder_options(directory, threads=4):
    directory = Path(directory)
    metadata = json.loads((directory / "decoder_weights.json").read_text())
    opts = options(threads)
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


def decoder_sessions(directory, device="cpu", threads=4):
    directory = Path(directory)
    providers(device)
    if device != "cpu":
        raise ValueError("Shared decoder sessions are CPU-only; phase CUDA sessions instead")
    opts, keepalive = _decoder_options(directory, threads)
    prefill = session(directory / "decoder_prefill.onnx", device, opts)
    step = session(directory / "decoder_step.onnx", device, opts)
    return prefill, step, keepalive


STEP_INPUTS = ["token"] + [f"past_{kind}_{i}" for i in range(12) for kind in ("key", "value")]
UNIFIED_INPUTS = ["context", "tokens", *STEP_INPUTS[1:]]


def decoder_session(directory, graph, device="cpu", threads=4, arena="kSameAsRequested"):
    """One decoder graph. On CPU its weights are memory-mapped shared initializers."""
    directory = Path(directory)
    if device == "cpu":
        opts, keepalive = _decoder_options(directory, threads)
        return session(directory / graph, "cpu", opts), keepalive
    return session(directory / graph, device, options(threads), arena), []


class ONNXLLMBackend:
    """Batch-one float32 graphs; K/V caches are opaque values passed back to decode_step.

    With decoder.onnx (the unified graph) one decoder session serves prefill and every
    step, so CUDA holds a single copy of the decoder weights. Older exports fall back to
    separate prefill and step graphs, whose CUDA sessions each hold a copy.

    By default every session stays loaded across calls, and on CUDA the K/V cache stays on
    the GPU between steps through I/O binding: only token ids go in and logits come out.
    low_memory=True keeps one session resident at a time and returns caches as NumPy arrays
    so small GPUs fit, at the cost of reloading weights for every clip.
    """

    def __init__(self, directory, device="cpu", threads=4, low_memory=False):
        self.directory = Path(directory)
        self.device = device
        self.threads = threads
        self.low_memory = low_memory
        self.arena = "kSameAsRequested" if low_memory else "kNextPowerOfTwo"
        providers(device)
        self.unified = (self.directory / "decoder.onnx").is_file()
        self.encoder_session = self.prefill_session = self.step_session = None
        self.decoder = None
        self.keepalive = []
        # Shrinking returns arena memory after every run; only phased sessions need it.
        self.run_options = run_options(device) if low_memory else None
        graphs = ["decoder.onnx"] if self.unified else ["decoder_prefill.onnx", "decoder_step.onnx"]
        for name in ["encoder.onnx", *graphs, "decoder.weights", "decoder_weights.json"]:
            if not (self.directory / name).is_file():
                raise FileNotFoundError(self.directory / name)

    @property
    def bound(self):
        return self.device == "cuda" and not self.low_memory

    def close(self):
        self.encoder_session = self.prefill_session = self.step_session = None
        self.decoder = None
        self.keepalive = []
        gc.collect()

    def encode(self, audio):
        if self.low_memory:
            self.close()
        if self.encoder_session is None:
            self.encoder_session = session(
                self.directory / "encoder.onnx", self.device, options(self.threads), self.arena
            )
        context = self.encoder_session.run(
            ["context"], {"audio": np.ascontiguousarray(audio)}, self.run_options
        )[0]
        if self.low_memory:
            self.encoder_session = None
            gc.collect()
        return context

    def _initialize_unified(self):
        if self.decoder is None:
            self.decoder, self.keepalive = decoder_session(
                self.directory, "decoder.onnx", self.device, self.threads, self.arena
            )
            inputs = self.decoder.get_inputs()
            if [value.name for value in inputs] != UNIFIED_INPUTS:
                raise ValueError("Invalid ONNX unified decoder input contract")
            hidden, (heads, dim) = inputs[0].shape[2], inputs[2].shape[2:]
            self.empty_context = np.zeros((1, 0, hidden), np.float32)
            self.empty_cache = [np.zeros((1, 0, heads, dim), np.float32)] * 24

    def initialize_decoder(self):
        if self.unified:
            self._initialize_unified()
            return
        if self.prefill_session is None:
            if self.device == "cpu":
                self.prefill_session, self.step_session, self.keepalive = decoder_sessions(
                    self.directory, self.device, self.threads
                )
            else:
                self.prefill_session = session(
                    self.directory / "decoder_prefill.onnx",
                    self.device,
                    options(self.threads),
                    self.arena,
                )
                if not self.low_memory:
                    self.step_session = self._step()
            if [v.name for v in self.prefill_session.get_inputs()] != ["context", "bos"]:
                raise ValueError("Invalid ONNX prefill input contract")
            if self.step_session is not None and [
                v.name for v in self.step_session.get_inputs()
            ] != STEP_INPUTS:
                raise ValueError("Invalid ONNX step input contract")

    def _step(self):
        result = session(
            self.directory / "decoder_step.onnx", self.device, options(self.threads), self.arena
        )
        if [value.name for value in result.get_inputs()] != STEP_INPUTS:
            raise ValueError("Invalid ONNX step input contract")
        return result

    def _run_bound(self, graph, host_inputs, device_inputs=()):
        """Run with logits copied to the host and every K/V output left on the GPU."""
        binding = graph.io_binding()
        for name, value in host_inputs.items():
            binding.bind_cpu_input(name, np.ascontiguousarray(value))
        for name, value in device_inputs:
            binding.bind_ortvalue_input(name, value)
        names = [value.name for value in graph.get_outputs()]
        binding.bind_output(names[0], "cpu")
        for name in names[1:]:
            binding.bind_output(name, "cuda")
        graph.run_with_iobinding(binding)
        outputs = binding.get_outputs()
        return outputs[0].numpy(), outputs[1:]

    def _run_unified(self, context, tokens, cache):
        self._initialize_unified()
        if self.bound:
            if cache is None:
                host = dict(zip(UNIFIED_INPUTS, [context, tokens, *self.empty_cache]))
                return self._run_bound(self.decoder, host)
            return self._run_bound(
                self.decoder,
                {"context": context, "tokens": tokens},
                zip(UNIFIED_INPUTS[2:], cache, strict=True),
            )
        feeds = [context, tokens, *(self.empty_cache if cache is None else cache)]
        outputs = self.decoder.run(
            None, dict(zip(UNIFIED_INPUTS, feeds, strict=True)), self.run_options
        )
        return outputs[0], outputs[1:]

    def prefill(self, context, bos):
        if self.unified:
            return self._run_unified(context, bos, None)
        self.initialize_decoder()
        if self.bound:
            return self._run_bound(self.prefill_session, {"context": context, "bos": bos})
        outputs = self.prefill_session.run(
            None, {"context": context, "bos": bos}, self.run_options
        )
        if self.device == "cuda":
            # Detach NumPy outputs from ORT-owned values before destroying the
            # prefill session and reclaiming its CUDA allocator.
            outputs = [np.array(value, copy=True) for value in outputs]
            self.prefill_session = None
            gc.collect()
            self.step_session = self._step()
        return outputs[0], outputs[1:]

    def decode_step(self, token, cache):
        if token.shape != (1, 1) or token.dtype != np.int64 or len(cache) != 24:
            raise ValueError("Expected one int64 token and 24 K/V tensors")
        if self.unified:
            self._initialize_unified()
            return self._run_unified(self.empty_context, token, cache)
        if self.step_session is None:
            if self.device == "cuda":
                raise RuntimeError("prefill() must initialize the CUDA step session")
            self.initialize_decoder()
        if self.bound:
            return self._run_bound(
                self.step_session, {"token": token}, zip(STEP_INPUTS[1:], cache, strict=True)
            )
        outputs = self.step_session.run(
            None,
            dict(zip(STEP_INPUTS, [token, *cache], strict=True)),
            self.run_options,
        )
        return outputs[0], outputs[1:]

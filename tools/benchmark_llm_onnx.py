"""Benchmark cached TorchScript and ONNX Runtime phases on identical forced tokens."""

import argparse
import gc
import json
import os
import resource
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np


class GPUMemoryMonitor:
    def __init__(self, enabled):
        self.enabled = enabled
        self.peak_mib = None
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *unused):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _poll(self):
        while not self._stop.wait(0.5):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                )
            except (OSError, subprocess.SubprocessError):
                return
            for line in result.stdout.splitlines():
                pid, used = (part.strip() for part in line.split(",", 1))
                if pid == str(os.getpid()):
                    value = float(used)
                    self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)


def median(values):
    return statistics.median(values)


def measure_torch(source, audio, tokens, repeats):
    import torch

    torch.set_num_threads(4)
    start = time.perf_counter()
    encoder = torch.jit.load(str(source / "encoder.pt")).eval()
    initialization = time.perf_counter() - start
    encoder_times = []
    with torch.inference_mode():
        for repeat in range(repeats + 1):
            start = time.perf_counter()
            context = encoder(torch.from_numpy(audio))
            elapsed = time.perf_counter() - start
            if repeat:
                encoder_times.append(elapsed)
        del encoder
        gc.collect()
        start = time.perf_counter()
        decoder = torch.jit.load(str(source / "decoder_cached.pt")).eval()
        initialization += time.perf_counter() - start
        trajectories = []
        for repeat in range(repeats + 1):
            start = time.perf_counter()
            _, cache = decoder.prefill(context, torch.tensor([[0]]))
            prefill = time.perf_counter() - start
            steps = []
            for token in tokens[:99]:
                start = time.perf_counter()
                _, cache = decoder.decode_step(torch.tensor([[token]]), cache)
                steps.append(time.perf_counter() - start)
            if repeat:
                trajectories.append((prefill, steps))
        del decoder
    return encoder_times, trajectories, initialization, context.shape[1]


def measure_ort(directory, audio, tokens, repeats, device):
    from fast_omniasr.llm.onnx_backend import decoder_sessions, run_options, session

    initialization = 0.0
    run_opts = run_options(device)
    start = time.perf_counter()
    encoder = session(directory / "encoder.onnx", device)
    initialization += time.perf_counter() - start
    encoder_times = []
    for repeat in range(repeats + 1):
        start = time.perf_counter()
        context = encoder.run(["context"], {"audio": audio}, run_opts)[0]
        elapsed = time.perf_counter() - start
        if repeat:
            encoder_times.append(elapsed)
    del encoder
    gc.collect()

    start = time.perf_counter()
    if device == "cpu":
        prefill_session, step_session, keepalive = decoder_sessions(directory, device)
    else:
        prefill_session = session(directory / "decoder_prefill.onnx", device)
        step_session, keepalive = None, []
    initialization += time.perf_counter() - start
    prefills, initial_caches = [], []
    bos = np.array([[0]], dtype=np.int64)
    for repeat in range(repeats + 1):
        start = time.perf_counter()
        outputs = prefill_session.run(None, {"context": context, "bos": bos}, run_opts)
        elapsed = time.perf_counter() - start
        if repeat:
            prefills.append(elapsed)
            initial_caches.append(outputs[1:])
    del prefill_session
    gc.collect()
    if step_session is None:
        start = time.perf_counter()
        step_session = session(directory / "decoder_step.onnx", device)
        initialization += time.perf_counter() - start

    names = [value.name for value in step_session.get_inputs()]
    trajectories = []
    for prefill, cache in zip(prefills, initial_caches, strict=True):
        steps = []
        for token in tokens[:99]:
            value = np.array([[token]], dtype=np.int64)
            start = time.perf_counter()
            outputs = step_session.run(
                None, dict(zip(names, [value, *cache], strict=True)), run_opts
            )
            steps.append(time.perf_counter() - start)
            cache = outputs[1:]
        trajectories.append((prefill, steps))
    del step_session, keepalive
    gc.collect()
    return encoder_times, trajectories, initialization, context.shape[1]


def summarize(encoder, trajectories, initialization):
    result = {
        "encoder_seconds": median(encoder),
        "session_initialization_seconds": initialization,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    for count in [10, 25, 50, 100]:
        totals, prefills, per_tokens = [], [], []
        for prefill, steps in trajectories:
            selected = steps[: count - 1]
            totals.append(prefill + sum(selected))
            prefills.append(prefill)
            per_tokens.append(statistics.mean(selected) if selected else 0.0)
        result[str(count)] = {
            "decoder_total_seconds": median(totals),
            "prefill_seconds": median(prefills),
            "mean_decode_step_seconds": median(per_tokens),
            "runs_seconds": totals,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--backend", choices=["torch", "onnx-cpu", "onnx-cuda"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from fast_omniasr.audio import load_audio

    reference = json.loads((args.source / "reference.json").read_text())
    audio = load_audio(args.source / reference["audio"])
    tokens = (reference["token_ids"] + [reference["token_ids"][-1]]) * 3
    with GPUMemoryMonitor(args.backend == "onnx-cuda") as gpu_memory:
        if args.backend == "torch":
            encoder, trajectories, initialization, context_positions = measure_torch(
                args.source, audio, tokens, args.repeats
            )
        else:
            encoder, trajectories, initialization, context_positions = measure_ort(
                args.directory,
                audio,
                tokens,
                args.repeats,
                args.backend.removeprefix("onnx-"),
            )
    report = {
        "backend": args.backend,
        "device": "cpu" if args.backend == "torch" else args.backend.rsplit("-", 1)[-1],
        "threads": 4,
        "repeats": args.repeats,
        "latencies_exclude_session_initialization": True,
        "fixture_context_positions": context_positions,
        "cache_bytes_per_position": 12 * 2 * 8 * 512 * 4,
        "cache_bytes_at_100_output_positions": (
            12 * 2 * (context_positions + 100) * 8 * 512 * 4
        ),
        "peak_gpu_process_mib": gpu_memory.peak_mib,
        "measurements": summarize(encoder, trajectories, initialization),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

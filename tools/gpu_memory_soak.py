"""Check a model for memory that grows with every request (issue #96).

Runs the same request (or a seeded mix of lengths) many times through the public API and
samples every memory counter the platform exposes after each one:

- `gpu_process_mb`: this process's GPU memory from NVML (discrete GPUs).
- `gpu_device_mb`: used memory of the whole device from NVML.
- `nvmap_mb`: `NvMapMemUsed` from /proc/meminfo. On integrated GPUs whose memory is shared
  with the host, GPU allocations live here, and NVML may report nothing.
- `cgroup_kernel_mb`: the `kernel` line of the container's cgroup v2 memory.stat, which
  counts driver memory charged to the container.
- `rss_mb`: the process's resident memory. GPU allocations don't show up here.

After `--warmup` requests the counters should be flat (with `--mixed`, host caches keep
filling for the first few dozen lengths, so the default warm-up is half the run). Exit status 1 when any counter grows
by more than `--tolerance-mb` from the end of warm-up to the last request.

    python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --device cuda
    python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-CTC-300M-v2-ONNX --device cuda \\
        --arena-shrinkage        # the per-run arena shrinkage some servers turn on
    python tools/gpu_memory_soak.py --model EmreAkgul/omniASR-LLM-300M-v2-ONNX --kind llm \\
        --device cuda --low-memory --requests 30
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

MB = 2**20


def meminfo_mb(field: str, path: str = "/proc/meminfo") -> float | None:
    """A /proc/meminfo field (reported in kB) in MiB, or None when it isn't there."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, rest = line.partition(":")
        if name == field:
            return int(rest.split()[0]) * 1024 / MB
    return None


def cgroup_kernel_mb(path: str = "/sys/fs/cgroup/memory.stat") -> float | None:
    """The `kernel` counter of the current cgroup v2, in MiB, or None without cgroup v2."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(" ")
        if name == "kernel":
            return int(value) / MB
    return None


def rss_mb() -> float | None:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / MB


class NVML:
    """Process and device memory from NVML; every reading is None when NVML is missing or
    unsupported (as on some integrated GPUs)."""

    def __init__(self, index: int = 0):
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.pynvml = pynvml
        except Exception:  # noqa: BLE001 -- not installed, no driver, or no NVML on this GPU
            self.handle = None

    def process_mb(self) -> float | None:
        if self.handle is None:
            return None
        try:
            processes = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        except Exception:  # noqa: BLE001
            return None
        used = [p.usedGpuMemory for p in processes if p.pid == os.getpid() and p.usedGpuMemory]
        return used[0] / MB if used else None

    def device_mb(self) -> float | None:
        if self.handle is None:
            return None
        try:
            return self.pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / MB
        except Exception:  # noqa: BLE001
            return None


def sample(nvml: NVML) -> dict:
    return {
        "gpu_process_mb": nvml.process_mb(),
        "gpu_device_mb": nvml.device_mb(),
        "nvmap_mb": meminfo_mb("NvMapMemUsed"),
        "cgroup_kernel_mb": cgroup_kernel_mb(),
        "rss_mb": rss_mb(),
    }


def growth(samples: list[dict], warmup: int) -> dict:
    """Per counter, the change from the last warm-up sample to the final sample. Counters
    that were never available are left out."""
    if warmup < 1 or len(samples) <= warmup:
        raise ValueError("need more requests than warm-up requests")
    start, end = samples[warmup - 1], samples[-1]
    return {name: end[name] - start[name] for name in end
            if start.get(name) is not None and end[name] is not None}


def lengths(count: int, seconds: float, mixed: bool, seed: int = 0) -> list[int]:
    """Sample counts per request: all `seconds` long, or seeded uniform in [1 s, seconds]."""
    if not mixed:
        return [int(seconds * 16000)] * count
    rng = np.random.default_rng(seed)
    return [int(value * 16000) for value in rng.uniform(1.0, seconds, count)]


class _ShrinkingSession:
    """Passes run options that release the CUDA arena after every run."""

    def __init__(self, session, options):
        self.session, self.options = session, options

    def run(self, names, feeds, run_options=None):
        return self.session.run(names, feeds, self.options)

    def __getattr__(self, name):
        return getattr(self.session, name)


def shrink_arena(model) -> None:
    """Make every ONNX Runtime CTC run release its CUDA arena afterwards, as
    `memory.enable_memory_arena_shrinkage` does in servers that turn it on."""
    import onnxruntime as ort

    session = getattr(model.backend, "session", None)
    if session is None:
        raise ValueError("--arena-shrinkage needs the ONNX Runtime CTC backend")
    options = ort.RunOptions()
    options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    model.backend.session = _ShrinkingSession(session, options)


def load(args):
    options = {"cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
    if args.precision:
        options["precision"] = args.precision
    if args.kind == "llm":
        from fast_omniasr.llm import OmniASRLLM

        return OmniASRLLM.from_pretrained(args.model, device=args.device,
                                          low_memory=args.low_memory, **options)
    from fast_omniasr import OmniASR

    if args.backend == "tensorrt":
        return OmniASR.from_pretrained(args.model, backend="tensorrt", **options)
    return OmniASR.from_pretrained(args.model, device=args.device, **options)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="Hugging Face repo id")
    parser.add_argument("--kind", choices=("ctc", "llm"), default="ctc")
    parser.add_argument("--backend", choices=("onnx", "tensorrt"), default="onnx")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision")
    parser.add_argument("--low-memory", action="store_true", help="LLM only")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--audio", help="audio file to repeat (default: seeded noise)")
    parser.add_argument("--seconds", type=float, default=20.0, help="noise length; default 20")
    parser.add_argument("--mixed", action="store_true",
                        help="seeded mix of noise lengths between 1 s and --seconds")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--warmup", type=int,
                        help="requests before counters must be flat; default 5, or half of "
                             "--requests with --mixed, since each new length can fill caches")
    parser.add_argument("--tolerance-mb", type=float, default=8.0)
    parser.add_argument("--arena-shrinkage", action="store_true",
                        help="CTC ONNX only: shrink the CUDA arena after every run")
    parser.add_argument("--output", type=Path, help="write every sample as JSON")
    args = parser.parse_args(argv)
    if args.warmup is None:
        args.warmup = args.requests // 2 if args.mixed else 5
    if args.requests <= args.warmup or args.warmup < 1:
        parser.error("--requests must be larger than --warmup, and --warmup at least 1")
    if args.audio and args.mixed:
        parser.error("--mixed uses noise; drop --audio")

    nvml = NVML()
    model = load(args)
    if args.arena_shrinkage:
        shrink_arena(model)
    llm = args.kind == "llm"
    # CTC takes arrays through transcribe_numpy; the LLM's transcribe takes either.
    run = model.transcribe if args.audio or llm else model.transcribe_numpy
    request = {"lang": "eng_Latn", "max_new_tokens": 32} if llm else {}
    rng = np.random.default_rng(1)
    samples = [{"request": 0, **sample(nvml)}]
    with model:
        for index, count in enumerate(lengths(args.requests, args.seconds, args.mixed), 1):
            audio = args.audio or (rng.standard_normal(count) * 0.05).astype(np.float32)
            run(audio, **request)
            samples.append({"request": index, "samples": None if args.audio else count,
                            **sample(nvml)})
            print(json.dumps(samples[-1]), file=sys.stderr, flush=True)
    counters = [{k: v for k, v in s.items() if k.endswith("_mb")} for s in samples[1:]]
    grown = growth(counters, args.warmup)
    summary = {
        "requests": args.requests,
        "warmup": args.warmup,
        "tolerance_mb": args.tolerance_mb,
        "growth_mb": {name: round(value, 1) for name, value in grown.items()},
        "per_request_mb": {name: round(value / (args.requests - args.warmup), 2)
                           for name, value in grown.items()},
        "leaking": sorted(name for name, value in grown.items() if value > args.tolerance_mb),
    }
    if args.output:
        args.output.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2))
    print(json.dumps(summary, indent=2))
    if not grown:
        print("no memory counter is available on this platform", file=sys.stderr)
        return 2
    return 1 if summary["leaking"] else 0


if __name__ == "__main__":
    sys.exit(main())

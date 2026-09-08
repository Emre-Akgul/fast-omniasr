"""Look up or build a locally-cached TensorRT engine for a given ONNX model + config.

TensorRT engines are not portable across GPUs, TensorRT versions or CUDA versions, so the
cache key covers everything that can invalidate a previously-built engine: the ONNX content,
the precision and optimization profile requested, the local TensorRT version and GPU identity,
and this module's own build-recipe version.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path

from .tensorrt_builder import BUILDER_CONFIG_VERSION, build_engine

_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "fast-omniasr" / "tensorrt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_environment() -> dict:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "Install fast-omniasr[tensorrt] and a compatible TensorRT installation"
        ) from exc
    try:
        from cuda.bindings import runtime as cudart
    except ImportError as exc:
        raise ImportError("Install fast-omniasr[tensorrt] for the cuda-python dependency") from exc
    error, prop = cudart.cudaGetDeviceProperties(0)
    if int(error) != 0:
        raise RuntimeError(f"Failed to query the local GPU: {error}")
    name = prop.name.decode() if isinstance(prop.name, bytes) else prop.name
    return {
        "tensorrt_version": trt.__version__,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "device_name": name,
    }


@contextlib.contextmanager
def _locked(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with lock_path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def get_or_build_engine(onnx_path: str | Path, *, precision: str,
                         min_samples: int = 16000, opt_samples: int = 80000,
                         max_samples: int = 480000, cache_dir: str | Path | None = None) -> Path:
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be 'fp32' or 'fp16'")
    onnx_path = Path(onnx_path)
    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_ROOT
    metadata = {
        "source_sha256": _sha256_file(onnx_path),
        "precision": precision,
        "profile": {"min": min_samples, "opt": opt_samples, "max": max_samples},
        "tf32": False,
        "builder_config_version": BUILDER_CONFIG_VERSION,
        **_local_environment(),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    entry_dir = root / key
    engine_path = entry_dir / "model.engine"
    metadata_path = entry_dir / "metadata.json"

    with _locked(root / f"{key}.lock"):
        if engine_path.exists() and metadata_path.exists():
            try:
                cached = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = None
            if cached == metadata:
                return engine_path
        build_engine(onnx_path, engine_path, precision=precision, min_samples=min_samples,
                     opt_samples=opt_samples, max_samples=max_samples)
        tmp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
        tmp_metadata_path.write_text(json.dumps(metadata, indent=2))
        os.replace(tmp_metadata_path, metadata_path)
    return engine_path

import builtins
import hashlib
import json
import shutil
import sys
import types

import pytest

from fast_omniasr import OmniASR, hub


def test_missing_huggingface_hub_raises_helpful_import_error(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("simulated: huggingface_hub not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match="fast-omniasr\\[hub\\]"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX")


def test_tensorrt_backend_requires_explicit_precision_before_any_download():
    with pytest.raises(ValueError, match="precision='fp32' or precision='fp16'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="tensorrt")


def test_unknown_backend_rejected_before_any_download():
    with pytest.raises(ValueError, match="backend must be 'onnx' or 'tensorrt'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="bogus")


def test_onnx_backend_rejects_unknown_precision_before_any_download():
    with pytest.raises(ValueError, match="precision must be one of fp32, fp16, bf16"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx", precision="int8")


def test_onnx_backend_rejects_bf16_on_cpu_before_any_download():
    with pytest.raises(ValueError, match="needs device='cuda'"):
        OmniASR.from_pretrained("EmreAkgul/omniASR-CTC-300M-v2-ONNX", backend="onnx", precision="bf16")


def test_onnx_fp16_downloads_only_its_variant(monkeypatch, tmp_path):
    remote = tmp_path / "remote"
    files = ["model.onnx", "model.onnx.data", "tokenizer.model", "fp16/model.onnx",
             "fp16/model.onnx.data", "bf16/model.onnx", "bf16/model.onnx.data"]
    entries = {}
    for name in files:
        (remote / name).parent.mkdir(parents=True, exist_ok=True)
        (remote / name).write_bytes(name.encode())
        entries[name] = {"sha256": hashlib.sha256(name.encode()).hexdigest(), "size": len(name)}
    (remote / "config.json").write_text(json.dumps({"format": 1, "files": entries}))
    fetched = []

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None):
        fetched.append(filename)
        (local_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / filename, local_dir / filename)
        return str(local_dir / filename)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    loaded = {}
    monkeypatch.setattr(OmniASR, "__init__", lambda self, *args, **kwargs: loaded.update(
        model=args[0], **kwargs
    ))
    OmniASR.from_pretrained(
        "EmreAkgul/omniASR-CTC-300M-v2-ONNX", precision="fp16", cache_dir=tmp_path / "cache"
    )
    assert sorted(fetched) == ["config.json", "fp16/model.onnx", "fp16/model.onnx.data", "tokenizer.model"]
    assert loaded["model"].parts[-2:] == ("fp16", "model.onnx")
    assert loaded["precision"] == "fp16"


@pytest.mark.parametrize("parameter", ["min_samples", "opt_samples", "max_samples"])
def test_tensorrt_profile_rejected_for_onnx_backend(parameter):
    with pytest.raises(ValueError, match="only used with backend='tensorrt'"):
        OmniASR.from_pretrained(
            "EmreAkgul/omniASR-CTC-300M-v2-ONNX",
            backend="onnx",
            **{parameter: 8000},
        )


# --- revision / repo_id names can't escape the cache root (issue #14) ---


UNSAFE_NAMES = ["..", ".", "../../etc", "/abs/path", "refs/pr/1", "a/../../b", "..\\..\\x",
                "C:\\temp", ".hidden", "a..b", "", "x" * 300, "tab\tand space", "ünïcode"]


@pytest.mark.parametrize("name", ["main", "v1.0", "1.0.0-rc1", "release_2",
                                  "0123456789abcdef0123456789abcdef01234567"])
def test_safe_names_are_kept_verbatim(name):
    assert hub._path_component(name) == name


@pytest.mark.parametrize("name", UNSAFE_NAMES)
def test_unsafe_names_become_one_safe_component(name):
    component = hub._path_component(name)
    assert component not in ("", ".", "..")
    assert "/" not in component and "\\" not in component and ":" not in component
    assert ".." not in component and not component.startswith(".")
    assert len(component) <= 255
    assert "%" in component  # encoded names can never collide with verbatim ones


def test_distinct_names_map_to_distinct_components():
    names = UNSAFE_NAMES + ["main", "refs%2Fpr%2F1", "refs-pr-1", "y" * 300, "%2E%2E",
                            "%empty", "~x", "%7Ex"]
    assert len({hub._path_component(n) for n in names}) == len(names)


def _fake_hub(monkeypatch, tmp_path):
    payload = b"tokenizer"
    config = {"format": 1, "files": {"tokenizer.model": {
        "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}}}
    written = []

    def hf_hub_download(repo_id, filename, revision=None, local_dir=None):
        target = local_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(json.dumps(config).encode() if filename == "config.json" else payload)
        written.append(target)
        return str(target)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=hf_hub_download)
    )
    return written


@pytest.mark.parametrize("revision", UNSAFE_NAMES)
def test_malicious_revision_stays_inside_cache_root(monkeypatch, tmp_path, revision):
    written = _fake_hub(monkeypatch, tmp_path)
    cache = tmp_path / "nested" / "cache"
    _, paths = hub.download_verified("owner/repo", {"tokenizer.model"}, revision=revision,
                                     cache_dir=cache)
    root = cache.resolve()
    for path in written + list(paths.values()):
        assert path.resolve().is_relative_to(root)
        assert path.resolve().parent.parent == root / "owner--repo"
    # nothing was created next to (or above) the cache root
    assert sorted(p.name for p in (tmp_path / "nested").iterdir()) == ["cache"]


@pytest.mark.parametrize("repo_id", ["..", "../..", "/etc", "owner/.."])
def test_malicious_repo_id_stays_inside_cache_root(monkeypatch, tmp_path, repo_id):
    written = _fake_hub(monkeypatch, tmp_path)
    cache = tmp_path / "cache"
    hub.download_verified(repo_id, {"tokenizer.model"}, cache_dir=cache)
    for path in written:
        assert path.resolve().parent.parent.parent == cache.resolve()


def test_default_revision_layout_is_unchanged(monkeypatch, tmp_path):
    _fake_hub(monkeypatch, tmp_path)
    _, paths = hub.download_verified("owner/repo", {"tokenizer.model"}, cache_dir=tmp_path)
    assert paths["tokenizer.model"] == tmp_path / "owner--repo" / "main" / "tokenizer.model"


def test_containment_check_rejects_escaping_layout(monkeypatch, tmp_path):
    # Defense in depth: even if component encoding regressed, the bundle dir is refused.
    monkeypatch.setattr(hub, "_path_component", lambda name: name)
    with pytest.raises(ValueError, match="outside"):
        hub._bundle_dir(tmp_path / "cache", "owner/repo", "../../../escape")

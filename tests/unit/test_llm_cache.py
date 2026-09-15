"""Exercise production cache math using Torch alone and an independent dense oracle."""

import importlib.abc
import importlib.util
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location(
    "llm_cache", Path(__file__).resolve().parents[2] / "tools" / "llm_cache.py"
)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


class BlockReferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fairseq2", "fairseq2n", "omnilingual_asr"}:
            raise AssertionError(f"Torch-only cache test imported {fullname}")


@pytest.fixture(autouse=True)
def limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class Residual(torch.nn.Module):
    def forward(self, x, residual):
        return x + residual


class GLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(32, 64, bias=False)
        self.up = torch.nn.Linear(32, 64, bias=False)
        self.down = torch.nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


def empty_module(cls):
    # Bypass only the fairseq2 weight-import constructor. Forward, rotate,
    # prefill, and decode_step remain the actual production implementations.
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def synthetic_decoder(kv_heads):
    model = empty_module(cache_module.ExplicitDecoder)
    model.embedding = torch.nn.Embedding(16, 32)
    model.norm = torch.nn.RMSNorm(32, eps=1e-5)
    model.proj = torch.nn.Linear(32, 16, bias=False)
    model.kv_heads, model.head_dim = kv_heads, 8
    model.layers = torch.nn.ModuleList()
    angles = torch.outer(torch.arange(256).float(), 10000.0 ** (-torch.arange(0, 8, 2) / 8))
    for _ in range(2):
        layer = empty_module(cache_module.CachedLayer)
        layer.norm = torch.nn.RMSNorm(32, eps=1e-5)
        layer.ffn_norm = torch.nn.RMSNorm(32, eps=1e-5)
        layer.ffn = GLU()
        layer.q = torch.nn.Linear(32, 32, bias=False)
        layer.k = torch.nn.Linear(32, kv_heads * 8, bias=False)
        layer.v = torch.nn.Linear(32, kv_heads * 8, bias=False)
        layer.out = torch.nn.Linear(32, 32, bias=False)
        layer.attn_residual = Residual()
        layer.ffn_residual = Residual()
        layer.head_dim, layer.groups = 8, 4 // kv_heads
        layer.register_buffer("freqs", torch.stack([angles.cos(), angles.sin()], dim=-1))
        model.layers.append(layer)
    return model.eval().requires_grad_(False)


def dense_reference(model, context, ids):
    """Full-prefix oracle: real-valued RoPE, explicit causal mask, no SDPA/cache."""
    x = torch.cat([context, model.embedding(ids)], dim=1)
    positions = torch.arange(x.shape[1], dtype=torch.float32)
    angles = positions[:, None] / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]

    def rotate(value):
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)

    cache = []
    for layer in model.layers:
        normalized = layer.norm(x)
        q = rotate(layer.q(normalized).reshape(1, -1, 4, 8))
        k = rotate(layer.k(normalized).reshape(1, -1, model.kv_heads, 8))
        v = layer.v(normalized).reshape(1, -1, model.kv_heads, 8)
        cache.extend([k, v])
        k = k.repeat_interleave(4 // model.kv_heads, dim=2).transpose(1, 2)
        v = v.repeat_interleave(4 // model.kv_heads, dim=2).transpose(1, 2)
        scores = q.transpose(1, 2) @ k.transpose(-1, -2) / math.sqrt(8)
        future = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1)
        weights = scores.masked_fill(future, float("-inf")).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(1, -1, 32)
        x = x + layer.out(attended)
        x = x + layer.ffn(layer.ffn_norm(x))
    return model.proj(model.norm(x)[:, -1]), cache


@pytest.mark.parametrize("kv_heads", [2, 4], ids=["gqa", "mha"])
def test_cached_graph_matches_full_prefix_at_unseen_lengths(tmp_path, monkeypatch, kv_heads):
    monkeypatch.setattr(sys, "meta_path", [BlockReferenceImports(), *sys.meta_path])
    torch.manual_seed(7)
    model = synthetic_decoder(kv_heads)
    bos = torch.tensor([[0]])
    with torch.inference_mode():
        context = torch.randn(1, 3, 32)
        _, cache = model.prefill(context, bos)
        graph = torch.jit.trace_module(
            model,
            {"prefill": (context, bos), "decode_step": (bos, cache)},
            check_trace=False,
            strict=False,
        )
        graph.save(str(tmp_path / "cache.pt"))
        graph = torch.jit.load(str(tmp_path / "cache.pt"))
        for prefix_len in (1, 7, 23):
            context = torch.randn(1, prefix_len, 32)
            logits, cache = graph.prefill(context, bos)
            initial_cache = [entry.clone() for entry in cache]
            ids = bos
            for step in range(8):
                expected, expected_cache = dense_reference(model, context, ids)
                torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-5)
                assert logits.argmax(-1).item() == expected.argmax(-1).item()
                for got, wanted in zip(cache, expected_cache, strict=True):
                    assert got.shape == (1, prefix_len + step + 1, kv_heads, 8)
                    torch.testing.assert_close(got, wanted, atol=2e-6, rtol=2e-5)
                token = expected.argmax(-1).reshape(1, 1)
                ids = torch.cat([ids, token], dim=1)
                old_cache = cache
                snapshots = [entry.clone() for entry in cache]
                logits, cache = graph.decode_step(token, cache)
                for before, after in zip(snapshots, old_cache, strict=True):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            _, fresh = graph.prefill(context, bos)
            for before, after in zip(initial_cache, fresh, strict=True):
                torch.testing.assert_close(before, after, rtol=0, atol=0)

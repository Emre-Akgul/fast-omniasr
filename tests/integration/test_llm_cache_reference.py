"""Optional fairseq2 integration coverage for the explicit-cache adapter."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fairseq2")
from fairseq2.models.llama import LLaMAConfig, LLaMAFactory
from fairseq2.nn import BatchLayout

spec = importlib.util.spec_from_file_location(
    "llm_cache", Path(__file__).resolve().parents[2] / "tools" / "llm_cache.py"
)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


@pytest.mark.integration
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_cached_graph_matches_full_prefix_at_unseen_lengths(tmp_path, kv_heads):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    config = LLaMAConfig(
        model_dim=32,
        num_layers=2,
        num_attn_heads=4,
        num_key_value_heads=kv_heads,
        ffn_inner_dim=64,
        ffn_inner_dim_multiple_of=1,
        vocab_size=16,
        max_seq_len=256,
    )
    factory = LLaMAFactory(config)
    model = SimpleNamespace(
        text_frontend=factory.create_embedding(),
        llama_decoder=factory.create_decoder(),
        final_proj=torch.nn.Linear(32, 16, bias=False),
    )
    for module in vars(model).values():
        module.eval().requires_grad_(False)
    explicit = cache_module.ExplicitDecoder(model).eval()
    bos = torch.tensor([[0]])
    with torch.inference_mode():
        context = torch.randn(1, 3, 32)
        _, cache = explicit.prefill(context, bos)
        graph = torch.jit.trace_module(
            explicit,
            {"prefill": (context, bos), "decode_step": (bos, cache)},
            check_trace=False,
            strict=False,
        )
        graph.save(str(tmp_path / "cache.pt"))
        graph = torch.jit.load(str(tmp_path / "cache.pt"))
        for prefix_len in (1, 7, 23):
            context = torch.randn(1, prefix_len, 32)
            logits, cache = graph.prefill(context, bos)
            ids = bos
            original_cache = [entry.clone() for entry in cache]
            for step in range(8):
                x = torch.cat([context, model.text_frontend(ids)], dim=1)
                layout = BatchLayout(x.shape[:2], seq_lens=None, device=x.device)
                expected = model.final_proj(model.llama_decoder(x, layout)[:, -1])
                torch.testing.assert_close(logits, expected, atol=2e-6, rtol=2e-5)
                assert all(
                    entry.shape == (1, prefix_len + step + 1, kv_heads, 8) for entry in cache
                )
                token = expected.argmax(-1).reshape(1, 1)
                ids = torch.cat([ids, token], dim=1)
                logits, cache = graph.decode_step(token, cache)
            # A new request must not retain the previous request's position/state.
            _, fresh = graph.prefill(context, bos)
            for a, b in zip(original_cache, fresh):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

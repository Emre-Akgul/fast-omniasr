"""Export-time explicit cache adapter for the supported fairseq2 LLaMA decoder.

Serialized graphs use only Torch operators. Cache entries alternate K,V per layer,
with shape [batch, sequence, kv_heads, head_dim]. Keys already include RoPE;
values do not. The sequence dimension is also the absolute next RoPE position.
"""

import torch
from torch import nn


class CachedLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        from fairseq2.models.transformer import TransformerNormOrder
        from fairseq2.nn import RotaryEncoder

        attn = layer.self_attn
        if layer.norm_order != TransformerNormOrder.PRE or not isinstance(
            attn.pos_encoder, RotaryEncoder
        ):
            raise ValueError("Expected pre-norm LLaMA with interleaved RoPE")
        if attn.q_norm is not None or attn.k_norm is not None:
            raise ValueError("Q/K normalization is not supported")
        self.norm = layer.self_attn_layer_norm
        self.ffn_norm = layer.ffn_layer_norm
        self.ffn = layer.ffn
        self.q = attn.q_proj
        self.k = attn.k_proj
        self.v = attn.v_proj
        self.out = attn.output_proj
        self.attn_residual = layer.self_attn_residual
        self.ffn_residual = layer.ffn_residual
        self.head_dim = attn.head_dim
        self.groups = attn.num_query_groups
        # fairseq2's first frequency row is padding, not position zero.
        self.register_buffer("freqs", attn.pos_encoder.freqs[1:].detach().clone())

    def rotate(self, x, offset):
        frequencies = torch.view_as_complex(self.freqs[offset : offset + x.shape[1]])
        pairs = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        return torch.view_as_real(pairs * frequencies[None, :, None]).flatten(-2).to(x.dtype)

    def forward(self, x, past_k, past_v, prefill: bool):
        residual = x
        x = self.norm(x)
        offset = past_k.shape[1]
        q = self.rotate(self.q(x).unflatten(-1, (-1, self.head_dim)), offset)
        k = self.rotate(self.k(x).unflatten(-1, (-1, self.head_dim)), offset)
        v = self.v(x).unflatten(-1, (-1, self.head_dim))
        k = torch.cat([past_k, k], dim=1)
        v = torch.cat([past_v, v], dim=1)
        keys, values = k, v
        if self.groups > 1:
            keys = keys.repeat_interleave(self.groups, dim=2)
            values = values.repeat_interleave(self.groups, dim=2)
        # A single new query attends to every cached position. is_causal=True
        # here would incorrectly mask all but the first key (upper-left mask).
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                is_causal=prefill,
            )
            .transpose(1, 2)
            .flatten(-2)
        )
        x = self.attn_residual(self.out(out), residual)
        x = self.ffn_residual(self.ffn(self.ffn_norm(x)), x)
        return x, k, v


class ExplicitDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.text_frontend
        self.layers = nn.ModuleList([CachedLayer(layer) for layer in model.llama_decoder.layers])
        self.norm = model.llama_decoder.layer_norm
        self.proj = model.final_proj
        attn = model.llama_decoder.layers[0].self_attn
        self.kv_heads = attn.num_key_value_heads
        self.head_dim = attn.head_dim

    def prefill(self, context, bos):
        x = torch.cat([context, self.embedding(bos)], dim=1)
        cache = []
        for layer in self.layers:
            empty = x.new_empty((x.shape[0], 0, self.kv_heads, self.head_dim))
            x, k, v = layer(x, empty, empty, True)
            cache.extend([k, v])
        if self.norm is not None:
            x = self.norm(x)
        return self.proj(x[:, -1]), cache

    def decode_step(self, token, cache):
        x = self.embedding(token)
        updated = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, cache[2 * i], cache[2 * i + 1], False)
            updated.extend([k, v])
        if self.norm is not None:
            x = self.norm(x)
        return self.proj(x[:, -1]), updated

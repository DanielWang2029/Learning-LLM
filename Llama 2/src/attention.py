"""Grouped-Query Attention (GQA) — Llama 2's key architecture change.

Llama 2 uses grouped-query attention for its larger models (Llama 2 paper,
Section 2.2 and Appendix A.2.1; Ainslie et al. 2023). The query heads are
divided into ``n_kv_heads`` groups, and each group shares one key/value head:

    n_kv_heads == n_heads     -> Multi-Head Attention (MHA)   — most K/V memory
    1 < n_kv_heads < n_heads  -> Grouped-Query Attention (GQA) — the sweet spot
    n_kv_heads == 1           -> Multi-Query Attention (MQA)   — least K/V memory

Fewer key/value heads means a smaller key/value cache during autoregressive
generation — the dominant memory cost of serving an LLM at long context.

Shapes:
    q     : (batch, n_heads,    seq, head_dim)
    k, v  : (batch, n_kv_heads, seq, head_dim)  -> each repeated to n_heads
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Size in bytes of the K/V cache for autoregressive decoding.

    The cache stores both K and V for every layer and every position:
        2 (K and V) × n_layers × batch × seq_len × n_kv_heads × head_dim × bytes

    Only ``n_kv_heads`` enters — this is exactly why GQA/MQA shrink the cache.
    """
    return 2 * n_layers * batch_size * seq_len * n_kv_heads * head_dim * dtype_bytes


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, rope: RotaryEmbedding
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # query heads per K/V head (group size)
        self.head_dim = dim // n_heads
        self.rope = rope

        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand (b, n_kv_heads, s, d) -> (b, n_heads, s, d) by repeating groups."""
        if self.n_rep == 1:
            return x
        b, kvh, s, d = x.shape
        return (
            x[:, :, None, :, :]
            .expand(b, kvh, self.n_rep, s, d)
            .reshape(b, kvh * self.n_rep, s, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.w_q(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Share each K/V head across its group of query heads.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(out)

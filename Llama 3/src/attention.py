"""Grouped-Query Attention (GQA) with rotary positions — Llama 3, §3.1.

Standard multi-head attention keeps one key/value head per query head. Llama 3
uses **Grouped-Query Attention** (Ainslie et al. 2023): it keeps ``n_heads``
query heads but only ``n_kv_heads`` key/value heads, and each K/V head is shared
by a *group* of query heads. This shrinks the KV cache (and the K/V projections)
by ``n_heads / n_kv_heads`` with negligible quality loss, which is what makes
long-context inference affordable.

Setting ``n_kv_heads == n_heads`` recovers ordinary multi-head attention;
``n_kv_heads == 1`` is multi-query attention. Llama 3 uses 8 K/V heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand K/V heads so each is reused by a group of ``n_rep`` query heads.

    x: (batch, n_kv_heads, seq, head_dim) -> (batch, n_kv_heads*n_rep, seq, head_dim)
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int, rope: RotaryEmbedding
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.rope = rope

        # Q keeps all heads; K/V only project to the (smaller) number of KV heads.
        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.w_q(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Share each K/V head across its group of query heads.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(out)

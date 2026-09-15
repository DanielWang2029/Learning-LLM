"""Multi-Query Attention (MQA) with rotary positions.

PaLM uses multi-query attention (PaLM paper, Section 2, "Multi-Query
Attention"): the queries still have ``n_heads`` separate projections, but the
keys and values share a *single* head that is broadcast across all query heads.
This barely changes training quality but dramatically shrinks the key/value
cache at autoregressive decoding time (fewer K/V vectors to store and reload),
which is the main cost of generation.

Shapes:
    x       : (batch, seq, dim)
    q       : (batch, n_heads, seq, head_dim)
    k, v    : (batch, 1,       seq, head_dim)   <- single shared K/V head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


class MultiQueryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, rope: RotaryEmbedding) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = rope

        # Queries: one projection per head. Keys/values: a single shared head.
        self.w_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(dim, self.head_dim, bias=False)
        self.w_v = nn.Linear(dim, self.head_dim, bias=False)
        self.w_o = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        q = self.w_q(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        # Single K/V head -> shape (batch, 1, seq, head_dim).
        k = self.w_k(x).view(batch, seq, 1, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq, 1, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # Broadcast the shared K/V head across all query heads (no extra memory
        # thanks to expand); this is what makes it "multi-query".
        k = k.expand(batch, self.n_heads, seq, self.head_dim)
        v = v.expand(batch, self.n_heads, seq, self.head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # (batch, n_heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(out)

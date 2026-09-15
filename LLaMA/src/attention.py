"""Causal multi-head self-attention with rotary positions.

LLaMA uses standard multi-head attention (one key/value head per query head),
with RoPE applied to the queries and keys at every layer and a causal mask so
each position attends only to itself and the past (LLaMA paper, Section 2.2).

A ``use_rope`` flag lets the demo ablate rotary positions and show that,
without any position information, the model can no longer solve the copy task.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary


class Attention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, rope: RotaryEmbedding, use_rope: bool = True
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope = rope
        self.use_rope = use_rope

        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)
        self.w_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        shape = (batch, seq, self.n_heads, self.head_dim)
        q = self.w_q(x).view(shape).transpose(1, 2)  # (b, h, s, d)
        k = self.w_k(x).view(shape).transpose(1, 2)
        v = self.w_v(x).view(shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(seq, x.device)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.w_o(out)

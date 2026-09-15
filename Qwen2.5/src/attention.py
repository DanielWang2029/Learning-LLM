"""Qwen2.5 attention: Grouped-Query Attention WITH a bias on Q, K, V.

The most visible architectural quirk of the Qwen2 / Qwen2.5 family is that,
against the modern trend of dropping all biases, it *adds a bias term to the
query, key and value projections* (Qwen2.5 Technical Report, arXiv:2412.15115,
"Architecture"; Qwen2 report §2.1). The output projection stays bias-free. The
authors report this QKV bias improves extrapolation.

The rest is standard: Grouped-Query Attention (fewer K/V heads than query heads)
with RoPE applied to Q and K. A ``qkv_bias`` flag lets the demo ablate the bias
and confirm the feature is really present and doing something.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RotaryEmbedding, apply_rotary


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


class QwenAttention(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, n_kv_heads: int,
        rope: RotaryEmbedding, qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.rope = rope
        self.qkv_bias = qkv_bias

        # Distinctive Qwen choice: bias=True on Q/K/V, bias=False on the output.
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(s, x.device)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.o_proj(out)

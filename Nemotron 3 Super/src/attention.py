"""Standard causal multi-head self-attention (the quadratic layer).

In the hybrid stack this is the layer that provides *global* token interaction —
every position attends to every earlier position. It costs O(L²), which is why
Nemotron 3 Super uses only a few of these and lets the linear SSM layers carry
most of the sequence mixing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, l, self.h, self.dk).transpose(1, 2)  # (B,h,L,dk)
        k = k.view(b, l, self.h, self.dk).transpose(1, 2)
        v = v.view(b, l, self.h, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,L,L)
        causal = torch.tril(torch.ones(l, l, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                            # (B,h,L,dk)
        out = out.transpose(1, 2).reshape(b, l, d)
        return self.proj(out)
